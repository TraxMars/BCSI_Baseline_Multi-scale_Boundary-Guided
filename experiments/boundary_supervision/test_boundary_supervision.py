"""Isolated gradient audits and regression against the pre-change MT trainer.

Run from the repository root. A nonzero exit code MUST block formal training.
"""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

from model.vnet import VNet
from trainer import Trainer, update_ema_variables
from utils.boundary_supervision import (
    MultiScaleBoundarySupervision, encoder_stage_parameters, boundary_target,
    binary_boundary_loss, local_boundary_gradients, add_local_gradients,
)
from utils.utils import boundary_f1_percase


REPORT = {'gradient_matrices': {}, 'shape_audits': {}}


def config(mode='none', **overrides):
    values = dict(in_channels=1, num_classes=2, device='cpu', base_lr=0.01,
                  max_iterations=30000, labeled_bs=2, ema_decay=0.9,
                  consistency=0.1, consistency_rampup=200, boundary_mode=mode,
                  boundary_weight=0.1, guidance_mode='none')
    values.update(overrides)
    return SimpleNamespace(**values)


def sample(size=32, batch=4):
    labels = torch.zeros(batch, size, size, size, dtype=torch.long)
    labels[:, size // 4:3 * size // 4, size // 4:3 * size // 4,
           size // 4:3 * size // 4] = 1
    return {'image': torch.randn(batch, 1, size, size, size), 'label': labels}


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_target_matches_independent_morphology(self):
        masks = [np.zeros((8, 10, 12)), np.ones((8, 10, 12))]
        cube = np.zeros((8, 10, 12))
        cube[2:6, 3:7, 4:8] = 1
        masks += [cube, np.random.RandomState(42).randint(0, 2, (8, 10, 12))]
        for array in masks:
            labels = torch.tensor(array)[None]
            for size in ((8, 10, 12), (4, 5, 6), (2, 2, 2)):
                boundary, mask = boundary_target(labels, size)
                resized = array[np.ix_(*[np.floor(np.arange(dst) * src / dst).astype(int)
                                        for src, dst in zip(array.shape, size)])]
                np.testing.assert_array_equal(mask[0, 0].numpy(), resized)
                expected = (ndimage.maximum_filter(resized, size=3, mode='nearest') -
                            ndimage.minimum_filter(resized, size=3, mode='nearest'))
                np.testing.assert_array_equal(boundary[0, 0].numpy(), expected)
                self.assertTrue(set(boundary.unique().tolist()) <= {0, 1})
                self.assertFalse(boundary.requires_grad)
        with self.assertRaises(ValueError):
            boundary_target(torch.zeros(1, 2, 8, 8, 8), (4, 4, 4))

    def test_loss_3d_formula_and_extreme_logits(self):
        target = torch.randint(0, 2, (2, 1, 4, 5, 6)).float()
        logits = torch.randn_like(target, requires_grad=True)
        dice = []
        for pred, truth in zip(logits.sigmoid(), target):
            dice.append(1 - (2 * (pred * truth).sum() + 1e-5) /
                        (pred.sum() + truth.sum() + 1e-5))
        expected = 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * torch.stack(dice).mean()
        torch.testing.assert_close(binary_boundary_loss(logits, target), expected)
        for value in (-1000., 1000.):
            for truth in (torch.zeros_like(target), torch.ones_like(target)):
                extreme = torch.full_like(target, value, requires_grad=True)
                loss = binary_boundary_loss(extreme, truth)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_all_five_isolated_gradient_scopes(self):
        # Use the actual default 16/32/64/128/256 VNet, with dropout disabled
        # only for this isolated loss audit. Training tests retain dropout.
        model = VNet(has_dropout=False)
        boundary = MultiScaleBoundarySupervision()
        stages = encoder_stage_parameters(model.encoder)
        self.assertEqual(len({id(p) for group in stages for p in group}),
                         len(list(model.encoder.parameters())))
        batch = sample(batch=2)
        for mode in ('deep', 'local'):
            matrix = []
            for index in range(5):
                model.zero_grad(set_to_none=True)
                boundary.zero_grad(set_to_none=True)
                features = model.encoder(batch['image'])
                result = boundary(features, batch['label'])
                feature_gradient, = torch.autograd.grad(result.losses[index],
                    result.features[index], retain_graph=True)
                self.assertTrue(torch.isfinite(feature_gradient).all())
                self.assertGreater(feature_gradient.abs().sum().item(), 0)
                if mode == 'deep':
                    (result.losses[index] * (0.1 / 5)).backward()
                else:
                    grads = local_boundary_gradients(result, stages, boundary.heads,
                        boundary_weight=0.1, stage_indices=[index])
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    add_local_gradients(grads)
                present = []
                for stage, parameters in enumerate(stages):
                    expected = stage <= index if mode == 'deep' else stage == index
                    self.assertEqual(any(p.grad is not None for p in parameters), expected,
                                     f'{mode}: loss {index + 1} -> E{stage + 1}')
                    if expected:
                        self.assertTrue(all(p.grad is not None for p in parameters))
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters), 0)
                    present.append(expected)
                for head_index, head in enumerate(boundary.heads):
                    self.assertEqual(any(p.grad is not None for p in head.parameters()),
                                     head_index == index)
                self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in
                    list(model.parameters()) + list(boundary.parameters()) if p.grad is not None))
                matrix.append(present)
            REPORT['gradient_matrices'][mode] = matrix

    def test_local_sum_matches_independent_derivatives(self):
        model = VNet(n_filters=2, normalization='none', has_dropout=False)
        heads = MultiScaleBoundarySupervision((2, 4, 8, 16, 32))
        stages = encoder_stage_parameters(model.encoder)
        batch = sample(batch=2)
        features = model.encoder(batch['image'])
        logits = model.decoder(features)
        main = F.cross_entropy(logits, batch['label'])
        result = heads(features, batch['label'])
        local = local_boundary_gradients(result, stages, heads.heads, 0.1)
        all_parameters = list(model.parameters()) + list(heads.parameters())
        expected = torch.autograd.grad(main, all_parameters, allow_unused=True, retain_graph=True)
        expected = {id(p): torch.zeros_like(p) if g is None else g.clone()
                    for p, g in zip(all_parameters, expected)}
        for index in range(5):
            parameters = tuple(stages[index]) + tuple(heads.heads[index].parameters())
            values = torch.autograd.grad(result.losses[index], parameters, retain_graph=True)
            for p, g in zip(parameters, values):
                expected[id(p)] += 0.1 / 5 * g
        main.backward()
        add_local_gradients(local)
        for p in all_parameters:
            torch.testing.assert_close(p.grad, expected[id(p)], rtol=1e-5, atol=1e-7)

    def test_same_initialization_rng_and_forward_in_both_modes(self):
        trainers, rngs = [], []
        for mode in ('none', 'deep', 'local'):
            torch.manual_seed(42)
            trainers.append(Trainer(config(mode)))
            rngs.append(torch.get_rng_state())
        for other in trainers[1:]:
            self.assert_state_equal(trainers[0].model.state_dict(), other.model.state_dict())
            self.assert_state_equal(trainers[0].ema_model.state_dict(), other.ema_model.state_dict())
        self.assertTrue(torch.equal(rngs[0], rngs[1]) and torch.equal(rngs[1], rngs[2]))
        batch = sample()
        outputs = []
        for trainer in trainers[1:]:
            torch.manual_seed(700)
            features = trainer.model.encoder(batch['image'])
            logits = trainer.model.decoder(features)
            result = trainer.boundary(features, batch['label'][:2])
            outputs.append((logits, result))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        for a, b in zip(outputs[0][1].logits, outputs[1][1].logits):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(outputs[0][1].losses, outputs[1][1].losses):
            self.assertTrue(torch.equal(a, b))
        torch.testing.assert_close(outputs[0][1].total_loss,
                                   torch.stack(outputs[0][1].losses).sum() / 5)
        # Unlabeled label changes cannot affect targets or boundary losses.
        features = trainers[1].model.encoder(batch['image'])
        result = trainers[1].boundary(features, batch['label'][:2])
        gradient, = torch.autograd.grad(result.losses[2], features[2], retain_graph=True)
        self.assertEqual(gradient[2:].abs().sum().item(), 0)

    def test_none_matches_pre_change_mt_for_three_steps(self):
        path = Path(__file__).parent / 'reference' / 'trainer_before.py'
        spec = importlib.util.spec_from_file_location('pre_boundary_trainer', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        args = config()
        del args.boundary_mode  # Legacy external callers need no new attributes.
        torch.manual_seed(42)
        before = original.Trainer(args)
        torch.manual_seed(42)
        after = Trainer(args)
        batch = sample()
        with tempfile.TemporaryDirectory() as directory:
            for iteration in range(3):
                torch.manual_seed(100 + iteration)
                before.train(batch, iteration, directory)
                torch.manual_seed(100 + iteration)
                after.train(batch, iteration, directory)
                self.assert_state_equal(before.model.state_dict(), after.model.state_dict())
                self.assert_state_equal(before.ema_model.state_dict(), after.ema_model.state_dict())
                self.assertEqual(before.scheduler.state_dict(), after.scheduler.state_dict())
                for p, q in zip(before.model.parameters(), after.model.parameters()):
                    self.assertTrue(torch.equal(before.optimizer.state[p]['momentum_buffer'],
                                                after.optimizer.state[q]['momentum_buffer']))
            self.assertEqual(list(Path(directory).iterdir()), [])
        before.model.eval()
        after.model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(before.model(batch['image']), after.model(batch['image'])))

    def test_student_teacher_single_updates_both_modes(self):
        batch = sample()
        for mode in ('deep', 'local'):
            torch.manual_seed(42)
            trainer = Trainer(config(mode))
            student_before = [p.detach().clone() for p in trainer.model.parameters()]
            teacher_before = [p.detach().clone() for p in trainer.ema_model.parameters()]
            heads_before = [p.detach().clone() for p in trainer.boundary.parameters()]
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
                    row = trainer.train(batch, 17, directory)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(trainer.scheduler.last_epoch, 1)
                for old, student, teacher in zip(teacher_before, trainer.model.parameters(),
                                                 trainer.ema_model.parameters()):
                    torch.testing.assert_close(teacher, old * 0.9 + student * 0.1)
                for group in trainer.stage_parameters:
                    self.assertTrue(any(p.grad.abs().sum() > 0 for p in group))
                self.assertTrue(all(p.grad is None for p in trainer.ema_model.parameters()))
                self.assertTrue(any(not torch.equal(p, old) for p, old in
                                    zip(trainer.model.parameters(), student_before)))
                self.assertTrue(all(not torch.equal(p, old) for p, old in
                                    zip(trainer.boundary.parameters(), heads_before)))
                for key, value in row.items():
                    if isinstance(value, (float, int)):
                        self.assertTrue(np.isfinite(value), key)
                self.assertAlmostEqual(row['total_loss'], row['supervised_loss'] +
                    row['consistency_weight'] * row['consistency_loss'] +
                    0.1 * row['total_boundary_loss'], places=6)
                with open(Path(directory) / 'training_metrics.csv') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 1)
                # Validation metric route and best plain-VNet Teacher save.
                metrics = dict(dice=0.5, hd95=2., asd=1., iou=1/3, boundary_f1_1voxel=0.6)
                with patch('trainer.test_calculate_metric', return_value=metrics) as metric:
                    trainer.test(directory, 18)
                self.assertTrue(metric.call_args.kwargs['return_details'])
                loaded = VNet()
                loaded.load_state_dict(torch.load(Path(directory) / 'Model_iter_18.pth',
                                                  weights_only=True), strict=True)
                self.assert_state_equal(loaded.state_dict(), trainer.ema_model.state_dict())
                state = torch.load(Path(directory) / 'boundary_training_state_latest.pth',
                                   weights_only=False)
                self.assertEqual(state['iteration'], 18)
                self.assert_state_equal(state['boundary'], trainer.boundary.state_dict())
                self.assertTrue((Path(directory) / 'BoundaryHeads_iter_18.pth').is_file())

    def test_real_default_shapes_96_and_la(self):
        model = VNet().eval()
        heads = MultiScaleBoundarySupervision().eval()
        for shape in ((96, 96, 96), (112, 112, 80)):
            labels = torch.zeros(1, *shape, dtype=torch.long)
            labels[:, shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4,
                   shape[2]//4:3*shape[2]//4] = 1
            with torch.no_grad():
                features = model.encoder(torch.randn(1, 1, *shape))
                result = heads(features, labels)
            for index, (feature, logits, target) in enumerate(zip(features, result.logits, result.targets)):
                self.assertEqual(feature.shape[1], 16 * 2**index)
                self.assertEqual(logits.shape, (1, 1, *(n//2**index for n in shape)))
                self.assertEqual(logits.shape, target.shape)
            REPORT['shape_audits']['x'.join(map(str, shape))] = result.diagnostics

    def test_guidance_rejected_and_surface_metric(self):
        with self.assertRaises(ValueError):
            Trainer(config('deep', guidance_mode='bgs'))
        with self.assertRaises(ValueError):
            Trainer(config('local', boundary_weight=float('nan')))
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[2:6, 2:6, 2:6] = True
        self.assertEqual(boundary_f1_percase(mask, mask), 1)
        self.assertEqual(boundary_f1_percase(mask, np.zeros_like(mask)), 0)
        self.assertEqual(boundary_f1_percase(np.zeros_like(mask), np.zeros_like(mask)), 1)
        self.assertEqual(boundary_f1_percase(mask, np.roll(mask, 6, axis=0)), 0)

    def test_original_metric_return_and_extended_metrics(self):
        import prediction
        import h5py
        truth = np.zeros((16, 16, 16), dtype=np.uint8)
        truth[4:12, 4:12, 4:12] = 1
        predicted = truth.copy()
        predicted[11] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.list').write_text('case_0\n')
            (root / 'case_0').mkdir()
            with h5py.File(root / 'case_0' / 'mri_norm2.h5', 'w') as handle:
                handle['image'] = truth.astype(np.float32)
                handle['label'] = truth
            args = SimpleNamespace(data_path=directory, dataset='LA', patch_size=[16]*3, num_classes=2)
            with patch('prediction.test_single_case', return_value=(predicted, None)):
                old = prediction.test_calculate_metric(args, None, val=True)
                details = prediction.test_calculate_metric(args, None, val=True, return_details=True)
            self.assertEqual(old, (details['dice'], details['hd95']))
            self.assertTrue(all(np.isfinite(value) for value in details.values()))
            self.assertGreater(details['asd'], 0)
        checkpoint = ROOT / 'Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth'
        if checkpoint.is_file():
            VNet().load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), passed=result.wasSuccessful(),
                  torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                  note='Validation persistence uses mocked metrics, not an evaluated accuracy result.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(REPORT, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
