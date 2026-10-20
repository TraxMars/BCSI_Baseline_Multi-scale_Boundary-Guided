"""Independent method and gradient checks for BFA A2/B2; no training results.

Run from the repository root with SSL29 Python. Validation persistence is mocked
in synthetic update tests; no accuracy claim is made by these tests.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F

from model.vnet import VNet
from trainer import Trainer
from utils.bfa_supervision import (
    BFAEncoderBoundarySupervision, LaplacianBoundary, feature_boundary_gradient,
)
from utils.boundary_supervision import (
    MultiScaleBoundarySupervision, add_local_gradients, binary_boundary_loss,
    encoder_stage_parameters, local_boundary_gradients,
)
from utils.gradient_diagnostics import boundary_gradient_diagnostics


REPORT = {'gradient_matrices': {}, 'shape_audits': {}}
ORIGINAL_ROOT = ROOT.parent / 'BCSI_Baseline_Multi-scale_Boundary-Guided'
if not ORIGINAL_ROOT.is_dir():
    ORIGINAL_ROOT = ROOT


def config(mode='none', **overrides):
    values = dict(in_channels=1, num_classes=2, device='cpu', base_lr=0.01,
                  max_iterations=30000, labeled_bs=2, batch_size=4, ema_decay=0.9,
                  consistency=0.1, consistency_rampup=200, boundary_mode=mode,
                  boundary_weight=0.1, guidance_mode='none', auxiliary_mode='none',
                  decoder_aux_weight=0.1, gradient_diagnostic_interval=0,
                  gradient_cosine_threshold=-0.2, gradient_ratio_threshold=0.1)
    values.update(overrides)
    return SimpleNamespace(**values)


def sample(size=32, batch=4):
    labels = torch.zeros(batch, size, size, size, dtype=torch.long)
    labels[:, size//4:3*size//4, size//4:3*size//4,
           size//4:3*size//4] = 1
    return {'image': torch.randn(batch, 1, size, size, size), 'label': labels}


def reference_trainer():
    path = ROOT / 'experiments/boundary_supervision/reference/trainer_before.py'
    spec = importlib.util.spec_from_file_location('original_mt_for_refactor', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original.Trainer


def tensor_state_equal(test, first, second):
    test.assertEqual(first.keys(), second.keys())
    for key in first:
        test.assertTrue(torch.equal(first[key], second[key]), key)


def toy_diagnostic_graph(main_kind='normal', boundary_kind='normal'):
    """Analytic graph with cumulative boundary dependence and no forward BN."""
    parameters = [nn.Parameter(torch.tensor(float(i+1))) for i in range(5)]
    heads = nn.ModuleList(nn.Linear(1, 1, bias=False) for _ in range(4))
    for head in heads:
        nn.init.ones_(head.weight)
    if main_kind == 'missing':
        main = torch.tensor(2., requires_grad=True)
    elif main_kind == 'zero':
        main = sum(p * 0 for p in parameters)
    elif main_kind == 'nonfinite':
        main = sum(parameters) * float('nan')
    else:
        main = -sum(parameters)
    if boundary_kind == 'missing':
        losses = [torch.tensor(1., requires_grad=True) for _ in range(4)]
    elif boundary_kind == 'zero':
        losses = [sum(parameters[:i+1]) * 0 * head.weight.sum()
                  for i, head in enumerate(heads)]
    else:
        losses = [sum(parameters[:i+1]) * head.weight.sum()
                  for i, head in enumerate(heads)]
    result = SimpleNamespace(losses=losses, total_loss=torch.stack(losses).mean())
    return parameters, heads, main, result


class BoundaryRefactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_none_matches_original_mt_three_updates_and_inference(self):
        args = config()
        torch.manual_seed(42)
        before = reference_trainer()(args)
        torch.manual_seed(42)
        after = Trainer(args)
        self.assertIsNone(after.boundary)
        batch = sample()
        with tempfile.TemporaryDirectory() as directory:
            for iteration in range(3):
                torch.manual_seed(100 + iteration)
                before.train(batch, iteration, directory)
                torch.manual_seed(100 + iteration)
                after.train(batch, iteration, directory)
                tensor_state_equal(self, before.model.state_dict(), after.model.state_dict())
                tensor_state_equal(self, before.ema_model.state_dict(), after.ema_model.state_dict())
                self.assertEqual(before.scheduler.state_dict(), after.scheduler.state_dict())
                for p, q in zip(before.model.parameters(), after.model.parameters()):
                    self.assertTrue(torch.equal(before.optimizer.state[p]['momentum_buffer'],
                                                after.optimizer.state[q]['momentum_buffer']))
            self.assertEqual(list(Path(directory).iterdir()), [])
        before.model.eval()
        after.model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(before.model(batch['image']), after.model(batch['image'])))

    def test_original_checkpoint_strict_load_and_final_tensor_inference(self):
        checkpoint = ORIGINAL_ROOT / 'Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth'
        if not checkpoint.is_file():
            self.skipTest(f'Original checkpoint unavailable: {checkpoint}')
        model = VNet().eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            output = model(torch.randn(1, 1, 32, 32, 32))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 2, 32, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        REPORT['original_checkpoint'] = str(checkpoint)

    def test_legacy_modes_keep_five_linear_heads_and_gradient_semantics(self):
        # Legacy numeric semantics are independently covered by its unchanged
        # ten-test suite. This verifies factory dispatch has not remapped modes.
        for mode in ('deep', 'local'):
            trainer = Trainer(config(mode))
            self.assertIs(type(trainer.boundary), MultiScaleBoundarySupervision)
            self.assertEqual(len(trainer.boundary.heads), 5)
            self.assertTrue(all(type(h) is nn.Conv3d and h.kernel_size == (1, 1, 1)
                                for h in trainer.boundary.heads))

    def test_experiments_are_independent_and_invalid_configuration_is_rejected(self):
        for mode in ('bfa_deep', 'bfa_local'):
            with self.assertRaises(ValueError):
                Trainer(config(mode, auxiliary_mode='decoder_deep_seg'))
            with self.assertRaises(ValueError):
                Trainer(config(mode, guidance_mode='bgs'))
            with self.assertRaises(ValueError):
                Trainer(config(mode, boundary_weight=float('nan')))
        with self.assertRaises(ValueError):
            Trainer(config('bfa_deep', gradient_diagnostic_interval=-1))
        with self.assertRaises(ValueError):
            Trainer(config('bfa_local', gradient_cosine_threshold=float('nan')))
        with self.assertRaises(ValueError):
            Trainer(config('bfa_local', gradient_ratio_threshold=-1))

    def test_original_resolution_laplacian_matches_scipy_and_padding_edges(self):
        masks = [np.zeros((7, 8, 9), dtype=np.float32),
                 np.ones((7, 8, 9), dtype=np.float32)]
        inside = np.zeros_like(masks[0]); inside[2:5, 2:6, 2:7] = 1
        touching = inside.copy(); touching[:3, 2:6, 2:7] = 1
        masks += [inside, touching, np.random.RandomState(9).randint(0, 2, (7, 8, 9)).astype(np.float32)]
        kernel = -np.ones((3, 3, 3), dtype=np.float32); kernel[1, 1, 1] = 26
        for padding in ('zero', 'replicate'):
            module = LaplacianBoundary(padding)
            self.assertEqual(dict(module.named_parameters()), {})
            self.assertIn('kernel', dict(module.named_buffers()))
            for mask in masks:
                labels = torch.from_numpy(mask).unsqueeze(0)
                observed = module(labels)
                response = ndimage.convolve(mask, kernel,
                    mode='constant' if padding == 'zero' else 'nearest', cval=0)
                expected = np.maximum(response, 0) > .1
                np.testing.assert_array_equal(observed[0, 0].numpy(), expected)
                self.assertTrue(torch.equal(observed, module(labels.unsqueeze(1))))
                self.assertFalse(observed.requires_grad)
                self.assertEqual(observed.device, labels.device)
                self.assertFalse((observed.bool() & ~labels.unsqueeze(1).bool()).any())
        zero = LaplacianBoundary('zero')(torch.ones(1, 7, 8, 9))
        replicate = LaplacianBoundary('replicate')(torch.ones(1, 7, 8, 9))
        self.assertEqual(zero.sum().item(), 7*8*9 - 5*6*7)
        self.assertEqual(replicate.sum().item(), 0)
        self.assertTrue(torch.equal(LaplacianBoundary('zero')(torch.from_numpy(inside)[None]),
                                    LaplacianBoundary('replicate')(torch.from_numpy(inside)[None])))
        difference = LaplacianBoundary('zero')(torch.from_numpy(touching)[None]) != LaplacianBoundary('replicate')(torch.from_numpy(touching)[None])
        self.assertFalse(difference[..., 1:-1, 1:-1, 1:-1].any())
        with self.assertRaises(ValueError):
            LaplacianBoundary('invalid')
        with self.assertRaises(ValueError):
            LaplacianBoundary()(torch.zeros(1, 2, 3, 4, 5))

    def test_four_heads_shapes_shared_full_resolution_target_and_formula(self):
        model = VNet(has_dropout=False).eval()
        heads = BFAEncoderBoundarySupervision().eval()
        self.assertEqual(len(heads.heads), 4)
        for index, head in enumerate(heads.heads):
            convs = [m for m in head.modules() if isinstance(m, nn.Conv3d)]
            self.assertEqual(len(convs), 2)
            self.assertEqual((convs[0].in_channels, convs[0].out_channels, convs[0].kernel_size),
                             (16*2**index, 16, (3, 3, 3)))
            self.assertEqual((convs[1].in_channels, convs[1].out_channels, convs[1].kernel_size),
                             (16, 1, (1, 1, 1)))
            self.assertTrue(any(isinstance(m, nn.BatchNorm3d) for m in head.modules()))
        for shape in ((32, 32, 32), (112, 112, 80)):
            labels = torch.zeros(1, *shape, dtype=torch.long)
            labels[:, shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4,
                   shape[2]//4:3*shape[2]//4] = 1
            with torch.no_grad():
                features = model.encoder(torch.randn(1, 1, *shape))
                result = heads(features, labels)
            self.assertEqual(len(result.features), 5)
            self.assertEqual(len(result.losses), 4)
            self.assertTrue(all(t is result.target for t in result.targets))
            self.assertTrue(torch.equal(result.target, LaplacianBoundary()(labels)))
            for i, (native, up, target, loss) in enumerate(zip(result.logits,
                    result.upsampled_logits, result.targets, result.losses)):
                self.assertEqual(tuple(native.shape), (1, 1, *(s//2**i for s in shape)))
                self.assertEqual(tuple(up.shape), (1, 1, *shape))
                expected_up = F.interpolate(native, size=shape, mode='trilinear', align_corners=True)
                torch.testing.assert_close(up, expected_up, rtol=0, atol=0)
                probability = up.sigmoid()
                dice_loss = (1 - (2*(probability*target).sum((2,3,4)) + 1.) /
                    (probability.sum((2,3,4)) + target.sum((2,3,4)) + 1.)).mean()
                expected = .5*F.binary_cross_entropy_with_logits(up, target) + .5*dice_loss
                torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(result.total_loss, torch.stack(result.losses).mean())
            REPORT['shape_audits']['x'.join(map(str, shape))] = result.diagnostics

    def test_labeled_prefix_enters_head_batchnorm_and_feature_gradient_interface(self):
        channels = (2, 4, 8, 16)
        heads = BFAEncoderBoundarySupervision(channels)
        clone = copy.deepcopy(heads)
        features = [torch.randn(4, c, *(16//2**i for _ in range(3)), requires_grad=True)
                    for i, c in enumerate((*channels, 32))]
        modified = [torch.cat((x[:2].detach(), x[2:].detach()+1000)) for x in features]
        labels = sample(size=16, batch=2)['label']
        first = heads(features, labels)
        second = clone(modified, labels)
        for a, b in zip(first.logits, second.logits):
            self.assertTrue(torch.equal(a, b))
        tensor_state_equal(self, heads.state_dict(), clone.state_dict())
        for i in range(4):
            gradient = feature_boundary_gradient(first, i)
            self.assertEqual(gradient.shape, features[i].shape)
            self.assertGreater(gradient[:2].abs().sum().item(), 0)
            self.assertEqual(gradient[2:].abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(all(p.grad is None for p in heads.parameters()))
        with self.assertRaises(IndexError):
            feature_boundary_gradient(first, 4)

    def test_extreme_bce_dice_losses_and_gradients_are_finite(self):
        for value in (-1000., 1000.):
            for truth in (0., 1.):
                logits = torch.full((2, 1, 5, 6, 7), value, requires_grad=True)
                target = torch.full_like(logits, truth)
                loss = binary_boundary_loss(logits, target, epsilon=1.)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_four_isolated_gradient_scopes_with_fifth_stage_excluded(self):
        model = VNet(has_dropout=False)
        heads = BFAEncoderBoundarySupervision()
        stages = encoder_stage_parameters(model.encoder)
        ids = [id(p) for group in stages for p in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.encoder.parameters()})
        batch = sample(batch=2)
        for mode in ('bfa_deep', 'bfa_local'):
            matrix = []
            for i in range(4):
                model.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
                features = model.encoder(batch['image'])
                result = heads(features, batch['label'])
                if mode == 'bfa_deep':
                    (.1/4 * result.losses[i]).backward()
                else:
                    gradients = local_boundary_gradients(result, stages, heads.heads, .1,
                                                         stage_indices=[i])
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    add_local_gradients(gradients)
                present = []
                for j, parameters in enumerate(stages):
                    expected = j <= i if mode == 'bfa_deep' else j == i
                    observed = any(p.grad is not None for p in parameters)
                    self.assertEqual(observed, expected, f'{mode}: loss{i+1}->E{j+1}')
                    if expected:
                        self.assertTrue(all(p.grad is not None for p in parameters))
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters), 0)
                    present.append(observed)
                for j, head in enumerate(heads.heads):
                    self.assertEqual(any(p.grad is not None for p in head.parameters()), j == i)
                self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in
                    list(model.parameters())+list(heads.parameters()) if p.grad is not None))
                matrix.append(present)
            REPORT['gradient_matrices'][mode] = matrix

    def test_local_routing_matches_independent_autograd_and_full_mt_sum(self):
        model = VNet(n_filters=2, normalization='none', has_dropout=False)
        heads = BFAEncoderBoundarySupervision((2, 4, 8, 16))
        stages = encoder_stage_parameters(model.encoder)
        batch = sample(batch=2)
        features = model.encoder(batch['image'])
        main = F.cross_entropy(model.decoder(features), batch['label'])
        result = heads(features, batch['label'])
        local = local_boundary_gradients(result, stages, heads.heads, .1)
        parameters = list(model.parameters()) + list(heads.parameters())
        main_grads = torch.autograd.grad(main, parameters, allow_unused=True, retain_graph=True)
        expected = {id(p): torch.zeros_like(p) if g is None else g.clone()
                    for p, g in zip(parameters, main_grads)}
        for i in range(4):
            group = tuple(stages[i]) + tuple(heads.heads[i].parameters())
            direct = torch.autograd.grad(result.losses[i], group, retain_graph=True)
            for p, g in zip(group, direct):
                expected[id(p)] += .1/4 * g
        main.backward(); add_local_gradients(local)
        for p in parameters:
            torch.testing.assert_close(p.grad, expected[id(p)], rtol=1e-5, atol=1e-7)

    def test_modes_share_vnet_rng_heads_forward_targets_and_loss_initialization(self):
        trainers, rngs = [], []
        for mode in ('none', 'bfa_deep', 'bfa_local'):
            torch.manual_seed(42)
            trainers.append(Trainer(config(mode)))
            rngs.append(torch.get_rng_state().clone())
        for other in trainers[1:]:
            tensor_state_equal(self, trainers[0].model.state_dict(), other.model.state_dict())
            tensor_state_equal(self, trainers[0].ema_model.state_dict(), other.ema_model.state_dict())
        self.assertTrue(all(torch.equal(rngs[0], other) for other in rngs[1:]))
        tensor_state_equal(self, trainers[1].boundary.state_dict(), trainers[2].boundary.state_dict())
        batch = sample()
        outputs = []
        for trainer in trainers[1:]:
            torch.manual_seed(700)
            features = trainer.model.encoder(batch['image'])
            logits = trainer.model.decoder(features)
            outputs.append((logits, trainer.boundary(features, batch['label'][:2])))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        for attr in ('logits', 'upsampled_logits', 'targets', 'losses'):
            for a, b in zip(getattr(outputs[0][1], attr), getattr(outputs[1][1], attr)):
                self.assertTrue(torch.equal(a, b), attr)

    def test_student_update_single_optimizer_scheduler_and_original_ema(self):
        import trainer as trainer_module
        batch = sample()
        for mode in ('bfa_deep', 'bfa_local'):
            trainer = Trainer(config(mode))
            old_student = [p.detach().clone() for p in trainer.model.parameters()]
            old_teacher = [p.detach().clone() for p in trainer.ema_model.parameters()]
            old_heads = [p.detach().clone() for p in trainer.boundary.parameters()]
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step, \
                     patch('trainer.update_ema_variables', wraps=trainer_module.update_ema_variables) as ema:
                    row = trainer.train(batch, 17, directory)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(ema.call_count, 1)
                self.assertEqual(trainer.scheduler.last_epoch, 1)
                for before, student, teacher in zip(old_teacher, trainer.model.parameters(),
                                                     trainer.ema_model.parameters()):
                    torch.testing.assert_close(teacher, before*.9+student*.1)
                self.assertTrue(all(p.grad is None for p in trainer.ema_model.parameters()))
                self.assertTrue(any(not torch.equal(a, b) for a,b in zip(old_student, trainer.model.parameters())))
                self.assertTrue(all(not torch.equal(a, b) for a,b in zip(old_heads, trainer.boundary.parameters())))
                self.assertAlmostEqual(row['total_loss'], row['supervised_loss'] +
                    row['consistency_weight']*row['consistency_loss']+.1*row['total_boundary_loss'], places=6)
                with open(Path(directory)/'training_metrics.csv') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 1)
                metrics = dict(dice=.5, hd95=2., asd=1., iou=1/3, boundary_f1_1voxel=.6)
                with patch('trainer.test_calculate_metric', return_value=metrics):
                    trainer.test(directory, 18)
                plain = VNet()
                plain.load_state_dict(torch.load(Path(directory)/'Model_iter_18.pth', weights_only=True), strict=True)
                tensor_state_equal(self, plain.state_dict(), trainer.ema_model.state_dict())

    def test_diagnostics_applied_vs_raw_and_do_not_touch_grads_rng_or_buffers(self):
        parameters, heads, main, result = toy_diagnostic_graph()
        stages = tuple((p,) for p in parameters)
        for p in parameters + list(heads.parameters()):
            p.grad = torch.full_like(p, 7.)
        grads_before = [p.grad.clone() for p in parameters + list(heads.parameters())]
        rng_before = torch.get_rng_state().clone()
        buffers_before = copy.deepcopy(heads.state_dict())
        local = boundary_gradient_diagnostics(main, result, stages, heads, .1, 'bfa_local')
        deep = boundary_gradient_diagnostics(main, result, stages, heads, .1, 'bfa_deep')
        self.assertEqual(len(local), 5)
        for i, (a, b) in enumerate(zip(local, deep)):
            self.assertAlmostEqual(a['main_norm'], 1.)
            self.assertAlmostEqual(a['boundary_norm'], .025 if i<4 else 0.)
            self.assertAlmostEqual(b['boundary_norm'], .025*(4-i) if i<4 else 0.)
            self.assertAlmostEqual(a['raw_boundary_norm'], b['boundary_norm'])
            if i<4:
                self.assertAlmostEqual(a['cosine'], -1.)
                self.assertAlmostEqual(a['ratio'], .025)
        for p, old in zip(parameters + list(heads.parameters()), grads_before):
            self.assertTrue(torch.equal(p.grad, old))
        tensor_state_equal(self, heads.state_dict(), buffers_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        REPORT['analytic_diagnostics'] = {'bfa_local': local, 'bfa_deep': deep}

    def test_diagnostic_zero_missing_and_nonfinite_have_explicit_status(self):
        for main_kind, boundary_kind in [('zero','normal'),('missing','normal'),
                                         ('normal','zero'),('normal','missing'),
                                         ('nonfinite','normal')]:
            parameters, heads, main, result = toy_diagnostic_graph(main_kind,boundary_kind)
            stages = tuple((p,) for p in parameters)
            rows = boundary_gradient_diagnostics(main,result,stages,heads,.1,'bfa_deep')
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row['status'] for row in rows))
            # Non-normal states must remain JSON serializable without NaN/Inf.
            json.dumps(rows, allow_nan=False)
            self.assertTrue(all(row['cosine'] is None for row in rows))
            self.assertTrue(all(p.grad is None for p in parameters+list(heads.parameters())))

    def test_diagnostics_enabled_do_not_change_actual_training_update(self):
        batch = sample()
        for mode in ('bfa_deep', 'bfa_local'):
            trainers, states = [], []
            for interval in (0, 1):
                torch.manual_seed(42)
                trainer = Trainer(config(mode,gradient_diagnostic_interval=interval))
                with tempfile.TemporaryDirectory() as directory:
                    for i in range(2):
                        torch.manual_seed(900+i)
                        trainer.train(batch, i, directory)
                    if interval:
                        files = list(Path(directory).glob('*gradient*'))
                        self.assertTrue(files, 'Diagnostics enabled but no output artifact')
                trainers.append(trainer); states.append(torch.get_rng_state().clone())
            for attr in ('model','ema_model','boundary'):
                tensor_state_equal(self,getattr(trainers[0],attr).state_dict(),
                                    getattr(trainers[1],attr).state_dict())
            self.assertTrue(torch.equal(states[0],states[1]))
            self.assertEqual(trainers[0].scheduler.state_dict(),trainers[1].scheduler.state_dict())
            for p,q in zip(trainers[0].optimizer.param_groups[0]['params'],
                           trainers[1].optimizer.param_groups[0]['params']):
                self.assertTrue(torch.equal(p.grad,q.grad))
                self.assertTrue(torch.equal(trainers[0].optimizer.state[p]['momentum_buffer'],
                                            trainers[1].optimizer.state[q]['momentum_buffer']))

    def test_validation_and_final_test_list_sources_are_explicit(self):
        import prediction
        truth = np.zeros((16,16,16),dtype=np.uint8); truth[4:12,4:12,4:12]=1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'test.list').write_text('actual_test_case\n')
            (root/'val.list').write_text('unused_val_case\n')
            (root/'actual_test_case').mkdir()
            with h5py.File(root/'actual_test_case/mri_norm2.h5','w') as f:
                f['image']=truth.astype(np.float32); f['label']=truth
            args=SimpleNamespace(data_path=directory,dataset='LA',patch_size=[16]*3,num_classes=2)
            with patch('prediction.test_single_case',return_value=(truth,None)), \
                 contextlib.redirect_stdout(io.StringIO()):
                val=prediction.test_calculate_metric(args,None,val=True,return_details=True)
                test=prediction.test_calculate_metric(args,None,val=False,return_details=True)
            self.assertEqual(val,test)
            self.assertEqual(val['dice'],1.)
        manifest=json.loads((ORIGINAL_ROOT/'Results/Boundary_A/seed_42/result_LA_10l/fold_0/data_split_manifest.json').read_text())
        train_ids=set(manifest['splits']['train']['ids']); test_ids=set(manifest['splits']['test']['ids'])
        self.assertEqual((len(train_ids),len(test_ids)),(80,20))
        self.assertFalse(train_ids&test_ids)
        current_files_verified = {}
        for name, split in manifest['splits'].items():
            source = Path(split['path'])
            current_files_verified[name] = source.is_file()
            if source.is_file():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),split['sha256'])
        REPORT['data_lists']={'validation':manifest['splits']['test']['path'],
            'final_test':manifest['splits']['test']['path'],'independent_validation_test':False,
            'test_sha256':manifest['splits']['test']['sha256'],'train_sha256':manifest['splits']['train']['sha256'],
            'current_list_files_available_and_verified':current_files_verified}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report',type=Path)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryRefactorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                  skipped=len(result.skipped),passed=result.wasSuccessful(),torch_version=torch.__version__,
                  device='cpu',note='Synthetic checks and real checkpoint load; accuracy metrics mocked except exact-mask list-protocol test. No formal training.')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
