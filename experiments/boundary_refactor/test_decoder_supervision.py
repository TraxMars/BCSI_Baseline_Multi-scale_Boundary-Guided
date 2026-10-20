"""Independent tests for the DuCiSC-style decoder segmentation control.

Run only after the A2/B2 synthetic audits and real-data smoke checks. No
formal experiment is started by this script. Tests use synthetic tensors.
"""
import argparse
import copy
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

import torch
from torch.nn import functional as F

from model.vnet import VNet
from utils.decoder_supervision import DecoderDeepSupervision
from utils.losses import DiceLoss


REPORT = {'decoder_shapes': {}, 'data': 'synthetic only'}


def original_vnet_class():
    source = Path(__file__).parent / 'reference/vnet_before_refactor.py'
    spec = importlib.util.spec_from_file_location('vnet_before_decoder_ds', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VNet


def segmentation_loss(logits, labels):
    return 0.5 * (F.cross_entropy(logits, labels) +
                  DiceLoss(logits.shape[1])(logits, labels, softmax=True))


class DecoderSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_default_and_optional_forward_match_reference_with_same_rng(self):
        # Real residual architecture and dropout sequence; smaller channel
        # count only reduces CPU audit cost, without changing its topology.
        kwargs = dict(n_filters=2, normalization='batchnorm', has_dropout=True)
        original = original_vnet_class()(**kwargs)
        image = torch.randn(2, 1, 32, 32, 32)
        for training in (True, False):
            for return_features in (False, True):
                before = copy.deepcopy(original).train(training)
                after = VNet(**kwargs).train(training)
                after.load_state_dict(before.state_dict(), strict=True)
                torch.manual_seed(701)
                old_logits = before(image)
                old_logits.square().mean().backward()
                old_rng = torch.get_rng_state()
                torch.manual_seed(701)
                if return_features:
                    features = after.encoder(image)
                    logits, taps = after.decoder(features, return_features=True)
                    self.assertEqual(len(taps), 3)
                else:
                    logits = after(image)
                    self.assertIsInstance(logits, torch.Tensor)
                logits.square().mean().backward()
                self.assertTrue(torch.equal(old_logits, logits))
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                self.assert_state_equal(before.state_dict(), after.state_dict())
                for (name, old), (other_name, new) in zip(before.named_parameters(),
                                                       after.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertTrue(torch.equal(old.grad, new.grad), name)
        REPORT['reference_exact_forward_backward_rng'] = True

    def test_side_features_match_exact_ducisc_pre_block_positions(self):
        model = VNet(n_filters=2, normalization='none', has_dropout=False).eval()
        captured = []
        def capture(_module, inputs):
            captured.append(inputs[0].detach().clone())
        handles = [block.register_forward_pre_hook(capture) for block in
                   (model.decoder.block_six, model.decoder.block_seven,
                    model.decoder.block_eight)]
        try:
            with torch.no_grad():
                features = model.encoder(torch.randn(2, 1, 32, 32, 32))
                final, taps = model.decoder(features, return_features=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(final.shape, (2, 2, 32, 32, 32))
        for tap, expected, channels, spatial in zip(taps, captured, (16, 8, 4), (4, 8, 16)):
            self.assertTrue(torch.equal(tap, expected))
            self.assertEqual(tap.shape, (2, channels, spatial, spatial, spatial))
        REPORT['tap_positions'] = ['before block_six, after skip4',
                                   'before block_seven, after skip3',
                                   'before block_eight, after skip2']

    def test_shapes_and_nearest_labels_match_all_four_outputs(self):
        module = DecoderDeepSupervision(channels=(8, 4, 2), num_classes=3)
        features = [torch.randn(3, c, *shape, requires_grad=True) for c, shape in
                    zip((8, 4, 2), ((2, 3, 4), (4, 6, 8), (8, 12, 16)))]
        labels = (torch.arange(16 * 24 * 32).reshape(1, 16, 24, 32) % 3).long()
        final = torch.randn(3, 3, 16, 24, 32, requires_grad=True)
        result = module(features, labels, final)
        self.assertEqual(len(result.outputs), 4)
        self.assertEqual(len(result.targets), 4)
        self.assertEqual(len(result.side_losses), 3)
        self.assertIs(result.targets[-1], labels)
        self.assertTrue(torch.equal(result.outputs[-1], final[:1]))
        for prediction, target, shape in zip(result.side_logits, result.side_targets,
                                            ((2, 3, 4), (4, 6, 8), (8, 12, 16))):
            self.assertEqual(prediction.shape, (1, 3, *shape))
            self.assertEqual(target.shape, (1, *shape))
            # Independent nearest-neighbor integer indices, no interpolation.
            indices = [torch.div(torch.arange(dst) * src, dst, rounding_mode='floor')
                       for src, dst in zip(labels.shape[-3:], shape)]
            expected = labels[:, indices[0]][:, :, indices[1]][:, :, :, indices[2]]
            self.assertTrue(torch.equal(target, expected))
            self.assertEqual(target.dtype, torch.long)
            self.assertTrue(set(target.unique().tolist()) <= {0, 1, 2})
        for logits, target, loss in zip(result.side_logits, result.side_targets, result.side_losses):
            torch.testing.assert_close(loss, segmentation_loss(logits, target), rtol=0, atol=0)
        torch.testing.assert_close(result.total_loss, torch.stack(result.side_losses).mean(),
                                   rtol=0, atol=0)
        without_final = module(features, labels)
        self.assertEqual(len(without_final.outputs), 3)
        self.assertEqual(len(without_final.targets), 3)

    def test_final_segmentation_loss_is_counted_once(self):
        module = DecoderDeepSupervision(channels=(8, 4, 2))
        features = [torch.randn(2, channel, size, size, size, requires_grad=True)
                    for channel, size in zip((8, 4, 2), (2, 4, 8))]
        labels = torch.randint(0, 2, (1, 16, 16, 16))
        final = torch.randn(2, 2, 16, 16, 16, requires_grad=True)
        result = module(features, labels, final)
        self.assertIsNone(torch.autograd.grad(result.total_loss, final, allow_unused=True,
                                              retain_graph=True)[0])
        primary_loss = segmentation_loss(final[:1], labels)
        primary_gradient = torch.autograd.grad(primary_loss, final, retain_graph=True)[0]
        total_gradient = torch.autograd.grad(primary_loss + 0.1 * result.total_loss,
                                             final, retain_graph=True)[0]
        self.assertTrue(torch.equal(primary_gradient, total_gradient))
        altered = module(features, labels, final + 500.)
        self.assertTrue(torch.equal(result.total_loss, altered.total_loss))
        REPORT['final_main_loss_weight'] = 1.0
        REPORT['side_mean_weight_for_update_test'] = 0.1
        REPORT['final_not_in_auxiliary_loss'] = True

    def test_labeled_only_head_gradients_and_finite_extreme_logits(self):
        module = DecoderDeepSupervision(channels=(8, 4, 2))
        features = [torch.randn(3, channel, size, size, size, requires_grad=True)
                    for channel, size in zip((8, 4, 2), (2, 4, 8))]
        labels = torch.randint(0, 2, (2, 16, 16, 16))
        batch_sizes = []
        handles = [head.register_forward_pre_hook(
            lambda _head, inputs: batch_sizes.append(inputs[0].shape[0])) for head in module.heads]
        try:
            result = module(features, labels)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(batch_sizes, [2, 2, 2])
        result.total_loss.backward()
        for feature, head in zip(features, module.heads):
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad[:2].abs().sum().item(), 0)
            self.assertEqual(feature.grad[2:].abs().sum().item(), 0)
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for value in (-1000., 1000.):
            logits = torch.full((2, 2, 2, 4, 6), value)
            logits[:, 1] *= -1
            logits.requires_grad_()
            for label in (0, 1):
                target = torch.full((2, 2, 4, 6), label, dtype=torch.long)
                loss = segmentation_loss(logits, target)
                gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(gradient).all())

    def test_la_native_side_shapes(self):
        module = DecoderDeepSupervision().eval()
        shapes = ((14, 14, 10), (28, 28, 20), (56, 56, 40))
        features = [torch.randn(2, channel, *shape) for channel, shape in
                    zip((128, 64, 32), shapes)]
        labels = torch.zeros(1, 112, 112, 80, dtype=torch.long)
        labels[:, 28:84, 28:84, 20:60] = 1
        final = torch.randn(2, 2, 112, 112, 80)
        with torch.no_grad():
            result = module(features, labels, final)
        expected_shapes = [(1, 2, *shape) for shape in shapes] + [(1, 2, 112, 112, 80)]
        self.assertEqual([tuple(x.shape) for x in result.outputs], expected_shapes)
        REPORT['decoder_shapes']['LA_112x112x80'] = {
            'outputs': [list(x.shape) for x in result.outputs],
            'targets': [list(x.shape) for x in result.targets],
            'target_foreground_ratios': [x.float().mean().item() for x in result.targets]}

    def test_old_state_dict_strict_inference_and_separate_heads(self):
        # Full default 16-channel architecture, no new VNet parameters/buffers.
        original = original_vnet_class()().eval()
        current = VNet().eval()
        current.load_state_dict(original.state_dict(), strict=True)
        self.assert_state_equal(original.state_dict(), current.state_dict())
        auxiliary = DecoderDeepSupervision()
        self.assertEqual(len(auxiliary.heads), 3)
        self.assertEqual(sum(1 for _ in auxiliary.parameters()), 6)
        self.assertFalse(any('head' in name for name in current.state_dict()))
        image = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(image), current(image)))
        REPORT['legacy_state_dict_strict_load'] = True

    def test_trainer_single_student_ema_update_and_original_initialization(self):
        from trainer import Trainer, update_ema_variables
        config = dict(in_channels=1, num_classes=2, device='cpu', base_lr=0.01,
                      max_iterations=30000, labeled_bs=2, ema_decay=0.9,
                      consistency=0.1, consistency_rampup=200,
                      boundary_mode='none', boundary_weight=0.1, guidance_mode='none',
                      auxiliary_mode='decoder_deep_seg', decoder_aux_weight=0.1,
                      final_supervision_weight=1.0, gradient_diagnostic_interval=0)
        torch.manual_seed(42)
        trainer = Trainer(SimpleNamespace(**config))
        auxiliary_rng = torch.get_rng_state()
        torch.manual_seed(42)
        baseline = Trainer(SimpleNamespace(**dict(config, auxiliary_mode='none')))
        baseline_rng = torch.get_rng_state()
        self.assert_state_equal(trainer.model.state_dict(), baseline.model.state_dict())
        self.assert_state_equal(trainer.ema_model.state_dict(), baseline.ema_model.state_dict())
        self.assertTrue(torch.equal(auxiliary_rng, baseline_rng))
        self.assertIsNone(trainer.boundary)
        self.assertIsInstance(trainer.decoder_auxiliary, DecoderDeepSupervision)
        self.assertFalse(any('head' in name for name in trainer.ema_model.state_dict()))
        student_before = [p.detach().clone() for p in trainer.model.parameters()]
        teacher_before = [p.detach().clone() for p in trainer.ema_model.parameters()]
        heads_before = [p.detach().clone() for p in trainer.decoder_auxiliary.parameters()]
        batch = {'image': torch.randn(4, 1, 32, 32, 32),
                 'label': torch.randint(0, 2, (4, 32, 32, 32))}
        optimizer_parameters = [p for group in trainer.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(optimizer_parameters), len({id(p) for p in optimizer_parameters}))
        self.assertEqual({id(p) for p in optimizer_parameters},
                         {id(p) for p in trainer.model.parameters()} |
                         {id(p) for p in trainer.decoder_auxiliary.parameters()})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step, \
                    patch.object(trainer.scheduler, 'step', wraps=trainer.scheduler.step) as schedule, \
                    patch('trainer.update_ema_variables', wraps=update_ema_variables) as ema, \
                    patch.object(trainer, 'segmentation_loss', wraps=trainer.segmentation_loss) as final_loss:
                row = trainer.train(batch, 17, directory)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(schedule.call_count, 1)
            self.assertEqual(ema.call_count, 1)
            self.assertEqual(final_loss.call_count, 1)
        self.assertTrue(any(not torch.equal(before, after) for before, after in
                            zip(student_before, trainer.model.parameters())))
        self.assertTrue(all(not torch.equal(before, after) for before, after in
                            zip(heads_before, trainer.decoder_auxiliary.parameters())))
        for old_teacher, new_student, new_teacher in zip(teacher_before,
                trainer.model.parameters(), trainer.ema_model.parameters()):
            torch.testing.assert_close(new_teacher, old_teacher * 0.9 + new_student * 0.1)
        self.assertTrue(all(p.grad is None for p in trainer.ema_model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in optimizer_parameters
                            if p.grad is not None))
        self.assertTrue(torch.isfinite(torch.tensor(row['total_loss'])))
        REPORT['trainer_C_single_step'] = {'optimizer_step': step.call_count,
                                          'ema_update': ema.call_count,
                                          'scheduler_step': schedule.call_count,
                                          'main_final_supervision_calls': final_loss.call_count,
                                          'initialization_rng_matches_none': True,
                                          'total_loss': row['total_loss']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DecoderSupervisionTests))
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), passed=result.wasSuccessful(),
                  torch_version=torch.__version__,
                  note='Synthetic tests only; no performance or formal-training claim.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(REPORT, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
