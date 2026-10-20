import copy
import csv
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vnet import VNet
from prediction import test_calculate_metric
from utils.losses import DiceLoss
from utils.boundary_supervision import (
    MultiScaleBoundarySupervision, encoder_stage_parameters,
    local_boundary_gradients, add_local_gradients,
)
from utils.bfa_supervision import BFAEncoderBoundarySupervision
from utils.decoder_supervision import DecoderDeepSupervision
from utils.gradient_diagnostics import boundary_gradient_diagnostics


class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Unchanged scheduler from the released training code."""

    def __init__(self, optimizer, base_lr, max_iters, power=0.9,
                 warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    """The released EMA helper, retained for the paper's MT baseline."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class Trainer(nn.Module):
    """Mean-Teacher baseline specified by the paper's no-SSP/BCI/CR ablation."""

    def __init__(self, args):
        super().__init__()
        self.best_performance = 0.0
        self.args = args
        self.boundary_mode = getattr(args, 'boundary_mode', 'none')
        self.boundary_weight = getattr(args, 'boundary_weight', 0.1)
        self.auxiliary_mode = getattr(args, 'auxiliary_mode', 'none')
        self.decoder_aux_weight = getattr(args, 'decoder_aux_weight', 0.1)
        self.final_supervision_weight = getattr(args, 'final_supervision_weight', 1.0)
        self.gradient_diagnostic_interval = getattr(args, 'gradient_diagnostic_interval', 0)
        self.gradient_cosine_threshold = getattr(args, 'gradient_cosine_threshold', -0.2)
        self.gradient_ratio_threshold = getattr(args, 'gradient_ratio_threshold', 0.1)
        if self.boundary_mode not in ('none', 'deep', 'local', 'bfa_deep', 'bfa_local'):
            raise ValueError('Unknown boundary_mode')
        if self.auxiliary_mode not in ('none', 'decoder_deep_seg'):
            raise ValueError('Unknown auxiliary_mode')
        if self.boundary_mode != 'none' and self.auxiliary_mode != 'none':
            raise ValueError('Encoder boundary and decoder supervision must be independent')
        self.auxiliary_active = self.boundary_mode != 'none' or self.auxiliary_mode != 'none'
        if self.auxiliary_active and getattr(args, 'guidance_mode', 'none') != 'none':
            raise ValueError('Auxiliary experiments require guidance_mode=none')
        if not np.isfinite(self.boundary_weight) or self.boundary_weight < 0:
            raise ValueError('boundary_weight must be finite and nonnegative')
        if not np.isfinite(self.decoder_aux_weight) or self.decoder_aux_weight < 0:
            raise ValueError('decoder_aux_weight must be finite and nonnegative')
        if not np.isfinite(self.final_supervision_weight) or self.final_supervision_weight <= 0:
            raise ValueError('final_supervision_weight must be finite and positive')
        if self.auxiliary_mode == 'none' and self.final_supervision_weight != 1.0:
            raise ValueError('final_supervision_weight is configurable only for Decoder_DS_C')
        if self.gradient_diagnostic_interval < 0:
            raise ValueError('gradient_diagnostic_interval must be nonnegative')
        if not np.isfinite(self.gradient_cosine_threshold) or not -1 <= self.gradient_cosine_threshold <= 0:
            raise ValueError('gradient_cosine_threshold must be in [-1, 0]')
        if not np.isfinite(self.gradient_ratio_threshold) or self.gradient_ratio_threshold < 0:
            raise ValueError('gradient_ratio_threshold must be finite and nonnegative')
        self.model = VNet(n_channels=args.in_channels, n_classes=args.num_classes).to(args.device)
        self.ema_model = copy.deepcopy(self.model).to(args.device)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.boundary = None
        self.decoder_auxiliary = None
        parameters = list(self.model.parameters())
        if self.boundary_mode != 'none':
            # Heads are constructed on CPU without consuming the baseline RNG
            # stream (data sampling and VNet dropout keep their original seeds).
            with torch.random.fork_rng(devices=[]):
                if self.boundary_mode in ('bfa_deep', 'bfa_local'):
                    self.boundary = BFAEncoderBoundarySupervision(
                        padding_mode=getattr(args, 'boundary_padding_mode', 'zero')).to(args.device)
                else:
                    self.boundary = MultiScaleBoundarySupervision().to(args.device)
            self.stage_parameters = encoder_stage_parameters(self.model.encoder)
            parameters += list(self.boundary.parameters())
        if self.auxiliary_mode == 'decoder_deep_seg':
            with torch.random.fork_rng(devices=[]):
                self.decoder_auxiliary = DecoderDeepSupervision(num_classes=args.num_classes).to(args.device)
            parameters += list(self.decoder_auxiliary.parameters())
        self.optimizer = torch.optim.SGD(
            parameters, lr=args.base_lr, momentum=0.9, weight_decay=0.00001)
        self.scheduler = PolyWarmRestartScheduler(
            self.optimizer, base_lr=args.base_lr, max_iters=args.max_iterations,
            power=0.9, warm_restart_iters=8000)
        self.dice_loss = DiceLoss(args.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self._reported_boundary_issues = set()

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def segmentation_loss(self, logits, labels):
        return 0.5 * (self.ce_loss(logits, labels) + self.dice_loss(logits, labels, softmax=True))

    def train(self, sampled_batch, iter_num, snapshot_path):
        volume_batch = sampled_batch['image'].to(self.args.device)
        label_batch = sampled_batch['label'].to(self.args.device)
        labeled_bs = self.args.labeled_bs
        if self.decoder_auxiliary is not None:
            features = self.model.encoder(volume_batch)
            student_logits, decoder_features = self.model.decoder(features, return_features=True)
        elif self.boundary is None:
            student_logits = self.model(volume_batch)
        else:
            features = self.model.encoder(volume_batch)
            student_logits = self.model.decoder(features)
        with torch.no_grad():
            teacher_logits = self.ema_model(volume_batch)
        supervised_loss = self.segmentation_loss(student_logits[:labeled_bs], label_batch[:labeled_bs])
        consistency_loss = F.mse_loss(
            torch.softmax(student_logits[labeled_bs:], dim=1),
            torch.softmax(teacher_logits[labeled_bs:], dim=1))
        consistency_weight = self.get_current_consistency_weight(iter_num // 150)
        if self.decoder_auxiliary is None:
            loss = supervised_loss + consistency_weight * consistency_loss
        else:
            loss = self.final_supervision_weight * supervised_loss + consistency_weight * consistency_loss
        self.optimizer.zero_grad()
        boundary_result = None
        decoder_result = None
        if self.boundary is not None:
            boundary_result = self.boundary(features, label_batch[:labeled_bs])
            boundary_loss = boundary_result.total_loss
            main_loss = loss
            loss = main_loss + self.boundary_weight * boundary_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite boundary/MT loss')
            gradients = None
            if self.boundary_mode in ('local', 'bfa_local'):
                gradients = local_boundary_gradients(
                    boundary_result, self.stage_parameters, self.boundary.heads,
                    self.boundary_weight)
            if self.gradient_diagnostic_interval and iter_num % self.gradient_diagnostic_interval == 0:
                diagnostics = boundary_gradient_diagnostics(
                    main_loss, boundary_result, self.stage_parameters, self.boundary.heads,
                    self.boundary_weight, self.boundary_mode,
                    self.gradient_cosine_threshold, self.gradient_ratio_threshold,
                    local_gradients=gradients)
                self._write_gradient_diagnostics(diagnostics, snapshot_path, iter_num)
            if self.boundary_mode in ('deep', 'bfa_deep'):
                loss.backward()
            else:
                main_loss.backward()
                add_local_gradients(gradients)
        elif self.decoder_auxiliary is not None:
            decoder_result = self.decoder_auxiliary(
                decoder_features, label_batch[:labeled_bs], student_logits)
            loss = loss + self.decoder_aux_weight * decoder_result.total_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite decoder/MT loss')
            loss.backward()
        else:
            loss.backward()
        if self.auxiliary_active:
            finite = [torch.isfinite(p.grad).all() for group in self.optimizer.param_groups
                      for p in group['params'] if p.grad is not None]
            if not torch.stack(finite).all():
                raise FloatingPointError('Nonfinite Student/auxiliary gradient')
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()
        update_ema_variables(self.model, self.ema_model, self.args.ema_decay, iter_num)
        self.scheduler.step()
        logging.info('iteration %d : loss : %f supervised : %f consistency : %f lr : %f',
                     iter_num, loss, supervised_loss, consistency_loss,
                     self.optimizer.param_groups[0]['lr'])
        if boundary_result is not None:
            row = dict(iteration=iter_num, supervised_loss=supervised_loss.item(),
                       consistency_loss=consistency_loss.item(),
                       consistency_weight=consistency_weight,
                       total_boundary_loss=boundary_loss.item(), total_loss=loss.item(),
                       learning_rate=learning_rate,
                       next_learning_rate=self.optimizer.param_groups[0]['lr'])
            for index, (value, diagnostic) in enumerate(zip(
                    boundary_result.losses, boundary_result.diagnostics), 1):
                row[f'boundary_loss_{index}'] = value.item()
                row[f'boundary_positive_ratio_{index}'] = diagnostic['positive_ratio']
                row[f'boundary_shape_{index}'] = json.dumps(diagnostic['shape'])
                if 'upsampled_shape' in diagnostic:
                    row[f'boundary_upsampled_shape_{index}'] = json.dumps(diagnostic['upsampled_shape'])
                    row[f'boundary_target_shape_{index}'] = json.dumps(diagnostic['target_shape'])
                row[f'boundary_ratios_per_sample_{index}'] = json.dumps(
                    diagnostic['per_sample_positive_ratio'])
                for key in ('foreground_ratio', 'vanished_foreground_samples',
                            'empty_boundary_samples', 'high_coverage_samples'):
                    row[f'boundary_{key}_{index}'] = diagnostic[key]
                for key in ('vanished_foreground_samples', 'empty_boundary_samples',
                            'high_coverage_samples'):
                    if diagnostic[key] and (index, key) not in self._reported_boundary_issues:
                        logging.warning('boundary scale %d: %s=%s, shape=%s, ratio=%s',
                                        index, key, diagnostic[key], diagnostic['shape'],
                                        diagnostic['per_sample_positive_ratio'])
                        self._reported_boundary_issues.add((index, key))
            path = os.path.join(snapshot_path, 'training_metrics.csv')
            exists = os.path.exists(path)
            with open(path, 'a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            logging.info('boundary mode=%s total=%f positive_ratios=%s',
                         self.boundary_mode, boundary_loss.item(),
                         [d['positive_ratio'] for d in boundary_result.diagnostics])
            return row
        if decoder_result is not None:
            row = dict(iteration=iter_num, supervised_loss=supervised_loss.item(),
                final_supervision_weight=self.final_supervision_weight,
                consistency_loss=consistency_loss.item(), consistency_weight=consistency_weight,
                decoder_aux_weight=self.decoder_aux_weight,
                total_decoder_aux_loss=decoder_result.total_loss.item(), total_loss=loss.item(),
                learning_rate=learning_rate, next_learning_rate=self.optimizer.param_groups[0]['lr'])
            for index, (logits, value) in enumerate(zip(
                    decoder_result.side_logits, decoder_result.side_losses), 1):
                row[f'decoder_aux_loss_{index}'] = value.item()
                row[f'decoder_output_shape_{index}'] = json.dumps(list(logits.shape))
            row['decoder_output_shape_4'] = json.dumps(list(student_logits[:labeled_bs].shape))
            self._append_metrics(os.path.join(snapshot_path, 'training_metrics.csv'), row)
            return row

    @staticmethod
    def _append_metrics(path, row):
        exists = os.path.exists(path)
        with open(path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_gradient_diagnostics(self, rows, snapshot_path, iter_num):
        for row in rows:
            self._append_metrics(os.path.join(snapshot_path, 'gradient_diagnostics.csv'),
                                 dict(iteration=iter_num, mode=self.boundary_mode, **row))

    def save_boundary_state(self, snapshot_path, iter_num, best=False):
        """Keep inference checkpoints plain VNet; save auxiliary state separately."""
        if not self.auxiliary_active:
            return
        if best:
            key = 'boundary' if self.boundary is not None else 'decoder_auxiliary'
            module = self.boundary if self.boundary is not None else self.decoder_auxiliary
            prefix = 'BoundaryHeads' if self.boundary is not None else 'DecoderHeads'
            torch.save({'iteration': iter_num, key: module.state_dict(),
                        'config': vars(self.args)}, os.path.join(snapshot_path, f'{prefix}_iter_{iter_num}.pth'))
        filename = ('boundary_training_state_latest.pth' if self.boundary is not None
                    else 'decoder_training_state_latest.pth')
        path = os.path.join(snapshot_path, filename)
        state = dict(iteration=iter_num, student=self.model.state_dict(),
                        teacher=self.ema_model.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(),
                        best_performance=self.best_performance,
                        config=vars(self.args), torch_rng=torch.get_rng_state())
        if self.boundary is not None:
            state['boundary'] = self.boundary.state_dict()
        if self.decoder_auxiliary is not None:
            state['decoder_auxiliary'] = self.decoder_auxiliary.state_dict()
        torch.save(state, path + '.tmp')
        os.replace(path + '.tmp', path)

    def test(self, snapshot_path, iter_num):
        self.ema_model.eval()
        if not self.auxiliary_active:
            dice, hd95 = test_calculate_metric(self.args, self.ema_model, val=True)
        else:
            metrics = test_calculate_metric(self.args, self.ema_model, val=True,
                                            return_details=True)
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError('Nonfinite boundary-experiment validation metric')
            dice, hd95 = metrics['dice'], metrics['hd95']
            path = os.path.join(snapshot_path, 'validation_metrics.csv')
            exists = os.path.exists(path)
            with open(path, 'a', newline='') as handle:
                row = dict(iteration=iter_num, **metrics)
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            logging.info('iteration %d : asd : %f boundary_f1_1voxel : %f',
                         iter_num, metrics['asd'], metrics['boundary_f1_1voxel'])
        if dice > self.best_performance:
            self.best_performance = dice
            torch.save(self.ema_model.state_dict(),
                       os.path.join(snapshot_path, 'Model_iter_' + str(iter_num) + '.pth'))
            if self.auxiliary_active:
                with open(os.path.join(snapshot_path, 'best_metrics.json'), 'w') as handle:
                    json.dump(dict(iteration=iter_num, **metrics), handle, indent=2)
                self.save_boundary_state(snapshot_path, iter_num, best=True)
        elif self.auxiliary_active:
            self.save_boundary_state(snapshot_path, iter_num)
        logging.info('iteration %d : mean_dice : %f mean_hd95 : %f', iter_num, dice, hd95)
        self.model.train()
        self.ema_model.train()
