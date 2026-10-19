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
        if self.boundary_mode not in ('none', 'deep', 'local'):
            raise ValueError('Unknown boundary_mode')
        if self.boundary_mode != 'none' and getattr(args, 'guidance_mode', 'none') != 'none':
            raise ValueError('Boundary experiments require guidance_mode=none')
        if not np.isfinite(self.boundary_weight) or self.boundary_weight < 0:
            raise ValueError('boundary_weight must be finite and nonnegative')
        self.model = VNet(n_channels=args.in_channels, n_classes=args.num_classes).to(args.device)
        self.ema_model = copy.deepcopy(self.model).to(args.device)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.boundary = None
        parameters = list(self.model.parameters())
        if self.boundary_mode != 'none':
            # Heads are constructed on CPU without consuming the baseline RNG
            # stream (data sampling and VNet dropout keep their original seeds).
            with torch.random.fork_rng(devices=[]):
                self.boundary = MultiScaleBoundarySupervision().to(args.device)
            self.stage_parameters = encoder_stage_parameters(self.model.encoder)
            parameters += list(self.boundary.parameters())
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
        if self.boundary is None:
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
        loss = supervised_loss + consistency_weight * consistency_loss
        self.optimizer.zero_grad()
        boundary_result = None
        if self.boundary is not None:
            boundary_result = self.boundary(features, label_batch[:labeled_bs])
            boundary_loss = boundary_result.total_loss
            main_loss = loss
            loss = main_loss + self.boundary_weight * boundary_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite boundary/MT loss')
            if self.boundary_mode == 'deep':
                loss.backward()
            else:
                gradients = local_boundary_gradients(
                    boundary_result, self.stage_parameters, self.boundary.heads,
                    self.boundary_weight)
                main_loss.backward()
                add_local_gradients(gradients)
            finite = [torch.isfinite(p.grad).all() for group in self.optimizer.param_groups
                      for p in group['params'] if p.grad is not None]
            if not torch.stack(finite).all():
                raise FloatingPointError('Nonfinite Student/boundary gradient')
        else:
            loss.backward()
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

    def save_boundary_state(self, snapshot_path, iter_num, best=False):
        """Keep inference checkpoints plain VNet; save auxiliary state separately."""
        if self.boundary is None:
            return
        if best:
            torch.save({'iteration': iter_num, 'boundary': self.boundary.state_dict(),
                        'config': vars(self.args)}, os.path.join(
                            snapshot_path, f'BoundaryHeads_iter_{iter_num}.pth'))
        path = os.path.join(snapshot_path, 'boundary_training_state_latest.pth')
        torch.save(dict(iteration=iter_num, student=self.model.state_dict(),
                        teacher=self.ema_model.state_dict(),
                        boundary=self.boundary.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(),
                        best_performance=self.best_performance,
                        config=vars(self.args), torch_rng=torch.get_rng_state()), path + '.tmp')
        os.replace(path + '.tmp', path)

    def test(self, snapshot_path, iter_num):
        self.ema_model.eval()
        if self.boundary is None:
            dice, hd95 = test_calculate_metric(self.args, self.ema_model, val=True)
        else:
            metrics = test_calculate_metric(self.args, self.ema_model, val=True,
                                            return_details=True)
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
            if self.boundary is not None:
                with open(os.path.join(snapshot_path, 'best_metrics.json'), 'w') as handle:
                    json.dump(dict(iteration=iter_num, **metrics), handle, indent=2)
                self.save_boundary_state(snapshot_path, iter_num, best=True)
        if self.boundary is not None:
            self.save_boundary_state(snapshot_path, iter_num)
        logging.info('iteration %d : mean_dice : %f mean_hd95 : %f', iter_num, dice, hd95)
        self.model.train()
        self.ema_model.train()
