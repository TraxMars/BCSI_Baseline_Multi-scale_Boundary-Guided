"""Check the prepared experiment's LR, real data loaders and isolation."""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_experiment import (ROOT, RUN_ROOT, SNAPSHOT, constant_lr_trainer,
                            now, sha256, write_json)


def load_trainer(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory / 'trainer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    metadata = json.loads((RUN_ROOT / 'metadata.json').read_text())
    sys.path.insert(0, str(SNAPSHOT))
    from dataloader.dataset import build_Dataset
    from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
    from utils.utils import patients_to_slices

    for relative, digest in metadata['baseline_source_sha256'].items():
        assert sha256(ROOT / relative) == digest, f'Baseline changed: {relative}'
    for relative, digest in metadata['snapshot_source_sha256'].items():
        assert sha256(SNAPSHOT / relative) == digest, f'Snapshot changed: {relative}'
        if relative != 'trainer.py':
            assert digest == metadata['baseline_source_sha256'][relative]
    assert (SNAPSHOT / 'trainer.py').read_text() == constant_lr_trainer(
        (ROOT / 'trainer.py').read_text())

    trainer_arguments = SimpleNamespace(in_channels=1, num_classes=2, device='cpu',
                                       base_lr=0.01, max_iterations=30000)
    fixed = load_trainer('fixed_lr_trainer', SNAPSHOT).Trainer(trainer_arguments)
    assert isinstance(fixed.scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert fixed.optimizer.param_groups[0]['lr'] == 0.01
    checked_lrs = {'0': fixed.optimizer.param_groups[0]['lr']}
    for iteration in range(1, 30001):
        fixed.optimizer.step()
        fixed.scheduler.step()
        assert fixed.optimizer.param_groups[0]['lr'] == 0.01, iteration
        if iteration in (1, 7999, 8000, 8001, 15999, 16000, 24000, 30000):
            checked_lrs[str(iteration)] = fixed.optimizer.param_groups[0]['lr']
    del fixed

    baseline = load_trainer('original_baseline_trainer', ROOT).Trainer(trainer_arguments)
    for _ in range(7999):
        baseline.optimizer.step()
        baseline.scheduler.step()
    original_decay_lr = baseline.optimizer.param_groups[0]['lr']
    assert 0 < original_decay_lr < 0.01
    baseline.optimizer.step()
    baseline.scheduler.step()
    assert baseline.optimizer.param_groups[0]['lr'] == 0.01
    del baseline

    checked_jobs = []
    for job in metadata['jobs']:
        command = job['command']
        assert command[command.index('--base_lr') + 1] == '0.01'
        assert command[command.index('--max_iterations') + 1] == '30000'
        assert command[command.index('--seed') + 1] == '42'
        assert Path(job['cwd']) == SNAPSHOT
        assert ROOT / 'Results' / 'fixed_lr_0p01_seed42' in Path(job['fold_root']).parents
        args = SimpleNamespace(dataset=job['dataset'], patch_size=job['patch_size'])
        random.seed(42)
        np.random.seed(42)
        dataset = build_Dataset(args, command[command.index('--data_path') + 1],
                                'train_' + job['dataset'], transform=True)
        assert len(dataset) == job['train_count']
        labeled = patients_to_slices(job['dataset'], job['labeled_percent'])
        assert labeled == job['labeled_count']
        sampler = TwoStreamBatchSampler(list(range(labeled)),
                                         list(range(labeled, len(dataset))), 4, 2)
        batch = next(iter(sampler))
        assert all(index < labeled for index in batch[:2])
        assert all(index >= labeled for index in batch[2:])
        sample = dataset[0]
        assert tuple(sample['image'].shape) == (1, *job['patch_size'])
        assert tuple(sample['label'].shape) == tuple(job['patch_size'])
        assert torch.isfinite(sample['image']).all()
        checked_jobs.append(dict(key=job['key'], training_cases=len(dataset),
                                 labeled_cases=labeled, sample_shape=list(sample['image'].shape),
                                 gpu=job['gpu']))
    assert len(checked_jobs) == 6
    for relative, digest in metadata['baseline_source_sha256'].items():
        assert sha256(ROOT / relative) == digest, f'Baseline changed during verification: {relative}'
    result = dict(checked_at=now(), passed=True, torch_version=torch.__version__,
                  fixed_lr_optimizer_updates_checked=30000, checked_lrs=checked_lrs,
                  original_schedule_lr_at_7999=original_decay_lr,
                  baseline_sources_unchanged=True,
                  only_snapshot_trainer_scheduler_changed=True, checked_jobs=checked_jobs)
    write_json(RUN_ROOT / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
