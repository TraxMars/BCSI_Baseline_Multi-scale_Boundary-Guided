"""Inspect deterministic real LA labeled crops at LA and 96-cube resolutions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader.dataset import build_Dataset
from utils.boundary_supervision import boundary_target
from utils.utils import patients_to_slices


def audit(data_path, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, examples = [], {}
    for shape in ((112, 112, 80), (96, 96, 96)):
        name = 'x'.join(map(str, shape))
        args = SimpleNamespace(dataset='LA', patch_size=shape)
        dataset = build_Dataset(args, str(data_path), 'train_LA', transform=True)
        for case_index in range(patients_to_slices('LA', 10)):
            for crop_index in range(2):
                seed = 42 + case_index * 2 + crop_index
                random.seed(seed)
                np.random.seed(seed)
                sample = dataset[case_index]
                labels = sample['label'][None]
                for scale in range(5):
                    size = tuple(n // 2**scale for n in shape)
                    target, mask = boundary_target(labels, size)
                    ratio = target.mean().item()
                    vanished = bool(labels.any() and not mask.any())
                    rows.append(dict(patch=name, case=dataset.image_list[case_index],
                        crop=crop_index, seed=seed, scale=scale+1, shape=str(size),
                        positive_ratio=ratio, foreground_ratio=mask.mean().item(),
                        vanished_foreground=vanished, empty_boundary=ratio == 0,
                        high_coverage=ratio >= 0.8))
                    if case_index == 0 and crop_index == 0:
                        examples[(name, scale)] = (mask[0, 0].numpy(), target[0, 0].numpy())
    with (output / 'boundary_ratios.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in sorted({row['patch'] for row in rows}):
        summary[name] = []
        for scale in range(1, 6):
            group = [row for row in rows if row['patch'] == name and row['scale'] == scale]
            values = [row['positive_ratio'] for row in group]
            summary[name].append(dict(scale=scale, shape=group[0]['shape'],
                count=len(group), positive_ratio_mean=float(np.mean(values)),
                positive_ratio_min=min(values), positive_ratio_max=max(values),
                vanished_foreground_count=sum(row['vanished_foreground'] for row in group),
                empty_boundary_count=sum(row['empty_boundary'] for row in group),
                high_coverage_count=sum(row['high_coverage'] for row in group)))
    # Each target is inspected through three central orthogonal slices.
    for name in summary:
        figure, axes = plt.subplots(5, 3, figsize=(9, 13))
        for scale in range(5):
            mask, target = examples[(name, scale)]
            for axis in range(3):
                sl = [slice(None)] * 3
                sl[axis] = mask.shape[axis] // 2
                a = axes[scale, axis]
                a.imshow(mask[tuple(sl)].T, cmap='gray', origin='lower', vmin=0, vmax=1)
                a.imshow(np.ma.masked_where(target[tuple(sl)].T == 0, target[tuple(sl)].T),
                         cmap='autumn', origin='lower', vmin=0, vmax=1, alpha=0.7)
                a.set_title(f'E{scale+1} {mask.shape}, ratio={target.mean():.3f}')
                a.axis('off')
        figure.tight_layout()
        figure.savefig(output / f'boundary_targets_{name}.png', dpi=120)
        plt.close(figure)
    report = dict(dataset='LA', labeled_percent=10, labeled_cases=8,
                  crops_per_case=2, summary=summary,
                  train_list_sha256=hashlib.sha256((data_path / 'train.list').read_bytes()).hexdigest(),
                  note='Fixed seeded original crop/rotation/flip; 96-cube is a target diagnostic, not a new LA experiment.')
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.data_path, args.output)
