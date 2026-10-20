"""Audit old/new boundary semantics on the original eight labeled LA cases.

The fixed patch is train.list[0], seed 42, original rotation/flip/random crop
112x112x80. Sixteen deterministic patches are inspected without training.
"""
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
from torch.nn import functional as F

from dataloader.dataset import build_Dataset
from utils.bfa_supervision import LaplacianBoundary
from utils.boundary_supervision import boundary_target


class _TraceTransform:
    """Observe baseline NumPy draws, then restore RNG before baseline call."""
    def __init__(self, transform, kind, trace):
        self.transform, self.kind, self.trace = transform, kind, trace

    def __call__(self, sample):
        state = np.random.get_state()
        if self.kind == 'rotation':
            self.trace['rotation_k'] = int(np.random.randint(0, 4))
            self.trace['flip_axis'] = int(np.random.randint(0, 2))
        else:
            shape = sample['label'].shape
            patch = self.transform.output_size
            padding = [0, 0, 0]
            if any(a <= b for a, b in zip(shape, patch)):
                padding = [max((p - n) // 2 + 3, 0)
                           for n, p in zip(shape, patch)]
            padded = [n + 2 * p for n, p in zip(shape, padding)]
            self.trace['rotated_shape'] = list(shape)
            self.trace['symmetric_zero_padding'] = padding
            self.trace['crop_origin_in_padded_rotated_volume'] = [
                int(np.random.randint(0, n - p)) for n, p in zip(padded, patch)]
        np.random.set_state(state)
        return self.transform(sample)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shell(shape):
    shell = torch.ones(shape, dtype=torch.bool)
    if min(shape) > 2:
        shell[1:-1, 1:-1, 1:-1] = False
    return shell


def _statistics(target, mask):
    values = target[0, 0]
    shell = _shell(values.shape)
    count = int(values.sum().item())
    edge_count = int(values[shell].sum().item())
    return {
        'native_shape': list(values.shape),
        'positive_count': count,
        'positive_ratio': values.mean().item(),
        'foreground_ratio': mask.float().mean().item(),
        'edge_positive_count': edge_count,
        'edge_positive_ratio': values[shell].mean().item(),
        'boundary_fraction_at_crop_edge': edge_count / count if count else 0.,
        'vanished_foreground': bool(not mask.any()),
        'empty_boundary': count == 0,
        'high_coverage': values.mean().item() >= 0.8,
    }


def _orthogonal_plot(rows, output, title):
    figure, axes = plt.subplots(len(rows), 3, figsize=(11, 2.9 * len(rows)),
                               squeeze=False)
    for row_index, (name, mask, target) in enumerate(rows):
        for axis in range(3):
            slices = [slice(None)] * 3
            slices[axis] = mask.shape[axis] // 2
            plane = tuple(slices)
            ax = axes[row_index, axis]
            ax.imshow(mask[plane].T, cmap='gray', origin='lower', vmin=0, vmax=1)
            if target is not None:
                boundary = target[plane].T
                ax.imshow(np.ma.masked_where(boundary == 0, boundary),
                    cmap='autumn_r', vmin=0, vmax=1, alpha=0.85, origin='lower')
            label = name.replace('; ', '\n')
            ax.set_title(f'{label}\naxis {axis} @ {slices[axis]}', fontsize=9)
            ax.axis('off')
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=120)
    plt.close(figure)


def _visualize(example, output):
    mask = example['mask'][0, 0].numpy()
    zero = example['zero'][0, 0].numpy()
    replicate = example['replicate'][0, 0].numpy()
    native_rows, aligned_rows = [], [('GT foreground', mask, None)]
    for scale, (target, coarse_mask) in enumerate(example['old'], start=1):
        ratio = target.mean().item()
        shape = tuple(target.shape[2:])
        native_rows.append((f'Old E{scale} native {shape}; ratio={ratio:.5f}',
                            coarse_mask[0, 0].numpy(), target[0, 0].numpy()))
        displayed = F.interpolate(target, size=mask.shape, mode='nearest')[0, 0].numpy()
        aligned_rows.append((f'Old E{scale} nearest display; native ratio={ratio:.5f}',
                             mask, displayed))
    aligned_rows.extend([
        (f'New full GT / zero; ratio={zero.mean():.5f}', mask, zero),
        (f'New full GT / replicate; ratio={replicate.mean():.5f}', mask, replicate)])
    _orthogonal_plot([('GT foreground', mask, None)],
        output / 'label_audit_fixed_patch_gt.png', 'First labeled LA case, seed 42')
    _orthogonal_plot(native_rows, output / 'label_audit_old_native.png',
        'Old native targets: dimensions differ; no resizing in this figure')
    _orthogonal_plot(aligned_rows, output / 'label_audit_comparison.png',
        'Old targets resized by nearest for display only; new target shared by all four heads')

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    difference = zero != replicate
    for axis in range(3):
        for side, index in enumerate((0, -1)):
            slices = [slice(None)] * 3
            slices[axis] = index
            plane = tuple(slices)
            ax = axes[side, axis]
            ax.imshow(mask[plane].T, cmap='gray', vmin=0, vmax=1, origin='lower')
            edge = zero[plane].T
            ax.imshow(np.ma.masked_where(edge == 0, edge), cmap='Reds',
                      vmin=0, vmax=1, alpha=.55, origin='lower')
            diff = difference[plane].T
            ax.imshow(np.ma.masked_where(~diff, diff), cmap='Wistia',
                      vmin=0, vmax=1, alpha=1, origin='lower')
            ax.set_title(f'Crop face axis {axis}, index {index}; yellow = zero-only')
            ax.axis('off')
    figure.tight_layout()
    figure.savefig(output / 'label_audit_crop_faces.png', dpi=120)
    plt.close(figure)


def audit(data_path, output):
    output.mkdir(parents=True, exist_ok=False)
    patch_size = (112, 112, 80)
    args = SimpleNamespace(dataset='LA', patch_size=patch_size)
    dataset = build_Dataset(args, str(data_path), 'train_LA', transform=True)
    if len(dataset) < 8:
        raise ValueError('LA 10% audit requires first eight labeled training cases')
    trace = {}
    dataset.randrotflip = _TraceTransform(dataset.randrotflip, 'rotation', trace)
    dataset.randcrop = _TraceTransform(dataset.randcrop, 'crop', trace)
    extract_zero, extract_replicate = LaplacianBoundary(), LaplacianBoundary('replicate')
    rows, patches, example = [], [], None
    for case_index in range(8):
        for crop_index in range(2):
            seed = 42 + 2 * case_index + crop_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = dataset[case_index]
            labels = sample['label'][None]
            mask = (labels[:, None] > 0).float()
            zero, replicate = extract_zero(labels), extract_replicate(labels)
            old = []
            metadata = {'case_index': case_index, 'case': dataset.image_list[case_index],
                        'crop_index': crop_index, 'seed': seed}
            patch_metadata = dict(metadata, **trace,
                label_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                zero_replicate_difference_count=int((zero != replicate).sum()),
                zero_only_count=int(((zero > 0) & (replicate == 0)).sum()),
                replicate_only_count=int(((replicate > 0) & (zero == 0)).sum()))
            patches.append(patch_metadata)
            shell = _shell(patch_size)
            assert not (zero[0, 0][~shell] != replicate[0, 0][~shell]).any()
            assert not ((zero > 0) & (mask == 0)).any()
            assert not ((replicate > 0) & (mask == 0)).any()
            assert not ((replicate > 0) & (zero == 0)).any()
            for scale in range(5):
                size = tuple(n // 2**scale for n in patch_size)
                target, coarse_mask = boundary_target(labels, size)
                old.append((target, coarse_mask))
                rows.append(dict(metadata, method='old_morphology', scale=scale+1,
                                 **_statistics(target, coarse_mask)))
            for name, target in (('new_bfa_zero', zero), ('new_bfa_replicate', replicate)):
                rows.append(dict(metadata, method=name, scale='shared_E1_E4',
                                 **_statistics(target, mask)))
            if example is None:
                example = dict(mask=mask, zero=zero, replicate=replicate, old=old)
    _visualize(example, output)
    with (output / 'label_audit_ratios.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method, scale in dict.fromkeys((row['method'], row['scale']) for row in rows):
        group = [row for row in rows if (row['method'], row['scale']) == (method, scale)]
        summary.append({
            'method': method, 'scale': scale, 'patch_count': len(group),
            'native_shape': group[0]['native_shape'],
            'positive_ratio_mean': float(np.mean([r['positive_ratio'] for r in group])),
            'positive_ratio_min': min(r['positive_ratio'] for r in group),
            'positive_ratio_max': max(r['positive_ratio'] for r in group),
            'edge_positive_ratio_mean': float(np.mean([r['edge_positive_ratio'] for r in group])),
            'vanished_foreground_count': sum(r['vanished_foreground'] for r in group),
            'empty_boundary_count': sum(r['empty_boundary'] for r in group),
            'high_coverage_count': sum(r['high_coverage'] for r in group),
        })
    report = {
        'dataset': 'LA', 'labeled_percent': 10, 'labeled_cases': 8,
        'crops_per_case': 2, 'patch_size': list(patch_size),
        'train_list': str(data_path / 'train.list'),
        'train_list_sha256': _sha256(data_path / 'train.list'),
        'labeled_case_ids': dataset.image_list[:8],
        'first_case_file_sha256': _sha256(Path(dataset.sample_list[0])),
        'source_hashes': {name: _sha256(ROOT / name) for name in (
            'dataloader/dataset.py', 'utils/transforms.py',
            'utils/boundary_supervision.py', 'utils/bfa_supervision.py',
            'experiments/boundary_refactor/audit_labels.py')},
        'patches': patches, 'summary': summary,
        'fixed_patch_statistics': [row for row in rows if row['case_index'] == 0 and row['crop_index'] == 0],
        'zero_replicate_difference_count_total': sum(p['zero_replicate_difference_count'] for p in patches),
        'padding_assertions_passed': True,
        'interpretation': [
            'Old targets are dilation-minus-erosion after nearest GT downsampling; their original-space thickness grows with scale.',
            'New targets are foreground-inner boundaries extracted once at original GT resolution and shared across four heads.',
            'zero reference padding may label crop-contact foreground; replicate is a diagnostic alternative, not a separately tuned experiment.',
            'Nearest enlargement of old targets in PNG is display only; new loss uses trilinearly upsampled logits, never resized GT.',
            'This is a fixed labeled-train-patch target audit, not validation/test performance.',
        ],
    }
    (output / 'label_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Fixed LA boundary label audit', '',
        'Only the first eight labeled cases from original train.list, two deterministic original-loader crops each.',
        'Fixed example: first train case, seed 42, 112×112×80. Existing RandomRotFlip and RandomCrop are reused.',
        '', '| Target | Native size | Mean boundary ratio | Min | Max |',
        '|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['method']} / {row['scale']} | {row['native_shape']} | "
                     f"{row['positive_ratio_mean']:.6f} | {row['positive_ratio_min']:.6f} | {row['positive_ratio_max']:.6f} |")
    lines.extend(['', 'All padding assertions passed: differences confined to crop faces, replicate boundary is a subset of zero boundary, and both mark foreground only.',
        f"Zero/replicate differences across sixteen patches: {report['zero_replicate_difference_count_total']} voxels.", '',
        '![GT foreground](label_audit_fixed_patch_gt.png)', '',
        '![Aligned comparison](label_audit_comparison.png)', '',
        '![Old native targets](label_audit_old_native.png)', '',
        '![Crop-face reference padding](label_audit_crop_faces.png)', '',
        'The red overlays show target boundaries. Yellow crop-face voxels are present only with zero padding.',
        'Old targets in the aligned figure are enlarged by nearest only for display. A2/B2 upsample logits to one shared GT target.',
        'No training, validation selection, hyperparameter search or modification to the data list was performed.'])
    (output / 'label_audit_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'patch_count': len(patches), 'summary': summary,
                      'zero_replicate_difference_count_total': report['zero_replicate_difference_count_total'],
                      'output': str(output)}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.data_path, args.output)
