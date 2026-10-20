"""Collect executed evidence; do not infer completion or invent accuracy results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from model.vnet import VNet


def main():
    report = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  automated_tests=json.loads((HERE / 'verification.json').read_text()),
                  target_audit=json.loads((HERE / 'target_audit/audit.json').read_text()),
                  smoke_runs={}, formal_runs={})
    assert report['automated_tests']['passed']
    configs, first_steps, formal_configs, formal_histories = [], [], [], []
    for name in ('Boundary_A', 'Boundary_B'):
        fold = ROOT / 'Results/boundary_supervision_smoke' / name / 'seed_42/result_LA_10l/fold_0'
        status = json.loads((fold / 'run_status.json').read_text())
        assert status['state'] == 'completed' and status['iteration'] == 2
        config = json.loads((fold / 'experiment_config.json').read_text())
        rows = list(csv.DictReader((fold / 'training_metrics.csv').open()))
        assert len(rows) == 2
        for row in rows:
            for key, value in row.items():
                if 'shape' not in key and 'ratios_per_sample' not in key:
                    assert math.isfinite(float(value)), (key, value)
            assert math.isclose(float(row['total_loss']),
                float(row['supervised_loss']) + float(row['consistency_weight']) * float(row['consistency_loss'])
                + 0.1 * float(row['total_boundary_loss']), abs_tol=1e-6)
        validation = list(csv.DictReader((fold / 'validation_metrics.csv').open()))
        assert len(validation) == 1
        assert all(math.isfinite(float(v)) for v in validation[0].values())
        checkpoint = fold / 'Model_iter_2.pth'
        VNet().load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        heads = torch.load(fold / 'BoundaryHeads_iter_2.pth', map_location='cpu', weights_only=False)
        assert len(heads['boundary']) == 10
        report['smoke_runs'][name] = dict(path=str(fold), status=status,
            training_steps=2, validation_cases=20, validation=validation[0],
            checkpoint_strict_load=True,
            note='Two-step smoke metrics only; not evidence of experimental effectiveness.')
        configs.append(config)
        first_steps.append(rows[0])
        formal = ROOT / 'Results' / name / 'seed_42/result_LA_10l/fold_0'
        state = formal / 'run_status.json'
        report['formal_runs'][name] = dict(path=str(formal), status=(
            json.loads(state.read_text()) if state.is_file() else {'state': 'not_started'}))
        for metric_file in ('best_metrics.json', 'test_metrics.json'):
            path = formal / metric_file
            if path.is_file():
                report['formal_runs'][name][path.stem] = json.loads(path.read_text())
        if (formal / 'experiment_config.json').is_file():
            formal_configs.append(json.loads((formal / 'experiment_config.json').read_text()))
        if (formal / 'training_metrics.csv').is_file():
            history = list(csv.DictReader((formal / 'training_metrics.csv').open()))
            history = [row for row in history if all(v is not None for v in row.values())]
            formal_histories.append(history)
            report['formal_runs'][name]['coverage_observation'] = dict(
                iterations_observed=len(history), labeled_samples=2*len(history),
                scale5_high_coverage_samples=sum(int(r['boundary_high_coverage_samples_5']) for r in history),
                scale5_empty_boundary_samples=sum(int(r['boundary_empty_boundary_samples_5']) for r in history),
                scale5_vanished_foreground_samples=sum(int(r['boundary_vanished_foreground_samples_5']) for r in history))
        if (formal / 'validation_metrics.csv').is_file():
            report['formal_runs'][name]['validation_history'] = list(csv.DictReader(
                (formal / 'validation_metrics.csv').open()))
    differences = [k for k in configs[0]['args'] if configs[0]['args'][k] != configs[1]['args'][k]]
    assert differences == ['boundary_mode']
    initial_differences = {k: abs(float(first_steps[0][k])-float(first_steps[1][k]))
                           for k in first_steps[0] if 'loss' in k}
    assert max(initial_differences.values()) < 1e-6
    report['smoke_comparison'] = dict(configuration_differences=differences,
        first_step_loss_absolute_differences=initial_differences,
        source_differences=[k for k in configs[0]['source_sha256'] if
                           configs[0]['source_sha256'][k] != configs[1]['source_sha256'][k]],
        source_difference_explanation='prediction.py gained an optional metrics_json CLI argument between smoke launches; validation implementation is the same. Formal A/B uses one shared frozen snapshot.')
    if len(formal_configs) == 2:
        differences = [k for k in formal_configs[0]['args'] if
                       formal_configs[0]['args'][k] != formal_configs[1]['args'][k]]
        assert differences == ['boundary_mode']
        assert formal_configs[0]['source_sha256'] == formal_configs[1]['source_sha256']
        report['formal_comparison'] = dict(configuration_differences=differences,
            source_hashes_identical=True, same_initialization_verified_by_automated_tests=True)
        if len(formal_histories) == 2:
            keys = [f'boundary_ratios_per_sample_{i}' for i in range(1, 6)]
            assert all(a[k] == b[k] for a, b in zip(*formal_histories) for k in keys)
            report['formal_comparison']['matching_target_ratio_iterations'] = min(map(len, formal_histories))
    unchanged = {}
    for line in (HERE / 'reference/source_before.sha256').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        if name in ('model/vnet.py', 'utils/losses.py', 'dataloader/dataset.py',
                    'dataloader/TwoStreamBatchSampler.py', 'utils/transforms.py'):
            unchanged[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    assert all(unchanged.values())
    report['unchanged_core_files'] = unchanged
    report['limitations'] = [
        'This A/B comparison uses one seed and one dataset; results do not establish multi-seed statistical significance.',
        'LA train-time validation continues to use baseline test.list.',
        'Exact mid-epoch DataLoader/sampler resume is not implemented; model-only resume is rejected for boundary experiments.',
        'Fixed 3x3x3 morphology makes coarse boundaries relatively thick; coverage is logged throughout training.',
        'Formal random crops include occasional E5 coverage >=80%, unlike the 16 fixed audit crops; both modes log the same targets without unilateral changes.',
    ]
    (HERE / 'validation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(automated_tests=report['automated_tests']['tests_run'],
                         smoke_runs={k:v['status'] for k,v in report['smoke_runs'].items()},
                         formal_runs=report['formal_runs'], unchanged_core=unchanged), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
