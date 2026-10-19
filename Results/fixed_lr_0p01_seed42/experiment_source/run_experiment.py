"""Isolated constant-LR Mean-Teacher experiments: three datasets x 10/20%."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
RUN_ROOT = ROOT / 'Results' / 'fixed_lr_0p01_seed42'
SNAPSHOT = RUN_ROOT / 'source_snapshot'
STATUS_PATH = RUN_ROOT / 'run_status.json'
SHANGHAI = timezone(timedelta(hours=8))
DATASETS = (
    ('LA', Path('/home/duanxuelong/Dataset/LA'), [112, 112, 80], 80, {10: 8, 20: 16}),
    ('Pancreas', Path('/home/duanxuelong/Dataset/Pancreas'), [96, 96, 96], 62, {10: 6, 20: 12}),
    ('BraTS2019', Path('/home/duanxuelong/Dataset/brats19'), [96, 96, 96], 250, {10: 25, 20: 50}),
)
ORIGINAL_SCHEDULER = (
    '        self.scheduler = PolyWarmRestartScheduler(\n'
    '            self.optimizer, base_lr=args.base_lr, max_iters=args.max_iterations,\n'
    '            power=0.9, warm_restart_iters=8000)'
)
CONSTANT_SCHEDULER = (
    '        # This experiment keeps the optimizer LR fixed for all 30,000 steps.\n'
    '        self.scheduler = torch.optim.lr_scheduler.LambdaLR(\n'
    '            self.optimizer, lr_lambda=lambda step: 1.0)'
)


def now():
    return datetime.now(SHANGHAI).isoformat(timespec='seconds')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def write_status(status):
    status['updated_at'] = now()
    write_json(STATUS_PATH, status)


def constant_lr_trainer(source):
    if source.count(ORIGINAL_SCHEDULER) != 1:
        raise RuntimeError('Baseline scheduler changed; inspect before preparing a new snapshot')
    return source.replace(ORIGINAL_SCHEDULER, CONSTANT_SCHEDULER, 1)


def case_path(dataset, data_root, case_id):
    if dataset == 'LA':
        return data_root / '2018LA_Seg_TrainingSet' / case_id / 'mri_norm2.h5'
    if dataset == 'Pancreas':
        return data_root / 'Pancreas_h5' / (case_id + '_norm.h5')
    return data_root / 'Flods' / (case_id + '.h5')


def inspect_splits():
    splits = {}
    for dataset, data_root, patch, total, counts in DATASETS:
        list_root = data_root / 'Flods' if dataset == 'BraTS2019' else data_root
        names = ('train', 'val', 'test') if dataset == 'BraTS2019' else ('train', 'test')
        lists = {}
        for name in names:
            path = list_root / (name + '.list')
            ids = path.read_text().splitlines()
            if not ids or any(not case_id.strip() for case_id in ids) or len(ids) != len(set(ids)):
                raise ValueError(f'{path}: empty or duplicate case IDs')
            for case_id in ids:
                if not case_path(dataset, data_root, case_id).is_file():
                    raise FileNotFoundError(case_path(dataset, data_root, case_id))
            lists[name] = dict(path=str(path), sha256=sha256(path), ids=ids)
        if len(lists['train']['ids']) != total:
            raise ValueError(f'{dataset}: expected {total} training cases')
        for index, name in enumerate(names):
            for other in names[index + 1:]:
                if set(lists[name]['ids']) & set(lists[other]['ids']):
                    raise ValueError(f'{dataset}: {name}/{other} split overlap')
        splits[dataset] = lists
    return splits


def prepare(gpus):
    if len(set(gpus)) != 3 or any(gpu < 0 for gpu in gpus):
        raise ValueError('Use three distinct physical GPU indices')
    splits = inspect_splits()
    trainer_source = constant_lr_trainer((ROOT / 'trainer.py').read_text())
    # Refuse to overwrite any existing run, including unfinished runs.
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    SNAPSHOT.mkdir()
    sources = [ROOT / name for name in ('train.py', 'trainer.py', 'prediction.py',
                                      'readme.md', 'BASELINE_AUDIT.md')]
    for directory in ('model', 'utils', 'dataloader'):
        sources.extend((ROOT / directory).rglob('*.py'))
    baseline_hashes = {}
    snapshot_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = SNAPSHOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == Path('trainer.py'):
            destination.write_text(trainer_source)
        else:
            shutil.copy2(source, destination)
        baseline_hashes[str(relative)] = sha256(source)
        snapshot_hashes[str(relative)] = sha256(destination)
    experiment_copy = RUN_ROOT / 'experiment_source'
    experiment_copy.mkdir()
    for source in EXPERIMENT.iterdir():
        if source.suffix in ('.py', '.md'):
            shutil.copy2(source, experiment_copy / source.name)
    for dataset, lists in splits.items():
        split_directory = RUN_ROOT / 'data_splits' / dataset
        split_directory.mkdir(parents=True)
        for name, split in lists.items():
            shutil.copy2(split['path'], split_directory / (name + '.list'))
    jobs = []
    for (dataset, data_root, patch, total, counts), gpu in zip(DATASETS, gpus):
        for percent in (10, 20):
            key = f'{dataset}_{percent}pct'
            job_root = RUN_ROOT / f'seed_42_{percent}pct'
            job_root.mkdir(exist_ok=True)
            fold_root = job_root / f'result_{dataset}_{percent}l' / 'fold_0'
            command = [sys.executable, '-u', '-B', str(SNAPSHOT / 'train.py'),
                       '--dataset', dataset, '--data_path', str(data_root),
                       '--labeled_num', str(percent), '--seed', '42', '--n_fold', '1',
                       '--patch_size', *map(str, patch), '--batch_size', '4',
                       '--labeled_bs', '2', '--max_iterations', '30000',
                       '--base_lr', '0.01', '--test_interval', '200',
                       '--consistency', '0.1', '--consistency_rampup', '200',
                       '--ema_decay', '0.9', '--device', 'cuda:0',
                       '--output_dir', str(job_root)]
            jobs.append(dict(key=key, dataset=dataset, labeled_percent=percent, gpu=gpu,
                             train_count=total, labeled_count=counts[percent],
                             labeled_ids=splits[dataset]['train']['ids'][:counts[percent]],
                             patch_size=patch, command=command, cwd=str(SNAPSHOT),
                             fold_root=str(fold_root),
                             console_log=str(job_root / f'{dataset}_console.log'),
                             test_log=str(job_root / f'{dataset}_test.log'),
                             train_log=str(fold_root / 'log.txt')))
    metadata = dict(prepared_at=now(), seed=42, labeled_percents=[10, 20],
                    max_iterations=30000, base_lr=0.01, lr_schedule='constant',
                    test_interval=200, batch_size=4, labeled_bs=2,
                    consistency=0.1, consistency_rampup=200, ema_decay=0.9,
                    source_root=str(ROOT), source_snapshot=str(SNAPSHOT),
                    baseline_source_sha256=baseline_hashes,
                    snapshot_source_sha256=snapshot_hashes,
                    snapshot_changes={'trainer.py': 'PolyWarmRestartScheduler -> LambdaLR factor=1'},
                    data_splits=splits, jobs=jobs)
    write_json(RUN_ROOT / 'metadata.json', metadata)
    write_status(dict(state='prepared', seed=42, learning_rate=0.01,
                      lr_schedule='constant', max_iterations=30000,
                      experiments={job['key']: dict(
                          state='pending', dataset=job['dataset'],
                          labeled_percent=job['labeled_percent'], gpu=job['gpu'],
                          labeled_count=job['labeled_count'], current_iteration=0,
                          train_log=job['train_log'], console_log=job['console_log'])
                          for job in jobs}))
    print(f'Prepared six experiments at {RUN_ROOT}', flush=True)


def update_progress(entry):
    logfile = Path(entry['train_log'])
    if not logfile.exists():
        return
    with logfile.open('rb') as handle:
        handle.seek(max(0, logfile.stat().st_size - 65536))
        lines = handle.read().decode(errors='replace')
    losses = re.findall(r'iteration (\d+) : loss : (\S+).*? lr : (\S+)', lines)
    for iteration, loss, lr in losses:
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f'Nonfinite loss at iteration {iteration}')
        if not math.isclose(float(lr), 0.01, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f'LR changed to {lr} at iteration {iteration}')
    if losses:
        iteration, loss, lr = losses[-1]
        entry.update(current_iteration=int(iteration) + 1,
                     last_loss=float(loss), learning_rate=float(lr))
    for iteration, dice, hd95 in re.findall(
            r'iteration (\d+) : mean_dice : (\S+) mean_hd95 : (\S+)', lines):
        dice, hd95 = float(dice), float(hd95)
        if not math.isfinite(dice) or not math.isfinite(hd95):
            raise FloatingPointError(f'Nonfinite validation at iteration {iteration}')
        entry.update(last_validation_iteration=int(iteration), last_dice=dice,
                     last_hd95=hd95, best_dice=max(entry.get('best_dice', 0.0), dice))


def best_checkpoint(job):
    checkpoints = list(Path(job['fold_root']).glob('Model_iter_*.pth'))
    if not checkpoints:
        raise RuntimeError('Training did not save a best checkpoint')
    # The baseline saves weights only when its validation Dice improves.
    return max(checkpoints, key=lambda path: int(path.stem.split('_')[-1]))


def launch(command, logfile, job):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']),
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with open(logfile, 'x') as console:
        return subprocess.Popen(command, cwd=job['cwd'], env=environment,
                                stdin=subprocess.DEVNULL, stdout=console,
                                stderr=subprocess.STDOUT)


def test_command(job, checkpoint):
    train_command = job['command']
    data_path = train_command[train_command.index('--data_path') + 1]
    return [sys.executable, '-u', '-B', str(SNAPSHOT / 'prediction.py'),
            '--dataset', job['dataset'], '--data_path', data_path,
            '--patch_size', *map(str, job['patch_size']),
            '--model_path', str(checkpoint), '--device', 'cuda:0']


def collect_test_metrics(job, metadata):
    lines = Path(job['test_log']).read_text().splitlines()
    rows = []
    for line in lines:
        fields = line.split()
        if len(fields) == 4:
            try:
                values = list(map(float, fields))
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError('Nonfinite test metric')
            rows.append(values)
    case_ids = metadata['data_splits'][job['dataset']]['test']['ids']
    if len(rows) != len(case_ids):
        raise RuntimeError(f'Expected {len(case_ids)} test cases; got {len(rows)}')
    output = Path(job['fold_root']) / 'per_case_metrics.csv'
    with output.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'dice', 'iou', 'hd95', 'asd'])
        writer.writerows([case_id, *values] for case_id, values in zip(case_ids, rows))
    metrics = {name: sum(row[index] for row in rows) / len(rows)
               for index, name in enumerate(('dice', 'iou', 'hd95', 'asd'))}
    metrics['case_count'] = len(rows)
    write_json(Path(job['fold_root']) / 'test_metrics.json', metrics)
    return metrics


def write_summary(status):
    output = RUN_ROOT / 'summary.csv'
    with output.with_suffix('.csv.tmp').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'labeled_percent', 'seed', 'lr', 'state',
                         'current_iteration', 'best_validation_dice', 'best_checkpoint',
                         'test_dice', 'test_iou', 'test_hd95', 'test_asd'])
        for entry in status['experiments'].values():
            metrics = entry.get('test_metrics', {})
            writer.writerow([entry['dataset'], entry['labeled_percent'], 42, 0.01,
                             entry['state'], entry['current_iteration'],
                             entry.get('best_dice', ''), entry.get('best_checkpoint', ''),
                             *[metrics.get(name, '') for name in ('dice', 'iou', 'hd95', 'asd')]])
    output.with_suffix('.csv.tmp').replace(output)


def run():
    metadata = json.loads((RUN_ROOT / 'metadata.json').read_text())
    with (RUN_ROOT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads(STATUS_PATH.read_text())
        if status['state'] != 'prepared':
            raise RuntimeError('Run already started; refusing duplicate or overwritten training')
        for relative, digest in metadata['snapshot_source_sha256'].items():
            if sha256(SNAPSHOT / relative) != digest:
                raise RuntimeError(f'Snapshot changed: {relative}')
        for lists in metadata['data_splits'].values():
            for split in lists.values():
                if sha256(split['path']) != split['sha256']:
                    raise RuntimeError(f'Data split changed: {split["path"]}')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; run outside the restricted sandbox')
        for gpu in {job['gpu'] for job in metadata['jobs']}:
            if gpu >= torch.cuda.device_count():
                raise RuntimeError(f'Physical GPU {gpu} is unavailable')
        active = {}
        jobs = {job['key']: job for job in metadata['jobs']}
        status.update(state='running', launcher_pid=os.getpid(), started_at=now())

        def interrupted(signum, frame):
            raise RuntimeError(f'Launcher interrupted by signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            for key, job in jobs.items():
                process = launch(job['command'], job['console_log'], job)
                active[key] = process
                status['experiments'][key].update(state='training', pid=process.pid,
                                                   started_at=now())
                write_status(status)
                print(f'Started {key}: PID {process.pid}, GPU {job["gpu"]}', flush=True)
            while active:
                for key, process in list(active.items()):
                    entry, job = status['experiments'][key], jobs[key]
                    try:
                        if entry['state'] == 'training':
                            update_progress(entry)
                        exit_code = process.poll()
                        if exit_code is None:
                            continue
                        if exit_code != 0:
                            raise RuntimeError(f'{entry["state"]} exited with code {exit_code}')
                        if entry['state'] == 'training':
                            if (entry['current_iteration'] != 30000 or
                                    entry.get('last_validation_iteration') != 30000):
                                raise RuntimeError('Training ended before completing 30,000 steps and validation')
                            checkpoint = best_checkpoint(job)
                            command = test_command(job, checkpoint)
                            write_json(Path(job['fold_root']) / 'test_command.json', command)
                            active[key] = launch(command, job['test_log'], job)
                            entry.update(state='testing', pid=active[key].pid,
                                         training_finished_at=now(), best_checkpoint=str(checkpoint))
                            print(f'Testing {key}: {checkpoint.name}', flush=True)
                        else:
                            entry.update(state='completed', exit_code=0, finished_at=now(),
                                         test_metrics=collect_test_metrics(job, metadata))
                            active.pop(key)
                            print(f'Completed {key}: {entry["test_metrics"]}', flush=True)
                    except Exception as error:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                        entry.update(state='failed', error=repr(error),
                                     exit_code=process.returncode, finished_at=now())
                        active.pop(key)
                        print(f'Failed {key}: {error}', flush=True)
                write_status(status)
                write_summary(status)
                if active:
                    time.sleep(10)
        except BaseException as error:
            status['error'] = repr(error)
            for key, process in active.items():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                status['experiments'][key].update(state='failed', error=repr(error),
                                                  finished_at=now())
            status.update(state='failed', finished_at=now())
            write_status(status)
            write_summary(status)
            raise
        status.update(state='completed' if all(entry['state'] == 'completed'
                      for entry in status['experiments'].values()) else 'failed', finished_at=now())
        write_status(status)
        write_summary(status)
        if status['state'] != 'completed':
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=3, default=[2, 4, 6],
                        help='Physical GPUs for LA, Pancreas, BraTS2019; two jobs per GPU')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    prepare(arguments.gpus) if arguments.prepare else run()


if __name__ == '__main__':
    main()
