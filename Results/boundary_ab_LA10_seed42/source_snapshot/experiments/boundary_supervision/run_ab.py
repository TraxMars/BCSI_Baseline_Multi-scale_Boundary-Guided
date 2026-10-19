"""Recover a finished baseline's arguments; print or execute a fresh A/B pair.

Execution always reruns the mandatory gradient tests before any training.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    baseline = args.baseline_run or ROOT / 'Results' / 'seed_42' / f'result_{args.dataset}_{args.labeled_num}l' / 'fold_0'
    text = (baseline / 'log.txt').read_text()
    namespace = next(line[line.index('Namespace('):] for line in text.splitlines()
                     if 'Namespace(' in line)
    call = ast.parse(namespace, mode='eval').body
    settings = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    if (settings['dataset'] != args.dataset or settings['labeled_num'] != args.labeled_num
            or settings['seed'] != 42 or settings['max_iterations'] != 30000
            or 'iteration 30000 : mean_dice' not in text):
        raise ValueError('Select a completed matching 30k seed-42 baseline')
    saved_trainer = (baseline / 'code' / 'trainer.py').read_text()
    reference = (HERE / 'reference' / 'trainer_before.py').read_text()
    if saved_trainer != reference:
        raise ValueError('Selected baseline training code differs from audited MT reference')
    if digest(baseline / 'code' / 'vnet.py') != digest(ROOT / 'model' / 'vnet.py'):
        raise ValueError('Backbone differs from selected completed baseline')
    if 'warm_restart_iters=8000' not in saved_trainer:
        raise ValueError('Expected original warm-restart baseline, not fixed-LR experiment')
    split_root = args.data_path
    if args.dataset == 'BraTS2019':
        split_root = split_root / ('Flods' if (split_root / 'Flods').is_dir() else 'data')
    splits = {}
    for name in ('train', 'val', 'test'):
        current = split_root / (name + '.list')
        if current.is_file():
            archived = ROOT / 'Results' / 'fixed_lr_0p01_seed42' / 'data_splits' / args.dataset / current.name
            if not archived.is_file() or current.read_bytes() != archived.read_bytes():
                raise ValueError(f'Split must match archived baseline list: {current}')
            splits[name] = dict(path=str(current), sha256=digest(current))
    if not {'train', 'test'} <= splits.keys():
        raise ValueError('Missing baseline train/test lists')
    commands = []
    keys = ('dataset', 'labeled_num', 'seed', 'n_fold', 'batch_size', 'labeled_bs',
            'max_iterations', 'base_lr', 'test_interval', 'consistency',
            'consistency_rampup', 'ema_decay', 'in_channels', 'num_classes', 'nms')
    for mode in args.modes:
        command = [sys.executable, '-u', 'train.py']
        for key in keys:
            value = settings[key]
            command += ['--' + key, str(int(value) if isinstance(value, bool) else value)]
        command += ['--patch_size', *map(str, settings['patch_size']),
                    '--data_path', str(args.data_path), '--device', args.device,
                    '--guidance_mode', 'none', '--boundary_mode', mode,
                    '--boundary_weight', '0.1', '--output_dir', str(args.output_dir)]
        label = 'Boundary_A' if mode == 'deep' else 'Boundary_B'
        for fold in range(settings['n_fold']):
            target = args.output_dir / label / 'seed_42' / f'result_{args.dataset}_{args.labeled_num}l' / f'fold_{fold}'
            if target.exists():
                raise FileExistsError(f'Refusing an existing run: {target}')
        commands.append(command)
    manifest = dict(baseline=str(baseline), baseline_settings=settings,
                    baseline_log_sha256=digest(baseline / 'log.txt'),
                    original_data_path=settings['data_path'], actual_data_path=str(args.data_path),
                    splits=splits, commands=commands,
                    scheduler='original PolyWarmRestartScheduler, 8000-iteration cycles',
                    physical_gpus=args.gpus, training_started=False)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
    for command in commands:
        print(shlex.join(command), flush=True)
    return commands


def execute(args, commands):
    import torch
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('CUDA unavailable; no formal training started')
    if args.gpus is not None and len(args.gpus) != len(commands):
        raise ValueError('Provide one physical GPU per selected mode')
    run_root = args.output_dir / f'boundary_ab_{args.dataset}{args.labeled_num}_seed42'
    run_root.mkdir(parents=True, exist_ok=False)
    snapshot = run_root / 'source_snapshot'
    snapshot.mkdir()
    sources = ['train.py', 'trainer.py', 'prediction.py', 'BASELINE_AUDIT.md',
               'experiments/boundary_supervision/run_ab.py',
               'experiments/boundary_supervision/README.md',
               'experiments/boundary_supervision/test_boundary_supervision.py',
               'experiments/boundary_supervision/reference/trainer_before.py']
    for directory in ('model', 'utils', 'dataloader'):
        sources.extend(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob('*.py'))
    hashes = {}
    for relative in sources:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        hashes[relative] = digest(destination)
    (run_root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    if args.manifest:
        shutil.copy2(args.manifest, run_root / 'baseline_manifest.json')
    state = dict(state='testing', jobs={}, source_snapshot=str(snapshot))
    status = run_root / 'run_status.json'

    def save():
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(status)

    save()
    processes, handles = {}, []
    try:
        subprocess.run([sys.executable, str(snapshot / 'experiments/boundary_supervision/test_boundary_supervision.py'),
                        '--report', str(run_root / 'verification.json')], cwd=snapshot, check=True)
        state['state'] = 'training'
        for index, (mode, command) in enumerate(zip(args.modes, commands)):
            environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            if args.gpus is not None:
                environment['CUDA_VISIBLE_DEVICES'] = str(args.gpus[index])
            log = run_root / f'{mode}_console.log'
            handle = log.open('x')
            handles.append(handle)
            process = subprocess.Popen(command, cwd=snapshot, env=environment,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes[mode] = process
            state['jobs'][mode] = dict(state='training', pid=process.pid,
                command=command, console_log=str(log),
                physical_gpu=environment.get('CUDA_VISIBLE_DEVICES'))
            save()
        errors = []
        for mode, process in processes.items():
            entry = state['jobs'][mode]
            entry['exit_code'] = process.wait()
            if process.returncode:
                entry['state'] = 'failed'
                errors.append(mode)
                save()
                continue
            label = 'Boundary_A' if mode == 'deep' else 'Boundary_B'
            fold = args.output_dir / label / 'seed_42' / f'result_{args.dataset}_{args.labeled_num}l' / 'fold_0'
            training_status = json.loads((fold / 'run_status.json').read_text())
            if training_status['state'] != 'completed' or training_status['iteration'] != 30000:
                raise RuntimeError(f'{mode} did not complete 30k iterations')
            # Baseline writes Model_iter_* only when Teacher validation Dice improves.
            checkpoints = list(fold.glob('Model_iter_*.pth'))
            if not checkpoints:
                raise RuntimeError(f'{mode} has no best Teacher checkpoint')
            checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            train_command = entry['command']
            patch_index = train_command.index('--patch_size')
            test_command = [sys.executable, '-u', 'prediction.py', '--dataset', args.dataset,
                '--data_path', str(args.data_path), '--patch_size',
                *train_command[patch_index+1:patch_index+4], '--model_path', str(checkpoint),
                '--device', args.device, '--metrics_json', str(fold / 'test_metrics.json')]
            environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            if entry['physical_gpu']:
                environment['CUDA_VISIBLE_DEVICES'] = entry['physical_gpu']
            entry.update(state='testing', best_teacher=str(checkpoint), test_command=test_command)
            save()
            with (run_root / f'{mode}_test.log').open('x') as handle:
                subprocess.run(test_command, cwd=snapshot, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            entry.update(state='completed', test_metrics=json.loads((fold / 'test_metrics.json').read_text()))
            save()
        state['state'] = 'failed' if errors else 'completed'
        save()
        if errors:
            raise RuntimeError(f'Training failed: {errors}')
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        for mode, process in processes.items():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=60)
                state['jobs'][mode]['state'] = 'terminated'
        save()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['LA', 'Pancreas', 'BraTS2019'], default='LA')
    parser.add_argument('--labeled_num', type=int, default=10)
    parser.add_argument('--baseline_run', type=Path)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'Results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--modes', choices=['deep', 'local'], nargs='+', default=['deep', 'local'])
    parser.add_argument('--gpus', type=int, nargs='+',
                        help='Physical GPU per mode; independent A/B jobs run concurrently')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate modes are not allowed')
    commands = prepare(args)
    if args.execute:
        execute(args, commands)
