"""Print a matched MT command; execute only with an explicit --execute.

No training is launched by default. Smoke mode is capped at ten iterations.
Formal experiments are separate invocations after implementation validation.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPERIMENTS = ('BFA_Deep_A2', 'BFA_Local_B2', 'Decoder_DS_C')


def build_command(args):
    source = ROOT / 'Results/Boundary_A/seed_42/result_LA_10l/fold_0/experiment_config.json'
    baseline = json.loads(source.read_text())['args']
    if (baseline['dataset'] != 'LA' or baseline['labeled_num'] != 10 or
            baseline['seed'] != 42 or baseline['max_iterations'] != 30000):
        raise ValueError('Expected audited LA10 seed42 warm-restart configuration')
    keys = ('dataset', 'labeled_num', 'seed', 'n_fold', 'batch_size', 'labeled_bs',
            'base_lr', 'consistency', 'consistency_rampup', 'ema_decay',
            'in_channels', 'num_classes')
    command = [sys.executable, '-u', 'train.py']
    for key in keys:
        command.extend(['--' + key, str(baseline[key])])
    iterations = args.smoke_iterations if args.stage == 'smoke' else 30000
    test_interval = iterations if args.stage == 'smoke' else baseline['test_interval']
    command.extend(['--patch_size', *map(str, baseline['patch_size']),
                    '--data_path', str(args.data_path), '--device', args.device,
                    '--max_iterations', str(iterations), '--test_interval', str(test_interval),
                    '--output_dir', str(args.output_dir), '--guidance_mode', 'none'])
    if args.experiment == 'Decoder_DS_C':
        command.extend(['--boundary_mode', 'none', '--auxiliary_mode', 'decoder_deep_seg',
                        '--decoder_aux_weight', str(args.decoder_aux_weight),
                        '--final_supervision_weight', '1.0', '--gradient_diagnostic_interval', '0'])
    else:
        mode = 'bfa_deep' if args.experiment == 'BFA_Deep_A2' else 'bfa_local'
        interval = 1 if args.stage == 'smoke' else args.diagnostic_interval
        command.extend(['--boundary_mode', mode, '--auxiliary_mode', 'none',
                        '--boundary_weight', str(args.boundary_weight),
                        '--boundary_padding_mode', 'zero',
                        '--gradient_diagnostic_interval', str(interval),
                        '--gradient_cosine_threshold', '-0.2', '--gradient_ratio_threshold', '0.1'])
    return command


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--stage', choices=['smoke', 'formal'], default='smoke')
    parser.add_argument('--data_path', type=Path,
                        default=Path('/aita_data/home/aita_duanxuelong/Dataset/LA'))
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'Results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke_iterations', type=int, choices=range(1, 11), default=2)
    parser.add_argument('--boundary_weight', type=float, default=0.1)
    parser.add_argument('--decoder_aux_weight', type=float, default=0.1)
    parser.add_argument('--diagnostic_interval', type=int, default=200)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.execute:
        folder = args.output_dir / args.experiment / 'seed_42/result_LA_10l/fold_0'
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite an existing experiment: {folder}')
        # Mandatory gradient/MT/decoder checks guard any later manual launch.
        with tempfile.TemporaryDirectory(prefix='boundary-refactor-preflight-') as temporary:
            for script in ('test_boundary_refactor.py', 'test_decoder_supervision.py'):
                subprocess.run([sys.executable, str(HERE / script), '--report',
                                str(Path(temporary) / (script + '.json'))], cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, check=True)
