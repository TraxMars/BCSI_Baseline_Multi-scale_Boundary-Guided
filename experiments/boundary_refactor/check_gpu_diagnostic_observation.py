"""One retained real-LA graph per mode; diagnose after actual backward.

No optimizer, scheduler, or EMA updates. Unlike independent GPU training runs,
this directly observes whether a diagnostic call mutates an existing gradient,
model/buffer state, or RNG stream. It does not claim GPU backward determinism.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import torch

from check_gpu_diagnostics import compare, config, real_batch, snapshot
from trainer import Trainer, update_ema_variables
from utils.gradient_diagnostics import boundary_gradient_diagnostics


def observe(mode,batch):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trainer=Trainer(config(mode,1))
    torch.manual_seed(1000); torch.cuda.manual_seed_all(1000)
    captured={}
    class StopBeforeUpdate(RuntimeError): pass

    def observe_call(main_loss,boundary,stage_parameters,heads,boundary_weight,
                     boundary_mode,*positional,**keyword):
        torch.cuda.synchronize()
        before=snapshot(trainer)
        # This hook is in the actual Trainer.train path; nonempty .grad proves
        # that both actual backward and any local accumulation already ran.
        captured['actual_gradient_tensors']=sum(p.grad is not None
            for group in trainer.optimizer.param_groups for p in group['params'])
        captured['actual_gradients_finite']=all(bool(torch.isfinite(p.grad).all())
            for group in trainer.optimizer.param_groups for p in group['params'] if p.grad is not None)
        captured['head_gradients_present']=all(p.grad is not None for p in trainer.boundary.parameters())
        diagnostics=boundary_gradient_diagnostics(main_loss,boundary,stage_parameters,
            heads,boundary_weight,boundary_mode,*positional,**keyword)
        torch.cuda.synchronize()
        after=snapshot(trainer)
        captured['comparison']=compare(before,after)
        captured['diagnostics']=diagnostics
        captured['losses']=dict(main_MT=float(main_loss),boundary=float(boundary.total_loss),
            total=float(main_loss+boundary_weight*boundary.total_loss))
        return diagnostics

    with tempfile.TemporaryDirectory(prefix='bcsi_gpu_observe_') as directory, \
         patch.object(trainer.optimizer,'step',side_effect=StopBeforeUpdate) as step, \
         patch.object(trainer.scheduler,'step',wraps=trainer.scheduler.step) as schedule, \
         patch('trainer.update_ema_variables',wraps=update_ema_variables) as ema, \
         patch('trainer.boundary_gradient_diagnostics',side_effect=observe_call) as diagnostic:
        try:
            trainer.train(batch,0,directory)
            raise AssertionError('Expected to stop before any optimizer update')
        except StopBeforeUpdate:
            pass
        calls=dict(optimizer_step_attempts_intercepted=step.call_count,
            actual_optimizer_updates=0,scheduler_step=schedule.call_count,
            ema_update=ema.call_count,diagnostic_calls=diagnostic.call_count)
        result=dict(**captured,calls=calls)
        result['passed']=(bool(captured) and captured['comparison']['bitwise_equal']
            and captured['actual_gradients_finite'] and captured['actual_gradient_tensors']>0
            and captured['head_gradients_present'] and step.call_count==1
            and schedule.call_count==0 and ema.call_count==0 and diagnostic.call_count==1)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f'{mode}: bitwise_equal={result["comparison"]["bitwise_equal"]}, '
          f'max_abs_error={result["comparison"]["max_abs_error"]}, '
          f'actual_gradient_tensors={result["actual_gradient_tensors"]}, calls={calls}',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=Path,default=Path('/aita_data/home/aita_duanxuelong/Dataset/LA'))
    parser.add_argument('--report',type=Path,default=ROOT/'experiments/boundary_refactor/artifacts/gpu_diagnostic_observation.json')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Real CUDA is required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    batch,records=real_batch(args.data_path)
    report=dict(date='2026-10-20',purpose='Same-graph post-backward observational diagnostic non-mutation',
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),patch_size=[112,112,80],
        batch_size=4,labeled_bs=2,input_records=records,
        optimizer_updates=0,scheduler_updates=0,ema_updates=0,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['trainer.py','utils/gradient_diagnostics.py','utils/bfa_supervision.py']},
        modes={mode:observe(mode,batch) for mode in ('bfa_deep','bfa_local')})
    report['passed']=all(item['passed'] for item in report['modes'].values())
    report['interpretation']=('Only mutation of the SAME existing gradients/state/RNG is tested. '
        'This result does not erase the separate strict cross-run CUDA invariance failure, '
        'and does not assert that all cross-run differences have one confirmed cause.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['passed']: raise AssertionError('Same-graph diagnostic call mutated observed state or failed validity checks')
    print('PASS: deep/local diagnostic calls preserve every observed tensor and RNG stream bitwise; zero optimizer/EMA/scheduler updates.',flush=True)


if __name__=='__main__': main()
