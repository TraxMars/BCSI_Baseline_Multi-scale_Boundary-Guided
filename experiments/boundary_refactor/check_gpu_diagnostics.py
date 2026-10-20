"""Two real-LA GPU updates per trial: diagnostics disabled versus enabled.

Four trials / eight optimizer updates total; no checkpoints or old-result writes.
Both bitwise equality and numerical tolerance are reported, never conflated.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

from dataloader.dataset import build_Dataset
from trainer import Trainer, update_ema_variables
from utils.transforms import RandomCrop, RandomRotFlip, ToTensor


def config(mode, interval):
    return SimpleNamespace(
        in_channels=1, num_classes=2, device='cuda:0', base_lr=0.01,
        max_iterations=30000, labeled_bs=2, batch_size=4, ema_decay=0.9,
        consistency=0.1, consistency_rampup=200., boundary_mode=mode,
        boundary_weight=.1, boundary_padding_mode='zero', guidance_mode='none',
        auxiliary_mode='none', decoder_aux_weight=.1, final_supervision_weight=1.,
        gradient_diagnostic_interval=interval, gradient_cosine_threshold=-.2,
        gradient_ratio_threshold=.1, patch_size=[112,112,80], dataset='LA')


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_copy(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def snapshot(trainer):
    named = [('student.'+name,p) for name,p in trainer.model.named_parameters()]
    named += [('heads.'+name,p) for name,p in trainer.boundary.named_parameters()]
    gradients = {name:None if p.grad is None else cpu_copy(p.grad) for name,p in named}
    momentum = {name:cpu_copy(trainer.optimizer.state.get(p,{}).get('momentum_buffer'))
                for name,p in named}
    return dict(student=cpu_copy(trainer.model.state_dict()),
                teacher=cpu_copy(trainer.ema_model.state_dict()),
                heads=cpu_copy(trainer.boundary.state_dict()), gradients=gradients,
                momentum=momentum, teacher_gradients={name:None if p.grad is None
                    else cpu_copy(p.grad) for name,p in trainer.ema_model.named_parameters()},
                cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=[state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
                # Counting wraps ``step`` with a callable mock; this is test
                # instrumentation, not an LR-scheduler state variable.
                scheduler=copy.deepcopy({key:value for key,value in
                    trainer.scheduler.state_dict().items() if not callable(value)}))


def compare(first, second, rtol=1e-5, atol=1e-7):
    leaves=[]
    def walk(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():
                leaves.append(dict(path=path,bitwise_equal=False,within_tolerance=False,
                                   max_abs_error=None,reason='different keys'))
                return
            for key in a: walk(a[key],b[key],f'{path}.{key}')
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):
                leaves.append(dict(path=path,bitwise_equal=False,within_tolerance=False,
                                   max_abs_error=None,reason='different lengths'))
                return
            for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f'{path}[{i}]')
        elif torch.is_tensor(a) and torch.is_tensor(b):
            exact=torch.equal(a,b)
            same_shape=a.shape==b.shape and a.dtype==b.dtype
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            numerical=bool(same_shape and finite and (exact or torch.allclose(a,b,rtol=rtol,atol=atol)))
            error=float((a.double()-b.double()).abs().max()) if same_shape and finite and a.numel() else 0. if exact else None
            # RNG and integer buffers must always agree exactly.
            if 'rng' in path or not a.is_floating_point(): numerical=exact
            leaves.append(dict(path=path,bitwise_equal=exact,within_tolerance=numerical,
                               max_abs_error=error,finite=finite))
        else:
            exact=type(a) is type(b) and a==b
            leaves.append(dict(path=path,bitwise_equal=exact,within_tolerance=exact,
                               max_abs_error=0. if exact else None))
    walk(first,second,'state')
    unequal=[leaf for leaf in leaves if not leaf['bitwise_equal']]
    outside=[leaf for leaf in leaves if not leaf['within_tolerance']]
    return dict(bitwise_equal=not unequal,within_tolerance=not outside,
                max_abs_error=max((leaf['max_abs_error'] or 0 for leaf in leaves),default=0.),
                compared_leaves=len(leaves),unequal_leaves=unequal,outside_tolerance=outside,
                groups={name:dict(bitwise_equal=all(x['bitwise_equal'] for x in leaves if x['path'].startswith('state.'+name)),
                                 within_tolerance=all(x['within_tolerance'] for x in leaves if x['path'].startswith('state.'+name)),
                                 max_abs_error=max((x['max_abs_error'] or 0 for x in leaves if x['path'].startswith('state.'+name)),default=0.))
                        for name in first})


def real_batch(data_path):
    args=config('bfa_deep',0)
    transform=transforms.Compose([RandomRotFlip(),RandomCrop(args.patch_size),ToTensor()])
    dataset=build_Dataset(args,data_dir=str(data_path),split='train_LA',transform=transform)
    indices=[0,1,8,9]
    samples=[]; records=[]
    for index,seed in zip(indices,[42,43,44,45]):
        random.seed(seed); np.random.seed(seed)
        item=dataset[index]; samples.append(item)
        records.append(dict(train_index=index,augmentation_seed=seed,
            labeled=index<8,case_path=dataset.sample_list[index],
            image_sha256=hashlib.sha256(item['image'].numpy().tobytes()).hexdigest(),
            label_sha256=hashlib.sha256(item['label'].numpy().tobytes()).hexdigest()))
    return {key:torch.stack([item[key] for item in samples]) for key in ('image','label')},records


def trial(mode,interval,batch):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trainer=Trainer(config(mode,interval))
    captures=[snapshot(trainer)]; rows=[]; diagnostics=[]
    with tempfile.TemporaryDirectory(prefix='bcsi_gpu_diag_') as directory:
        with patch.object(trainer.optimizer,'step',wraps=trainer.optimizer.step) as step, \
             patch.object(trainer.scheduler,'step',wraps=trainer.scheduler.step) as schedule, \
             patch('trainer.update_ema_variables',wraps=update_ema_variables) as ema:
            for iteration in range(2):
                torch.manual_seed(1000+iteration); torch.cuda.manual_seed_all(1000+iteration)
                rows.append(trainer.train(batch,iteration,directory))
                torch.cuda.synchronize()
                captures.append(snapshot(trainer))
            calls=dict(optimizer=step.call_count,scheduler=schedule.call_count,ema=ema.call_count)
        path=Path(directory)/'gradient_diagnostics.csv'
        if path.is_file():
            diagnostics=list(csv.DictReader(path.open()))
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f'{mode} diagnostics_interval={interval}: {calls}, diagnostic_rows={len(diagnostics)}',flush=True)
    return dict(captures=captures,metrics=rows,calls=calls,diagnostic_rows=diagnostics)


def probe_trilinear():
    """No model/optimizer: check the required CUDA upsampling operator itself."""
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    source=torch.randn(2,1,14,14,10,device='cuda',requires_grad=True)
    upsampled=F.interpolate(source,size=[112,112,80],mode='trilinear',align_corners=True)
    output_gradient=torch.randn_like(upsampled)
    gradients=[torch.autograd.grad(upsampled,source,output_gradient,retain_graph=True)[0]
               for _ in range(4)]
    cuda_comparisons=[compare({'gradient':gradients[0].cpu()},
                              {'gradient':g.cpu()}) for g in gradients[1:]]
    cpu_source=source.detach().cpu().requires_grad_()
    cpu_output=F.interpolate(cpu_source,size=[112,112,80],mode='trilinear',align_corners=True)
    cpu_grads=[torch.autograd.grad(cpu_output,cpu_source,output_gradient.cpu(),retain_graph=True)[0]
               for _ in range(2)]
    old_setting=torch.are_deterministic_algorithms_enabled()
    strict_error=None
    try:
        torch.use_deterministic_algorithms(True)
        torch.autograd.grad(upsampled,source,output_gradient,retain_graph=True)
    except RuntimeError as error:
        strict_error=str(error)
    finally:
        torch.use_deterministic_algorithms(old_setting)
    return dict(purpose='Zero optimizer updates; repeated backward of the SAME required trilinear interpolation graph',
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,optimizer_updates=0,
        cuda_same_graph_repeats=cuda_comparisons,
        cuda_bitwise_equal=all(x['bitwise_equal'] for x in cuda_comparisons),
        cpu_same_graph_bitwise_equal=torch.equal(cpu_grads[0],cpu_grads[1]),
        strict_deterministic_backward_supported=strict_error is None,
        strict_deterministic_error=strict_error,
        source_note='Installed torch/__init__.py lines1391-1397 and nn/functional.py line4605 document nondeterministic CUDA trilinear backward.',
        conclusion='This probe can confirm an operator-level reproducibility limitation; it does not prove that every whole-model on/off difference is caused by this operator.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=Path,default=Path('/aita_data/home/aita_duanxuelong/Dataset/LA'))
    parser.add_argument('--report',type=Path,default=ROOT/'experiments/boundary_refactor/artifacts/gpu_diagnostic_invariance.json')
    parser.add_argument('--probe_operator_only',action='store_true',
                        help='Run only the zero-update CUDA interpolation reproducibility probe')
    args=parser.parse_args()
    started=time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('A real CUDA device is required; this check cannot be reported as passed on CPU')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if args.probe_operator_only:
        report=probe_trilinear()
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
        return
    batch,records=real_batch(args.data_path)
    report=dict(date='2026-10-20',purpose='Real-LA two-step GPU diagnostic non-interference check',
        torch_version=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(0),patch_size=[112,112,80],batch_size=4,labeled_bs=2,
        input_records=records,augmentations='Original RandomRotFlip + RandomCrop + ToTensor only; fixed per-case seeds42..45',
        rtol=1e-5,atol=1e-7,source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in [Path('trainer.py'),Path('model/vnet.py'),Path('utils/bfa_supervision.py'),Path('utils/gradient_diagnostics.py')]},
        experiments={},scope='8 total optimizer/EMA/scheduler updates; temporary CSV only, no checkpoints or formal training')
    for mode in ('bfa_deep','bfa_local'):
        baseline=trial(mode,0,batch); enabled=trial(mode,1,batch)
        comparisons=[compare(a,b) for a,b in zip(baseline['captures'],enabled['captures'])]
        report['experiments'][mode]=dict(initialization=comparisons[0],steps=comparisons[1:],
            disabled_calls=baseline['calls'],enabled_calls=enabled['calls'],
            disabled_metrics=baseline['metrics'],enabled_metrics=enabled['metrics'],
            diagnostic_rows=enabled['diagnostic_rows'],
            bitwise_equal=all(x['bitwise_equal'] for x in comparisons),
            within_tolerance=all(x['within_tolerance'] for x in comparisons))
        print(f'{mode}: bitwise_equal={report["experiments"][mode]["bitwise_equal"]}, '
              f'within_tolerance={report["experiments"][mode]["within_tolerance"]}, '
              f'max_abs_error={max(x["max_abs_error"] for x in comparisons):.9g}',flush=True)
    report['elapsed_seconds']=time.monotonic()-started
    report['passed']=all(x['within_tolerance'] for x in report['experiments'].values()) and all(
        x[key]==dict(optimizer=2,scheduler=2,ema=2) for x in report['experiments'].values()
        for key in ('disabled_calls','enabled_calls'))
    report['bitwise_equal']=all(x['bitwise_equal'] for x in report['experiments'].values())
    report['interpretation']=('All recorded state and RNG comparisons are bitwise equal.' if report['bitwise_equal']
        else 'Bitwise differences are reported in unequal_leaves; tolerance success alone does not prove identical optimization. GPU floating-point rounding must be considered and no difference is hidden.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['passed']:
        raise AssertionError('Diagnostic invariance check failed; inspect outside_tolerance and call counts in the written report')
    print(json.dumps({key:report[key] for key in ('passed','bitwise_equal','elapsed_seconds','interpretation')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
