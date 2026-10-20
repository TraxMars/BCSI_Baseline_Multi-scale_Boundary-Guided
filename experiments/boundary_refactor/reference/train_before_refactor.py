import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from utils.transforms import RandomRotFlip, RandomCrop, ToTensor

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer import Trainer
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/opt/data/private/data_3D',
                    help='Dataset root. For LA/Pancreas/BraTS2019 this is the dataset directory itself.')

# parser.add_argument('--dataset', type=str, default='/2018LA_Seg_Training Set',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/Pancreas',
#                     help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='BraTS2019',
                    choices=['LA', 'Pancreas', 'BraTS2019', 'Lung',
                             '/2018LA_Seg_Training Set', '/Pancreas', '/BraTS2019', '/Lung'],
                    help='Dataset name. The short names support the local dataset layout.')
# parser.add_argument('--dataset', type=str, default='/Lung',
#                     help='Name of Experiment')


parser.add_argument('--labeled_num', type=int, default=10,
                    help='Percentage of label quantity')

parser.add_argument('--nms', type=int,  default=True,
                    help='output channel of network')

parser.add_argument('--in_channels', type=int,  default=1,
                    help='output channel of network')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 96],
                    help='patch size of network input')

# parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
#                     help='patch size of network input')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')

parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--output_dir', type=str, default='./Results', help='Directory for logs and checkpoints')
parser.add_argument('--test_interval', type=int, default=200, help='Validate every N iterations (0 disables validation)')
parser.add_argument('--resume_model_path', type=str, default=None,
                    help='Model-only checkpoint to restore after an interrupted run')
parser.add_argument('--start_iteration', type=int, default=0,
                    help='Iteration represented by --resume_model_path')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--boundary_mode', choices=['none', 'deep', 'local'], default='none')
parser.add_argument('--boundary_weight', type=float, default=0.1)
parser.add_argument('--guidance_mode', choices=['none'], default='none',
                    help='Legacy BGS is disabled in this repository')

args = parser.parse_args()
if args.boundary_mode != 'none':
    if args.resume_model_path or args.start_iteration:
        parser.error('Boundary A/B must start independently; model-only resume loses auxiliary/optimizer state')
    if not 0 < args.labeled_bs < args.batch_size:
        parser.error('Boundary MT requires both labeled and unlabeled samples')
    if not np.isfinite(args.boundary_weight) or args.boundary_weight < 0:
        parser.error('boundary_weight must be finite and nonnegative')


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def boundary_status(snapshot_path, state, iteration, **extra):
    path = Path(snapshot_path) / 'run_status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(state=state, iteration=iteration,
                                        boundary_mode=args.boundary_mode, **extra), indent=2))
    temporary.replace(path)


def train(args, snapshot_path):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    patch_size = args.patch_size

    # model
    trainer = Trainer(args)
    if args.resume_model_path:
        checkpoint = torch.load(args.resume_model_path, map_location=args.device)
        trainer.model.load_state_dict(checkpoint)
        trainer.ema_model.load_state_dict(checkpoint)
        # Existing checkpoints in this repository contain model weights only.
        # Align the LR scheduler with the saved iteration; optimizer momentum is
        # necessarily restarted because it was not stored in the checkpoint.
        trainer.scheduler.step(args.start_iteration)
        logging.info('restored model weights from %s at iteration %d',
                     args.resume_model_path, args.start_iteration)

    # dataset
    if args.dataset in ('LA', "/2018LA_Seg_Training Set"):
        data_dir = args.data_path if args.dataset == 'LA' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_LA",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size),  ToTensor()]))
    elif args.dataset in ('Pancreas', "/Pancreas"):
        data_dir = args.data_path if args.dataset == 'Pancreas' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_Pancreas",
                                      transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]))
    elif args.dataset in ('BraTS2019', "/BraTS2019"):
        data_dir = args.data_path if args.dataset == 'BraTS2019' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_BraTS2019",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    elif args.dataset == "/Lung":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_Lung",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    
    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    logging.info("{} iterations per epoch".format(len(train_loader)))
    if args.boundary_mode != 'none':
        data_root = Path(train_dataset.data_dir)
        split_root = data_root
        if 'BraTS' in args.dataset:
            split_root = data_root / ('Flods' if (data_root / 'Flods').is_dir() else 'data')
        metadata = dict(labeled_indices=labeled_idxs, unlabeled_indices=unlabeled_idxs,
                        labeled_case_paths=train_dataset.sample_list[:labeled_slice],
                        splits={})
        for name in ('train', 'val', 'test'):
            source = split_root / (name + '.list')
            if source.is_file():
                metadata['splits'][name] = dict(path=str(source),
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    ids=source.read_text().splitlines())
        (Path(snapshot_path) / 'data_split_manifest.json').write_text(
            json.dumps(metadata, indent=2))
    # max_epoch = max_iterations // len(train_loader) + 1
    max_epoch = (max_iterations - args.start_iteration) // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = args.start_iteration

    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            row = trainer.train(sampled_batch, iter_num, snapshot_path)
            iter_num = iter_num + 1
            if args.boundary_mode != 'none':
                boundary_status(snapshot_path, 'training', iter_num,
                                total_loss=row['total_loss'])
            if args.test_interval and iter_num > 0 and iter_num % args.test_interval == 0:
                trainer.test(snapshot_path, iter_num)
            if iter_num >= max_iterations:
                if args.boundary_mode != 'none':
                    trainer.save_boundary_state(snapshot_path, iter_num)
                    boundary_status(snapshot_path, 'completed', iter_num,
                                    best_validation_dice=trainer.best_performance)
                return


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        run_seed = args.seed + fold
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dataset_name = args.dataset.strip('/').replace(' ', '_')
        snapshot_path = os.path.join(args.output_dir, "result_{}_{}l".format(dataset_name, args.labeled_num), "fold_" + str(fold))
        if args.boundary_mode != 'none':
            experiment = 'Boundary_A' if args.boundary_mode == 'deep' else 'Boundary_B'
            snapshot_path = os.path.join(args.output_dir, experiment, f'seed_{args.seed}',
                                        f'result_{dataset_name}_{args.labeled_num}l', f'fold_{fold}')
            # Fail before writing ANY source, log, config or checkpoint in an old run.
            os.makedirs(snapshot_path, exist_ok=False)

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train.py", snapshot_path + "/code/train.py")
        shutil.copyfile("./trainer.py", snapshot_path + "/code/trainer.py")
        shutil.copyfile("./model/vnet.py", snapshot_path + "/code/vnet.py")
        if args.boundary_mode != 'none':
            sources = ['train.py', 'trainer.py', 'prediction.py', 'model/vnet.py',
                       'utils/boundary_supervision.py', 'utils/losses.py',
                       'utils/utils.py', 'utils/transforms.py',
                       'dataloader/dataset.py', 'dataloader/TwoStreamBatchSampler.py']
            hashes = {}
            for relative in sources:
                source = Path(relative)
                destination = Path(snapshot_path) / 'code' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
            config = dict(args=vars(args), run_seed=run_seed, fold=fold,
                          command=[sys.executable, *sys.argv], source_sha256=hashes,
                          torch_version=torch.__version__,
                          boundary_target='nearest foreground >0; 3x3x3 morphological gradient; replicate padding',
                          boundary_loss='mean(0.5 BCEWithLogits + 0.5 spatial soft Dice), five equal scales',
                          scheduler='PolyWarmRestartScheduler(power=0.9, warm_restart_iters=8000)')
            (Path(snapshot_path) / 'experiment_config.json').write_text(
                json.dumps(config, indent=2))

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        try:
            train(args, snapshot_path)
        except BaseException as error:
            if args.boundary_mode != 'none':
                boundary_status(snapshot_path, 'failed', None, error=repr(error))
            raise
