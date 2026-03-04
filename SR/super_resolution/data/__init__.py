import math
import importlib
import os.path as osp
from copy import deepcopy

from ..utils.basicsr_util import scandir, get_root_logger
from ..utils.data_util import EnlargedSampler, build_dataloader
from ..utils.registry import DATASET_REGISTRY

data_folder = osp.dirname(osp.abspath(__file__))
data_filenames = [osp.splitext(v)[0] for v in scandir(
    dir_path=data_folder, suffix='_dataset.py', recursive=False, full_path=False)]
root_basename = osp.basename(osp.dirname(osp.dirname(__file__)))
_data_modules = [importlib.import_module(f'{root_basename}.data.{v}') for v in data_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset


def create_train_val_test_dataloader(opt, train=False, val=False, test=False):
    """Create training, testing and validation datasets based on
        configuration options.

    Args:
        opt (ndarray): YAML configuration options.
        train (bool): Default: True. If True, get train dataloader from training configuration.
        val (bool): Default: True. If True, get val dataloader from validation configuration.
        test (bool): Default: True. If True, get test dataloader from testing configuration.

    Returns:
        train_Dataloader, val_Dataloaders, test_Dataloaders, total_epochs, total_iters
    """
    train_Dataloader, val_Dataloaders, test_Dataloaders = None, [], []
    total_epochs, total_iters = None, None
    logger = get_root_logger()
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and train:
            # TODO: create train dataset
            train_Dataset = build_dataset(dataset_opt)
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_Sampler = EnlargedSampler(
                dataset=train_Dataset,
                num_replicas=opt['world_size'],
                rank=opt['rank'],
                ratio=dataset_enlarge_ratio)
            train_Dataloader = build_dataloader(
                dataset=train_Dataset,
                dataset_opt=dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_Sampler,
                seed=opt['manual_seed'])
            iters_per_epoch = math.ceil(
                (len(train_Dataset) * dataset_enlarge_ratio) / (dataset_opt["batch_size_per_gpu"] * opt['world_size']))
            total_iters = opt['train']['total_iter']
            total_epochs = math.ceil(total_iters / iters_per_epoch)  # total_epochs = total_iters / iters_per_epoch
            logger.info(
                f'Training statistics:'
                f'\n\tNumber of train samples in {dataset_opt["name"]}: {len(train_Dataset)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per GPU: {dataset_opt["batch_size_per_gpu"]}; world size(GPU number): {opt["world_size"]}'
                f'\n\tTotal iters: {total_iters}; total epochs: {total_epochs}; iters per epoch: {iters_per_epoch}')

        elif phase.split('_')[0] == 'val' and val:
            # TODO: create val dataset
            val_Dataset = build_dataset(dataset_opt)
            val_Dataloader = build_dataloader(
                dataset=val_Dataset,
                dataset_opt=dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            val_Dataloaders.append(val_Dataloader)
            logger.info(
                f'Validation statistics:'
                f'\n\tNumber of validation samples in {dataset_opt["name"]}: {len(val_Dataset)}')

        elif phase.split('_')[0] == 'test' and test:
            # TODO: create test dataset
            test_Dataset = build_dataset(dataset_opt)
            test_Dataloader = build_dataloader(
                dataset=test_Dataset,
                dataset_opt=dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            test_Dataloaders.append(test_Dataloader)
            logger.info(
                f'Testing statistics:'
                f'\n\tNumber of test samples in {dataset_opt["name"]}: {len(test_Dataset)}')

        elif phase.split('_')[0] not in ['train', 'val', 'test']:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_Dataloader, val_Dataloaders, test_Dataloaders, total_epochs, total_iters
