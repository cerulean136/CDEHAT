import importlib
import os.path as osp
from copy import deepcopy

from ..utils.basicsr_util import scandir, get_root_logger
from ..utils.registry import LOSS_REGISTRY

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(v)[0] for v in scandir(
    dir_path=loss_folder, suffix='_loss.py', recursive=False, full_path=False)]
root_basename = osp.basename(osp.dirname(osp.dirname(__file__)))
_loss_modules = [importlib.import_module(f'{root_basename}.losses.{v}') for v in loss_filenames]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
