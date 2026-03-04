import importlib
import os.path as osp
from copy import deepcopy

from ..utils.basicsr_util import scandir, get_root_logger
from ..utils.registry import ARCH_REGISTRY

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(v)[0] for v in scandir(
    dir_path=arch_folder, suffix='_arch.py', recursive=False, full_path=False)]
root_basename = osp.basename(osp.dirname(osp.dirname(__file__)))
_arch_modules = [importlib.import_module(f'{root_basename}.archs.{v}') for v in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
