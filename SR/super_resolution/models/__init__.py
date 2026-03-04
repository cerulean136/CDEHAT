import importlib
import os.path as osp
from copy import deepcopy

from ..utils.basicsr_util import scandir, get_root_logger
from ..utils.registry import MODEL_REGISTRY

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(v)[0] for v in scandir(
    dir_path=model_folder, suffix='_model.py', recursive=False, full_path=False)]
root_basename = osp.basename(osp.dirname(osp.dirname(__file__)))
_model_modules = [importlib.import_module(f'{root_basename}.models.{v}') for v in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
