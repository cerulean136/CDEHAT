import importlib
import os.path as osp
from copy import deepcopy

from ..utils.basicsr_util import scandir
from ..utils.registry import METRIC_REGISTRY

metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(v)[0] for v in scandir(
    dir_path=metric_folder, suffix='_metric.py', recursive=False, full_path=False)]
root_basename = osp.basename(osp.dirname(osp.dirname(__file__)))
_metric_modules = [importlib.import_module(f'{root_basename}.metrics.{v}') for v in metric_filenames]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
