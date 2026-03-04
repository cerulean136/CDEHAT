import os
import yaml
import torch
import random
import argparse
import os.path as osp

from .basicsr_util import ordered_yaml, _postprocess_yml_value, init_dist, get_dist_info, set_random_seed


def yaml_load(yaml_string, is_path=True):
    if is_path:
        if osp.isfile(yaml_string):
            with open(yaml_string, 'r') as f:
                return yaml.load(f, Loader=ordered_yaml()[0])
        else:
            raise FileExistsError(f"'The options file '{yaml_string}' does not exist.'")
    else:
        return yaml.load(yaml_string, Loader=ordered_yaml()[0])


def parse_options(parser, root_path, is_train=True):
    # get Command-line argument
    args = parser.parse_args()

    # load yaml as dict
    yaml_string = args.opt
    opt = yaml_load(yaml_string, is_path=True)

    # distributed setting
    if args.launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)
    opt["rank"], opt["world_size"] = get_dist_info()

    # set random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    # force to update yaml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt["auto_resume"] = args.auto_resume
    opt["is_train"] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt["datasets"].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if (dataset.get('gt_size') is not None) and ((dataset['gt_size'] % dataset['scale']) != 0):  # check gt_size
            raise ValueError(f'gt is {dataset["scale"]} times larger than lq, '
                             f'so {phase} gt_size {dataset["gt_size"]} needs to be divisible '
                             f'by the ratio {dataset["scale"]}')
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.abspath(osp.expanduser(dataset['dataroot_gt']))
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.abspath(osp.expanduser(dataset['dataroot_lq']))

    # networks
    networks = [key for key in opt.keys() if key.startswith('network_')]
    for network in networks:
        if network == 'network_g':
            if opt.get('scale') is not None:
                opt['network_g']['scale'] = opt['scale']
        if network == 'network_d':
            pass

    # paths
    for key, value in opt['path'].items():
        if ('resume_state' in key or "pretrain_network" in key) and (value is not None):
            opt['path'][key] = osp.abspath(osp.expanduser(value))

    # train or test
    if is_train is True:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 10
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 10
    else:  # test
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args
