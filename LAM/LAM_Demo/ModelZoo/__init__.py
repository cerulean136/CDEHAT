import os
import torch

from SR.super_resolution.utils.options_util import yaml_load
from SR.super_resolution.archs import edsr_arch, rcan_arch, rrdbnet_arch, hat_arch, swinir_arch, ttst_arch, temp_arch

MODEL_DIR = './LAM_Demo/ModelZoo/models'

NN_LIST = [
    'EDSR',
    'RCAN',
    'RRDBNet',
    'SwinIR',
    'HAT',
    'TTST',
    'CDEHAT',
]  # todo

MODEL_LIST = {
    'EDSR': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'RCAN': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'RRDBNet': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'SwinIR': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'HAT': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'TTST': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth",
    },

    'CDEHAT': {
        'Base': r"C:\python_projects\SR\weights\net_g_latest.pth"
    }  # todo
}


def get_model(model_name):
    print(f'Getting SR Network `{model_name}`')
    if model_name.split('-')[0] in NN_LIST:
        if model_name == 'EDSR':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_EDSR_SRx4_on_AID.yml',
                            is_path=True)
            net = edsr_arch.EDSR(**opt['network_g'])

        elif model_name == 'RCAN':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_RCAN_SRx4_on_AID.yml',
                            is_path=True)
            net = rcan_arch.RCAN(**opt['network_g'])

        elif model_name == 'RRDBNet':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_ESRGAN_SRx4_on_AID.yml',
                            is_path=True)
            net = rrdbnet_arch.RRDBNet(**opt['network_g'])


        elif model_name == 'SwinIR':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_SwinIR_SRx4_on_AID.yml',
                            is_path=True)
            net = swinir_arch.SwinIR(**opt['network_g'])

        elif model_name == 'HAT':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_HAT_SRx4_on_AID.yml',
                            is_path=True)
            net = hat_arch.HAT(**opt['network_g'])

        elif model_name == 'TTST':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_TTST_SRx4_on_AID.yml',
                            is_path=True)
            net = ttst_arch.TTST(**opt['network_g'])

        elif model_name == 'CDEHAT':
            opt = yaml_load(r'C:\python_projects\SR\super_resolution\options\train\train_CDEHAT_SRx4_on_AID.yml',
                            is_path=True)
            net = temp_arch.Temp(**opt['network_g'])
        # todo
        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'

    net = get_model(model_name)

    state_dict_path = os.path.normpath(os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name]))
    if not os.path.exists(state_dict_path):
        state_dict_path = MODEL_LIST[model_name][training_name]

    print(f'Loading model `{state_dict_path}` for `{model_name}` network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    if 'params_ema' in state_dict:
        net.load_state_dict(state_dict['params'])
    elif 'params' in state_dict:
        net.load_state_dict(state_dict['params_ema'])
    else:
        net.load_state_dict(state_dict)

    return net


def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f'Network: {model_name}, with parameters: {num_params:,d}')
