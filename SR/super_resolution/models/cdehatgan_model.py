import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import random
from collections import OrderedDict
from torch.nn import functional as F
from copy import deepcopy

from ..utils.basicsr_util import USMSharp, DiffJPEG, get_root_logger, filter2D, paired_random_crop
from ..utils.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from ..utils.data_util import DataUtil
from ..utils.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from ..archs import build_network
from ..losses import build_loss
from ..metrics import calculate_metric
from ..utils.registry import MODEL_REGISTRY
from .srgan_model import SRModel, SRGANModel


@MODEL_REGISTRY.register()
class CDEHATGANModel(SRModel):

    def __init__(self, opt):
        super(SRModel, self).__init__(opt=opt)
        self.usm_sharper = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', None)

        self.encoder_iter = self.opt['network_g']['encoder_iter']
        self.is_diffusion = False
        self.is_freeze = False

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key,
                              self.opt['path'].get('module_name'))

        if self.is_train:
            self.init_training_settings()

    def load_network(self, net, load_path, strict=True, param_key='params', module_name=None):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        filter_state_dict = {}
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        if module_name is not None:
            for k, v in deepcopy(load_net).items():
                if k.startswith(module_name):
                    filter_state_dict[k] = v
        if module_name is None:
            self._print_different_keys_loading(net, load_net, strict)
            net.load_state_dict(load_net, strict=strict)
        else:
            strict = False
            net.load_state_dict(filter_state_dict, strict=strict)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema',
                                  self.opt['path'].get('module_name'))
            else:
                self.model_ema(0)  # copy net_g weight to net_g_ema
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)
        # set model state
        self.net_g.train()
        self.net_d.train()

        # define losses
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = (
            build_loss(train_opt['perceptual_opt']).to(self.device)) if train_opt.get('perceptual_opt') else None
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device) if train_opt.get('gan_opt') else None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set net_d iter strategy
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        if self.opt.pop('current_iter') <= self.encoder_iter:
            self.is_diffusion = False
            self.setup_optimizers()
        else:
            self.is_diffusion = True
            if not self.is_freeze:
                self.freeze_module(self.net_g.encoder_gt)
                self.setup_optimizers()
                self.is_freeze = True
        self.setup_schedulers()

    def setup_optimizers(self):
        """Set up optimizers."""
        self.optimizers = []
        train_opt = deepcopy(self.opt['train'])
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def setup_schedulers(self):
        """Set up schedulers."""
        self.schedulers = []
        train_opt = deepcopy(self.opt['train'])
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def optimize_parameters(self, current_iter):
        if current_iter <= self.encoder_iter:
            self.is_diffusion = False
        else:
            self.is_diffusion = True
            if not self.is_freeze:
                self.freeze_module(self.net_g.encoder_gt)
                self.setup_optimizers()
                self.is_freeze = True
                self.setup_schedulers()

        # on/off usm sharpen
        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] is True else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] is True else self.gt
        gan_gt = self.gt_usm if self.opt['gan_gt_usm'] is True else self.gt

        # TODO: optimize net_g
        for net_d_parameter in self.net_d.parameters():
            net_d_parameter.requires_grad = False
        self.optimizer_g.zero_grad()
        self.output, cdp, cdp_diff = self.net_g(self.lq, self.gt, is_diffusion=self.is_diffusion)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0) and (current_iter > self.net_d_init_iters):  # net_d iter strategy
            # for generator
            # pix loss
            if self.cri_pix is not None:
                l_g_pix = self.cri_pix(pred=self.output, target=l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual is not None:
                l_g_perceptual, l_g_style = self.cri_perceptual(x=self.output, gt=percep_gt)
                if l_g_perceptual is not None:
                    l_g_total += l_g_perceptual
                    loss_dict['l_g_perceptual'] = l_g_perceptual
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (measures how well the discriminator recognizes generated fake data as real data)
            if self.cri_gan is not None:
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(input=fake_g_pred, target_is_real=True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

            # TODO: cdp loss
            if self.is_diffusion is True:
                # pixel loss
                if self.cri_pix is not None:
                    l_pix_cdp = 0.1 * self.cri_pix(cdp, cdp_diff)
                    l_g_total += l_pix_cdp
                    loss_dict['l_pix_cdp'] = l_pix_cdp

            l_g_total.backward()  # backward
            self.optimizer_g.step()  # update weights and bias

        # TODO: optimize net_d
        for net_d_parameter in self.net_d.parameters():
            net_d_parameter.requires_grad = True
        self.optimizer_d.zero_grad()

        # for discriminator
        # real (the loss of real data being correctly recognized as true)
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(input=real_d_pred, target_is_real=True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        # fake (the loss of generated fake data being correctly recognized as fake)
        fake_d_pred = self.net_d(self.output.detach().clone())  # avoid affecting the gradient of net_g
        l_d_fake = self.cri_gan(input=fake_d_pred, target_is_real=False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake

        # discriminator prediction result (calculate mean)
        loss_dict['out_d_real'] = torch.mean(input=real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(input=fake_d_pred.detach())

        l_d_real.backward(), l_d_fake.backward()  # backward
        self.optimizer_d.step()  # update weights and bias

        # implement exponential moving average of model parameters for net_g_ema
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # reduce the loss values on different GPUs and then calculate the average
        self.log_dict = self.reduce_loss_dict(loss_dict=loss_dict)

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train is True:
            self.lq = data['lq'].to(self.device)
            self.gt = data['gt'].to(self.device)
            if (self.queue_size is not None) and (self.queue_size > 0):
                self._dequeue_and_enqueue()  # update gt and lq from training pair pool
            self.gt_usm = self.usm_sharper(self.gt)

        else:  # is_train is False
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharper(self.gt)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # initialize training pair pool for increasing the diversity in a batch. (only used when is_train is true)
        b, c_lq, h_lq, w_lq = self.lq.size()
        b, c_gt, h_gt, w_gt = self.gt.size()
        if not hasattr(self, 'queue_lq') or not hasattr(self, 'queue_gt'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lq = torch.zeros(size=(self.queue_size, c_lq, h_lq, w_lq)).cuda()
            self.queue_gt = torch.zeros(size=(self.queue_size, c_gt, h_gt, w_gt)).cuda()
            self.queue_ptr = 0
        # enqueue
        if self.queue_ptr < self.queue_size:
            self.queue_lq[self.queue_ptr:self.queue_ptr + b, ...] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, ...] = self.gt.clone()
            self.queue_ptr += b
            print(f'number of elements inside the training pair pool / total size of the training pair pool: '
                  f'{self.queue_ptr} / {self.queue_size}')
        # dequeue and enqueue
        elif self.queue_ptr == self.queue_size:
            indices = torch.randperm(self.queue_size)
            self.queue_lq, self.queue_gt = self.queue_lq[indices], self.queue_gt[indices]  # shuffle
            dequeue_lq = self.queue_lq[0:b, ...].clone()
            dequeue_gt = self.queue_gt[0:b, ...].clone()  # dequeue
            self.queue_lq[0:b, ...] = self.lq.clone()
            self.queue_gt[0:b, ...] = self.gt.clone()  # enqueue
            self.lq, self.gt = dequeue_lq, dequeue_gt  # get lq and gt tensor

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        torch.cuda.empty_cache()

        if isinstance(current_iter, int) and current_iter <= self.encoder_iter:
            self.is_diffusion = False
        else:
            print("Stage 2 validation!")
            self.is_diffusion = True

        # training flag is turned off during validation
        self.is_train = False

        # start validation
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None and \
                       dataloader.dataset.opt.get('dataroot_gt') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if 'lq_path' in val_data:
                lq_path = val_data['lq_path'][0]
                lq_path = lq_path[0] if not isinstance(lq_path, str) else lq_path
                img_name = osp.splitext(osp.basename(lq_path))[0]
            else:
                gt_path = val_data['gt_path'][0]
                gt_path = gt_path[0] if not isinstance(gt_path, str) else gt_path
                img_name = osp.splitext(osp.basename(gt_path))[0]

            # feed data
            self.feed_data(val_data)

            # start test
            if self.opt.get('tile') is not None:
                self.tile_test()  # crop large-size images into tiles to save GPU memory
            else:
                self.test()  # testing large-size images may require more GPU memory

            visuals = self.get_current_visuals()
            sr_img = DataUtil.tensor_to_image(
                tensor=visuals['result'], out_type=np.uint8, rgb_to_bgr=False, min_max=(0, 1))
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = DataUtil.tensor_to_image(
                    tensor=visuals['gt'], out_type=np.uint8, rgb_to_bgr=False, min_max=(0, 1))
                metric_data['img2'] = gt_img
                del self.gt


            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val'].get('suffix') is not None:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                # save tif
                if self.opt['val'].get('tif') is True:
                    save_img_path = save_img_path.replace('.png', '.tif')
                DataUtil.image_write_by_rasterio(input_image=sr_img, output_image=save_img_path, channel_axis=2)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # training flag is turned on at the end of validation
        self.is_train = True

        torch.cuda.empty_cache()

    def test(self):
        # pad the lq image size to an integer multiple of the window_size for the test
        is_lqpad = False
        old_h_lq, old_w_lq = None, None  # original lq height and width size
        scale = self.opt['scale']
        if self.opt['network_g'].get('window_size') is not None:
            b, c, old_h_lq, old_w_lq = self.lq.size()
            window_size = self.opt['network_g']['window_size']
            lqpad_size_in_h = window_size - old_h_lq % window_size if old_h_lq % window_size != 0 else 0
            lqpad_size_in_w = window_size - old_w_lq % window_size if old_w_lq % window_size != 0 else 0
            is_lqpad = True if lqpad_size_in_h != 0 or lqpad_size_in_w != 0 else False
            if is_lqpad:
                self.lq = F.pad(input=self.lq, pad=(0, lqpad_size_in_w, 0, lqpad_size_in_h), mode='reflect')
                if self.gt is not None:
                    self.gt = F.pad(input=self.gt, pad=(0, lqpad_size_in_w * scale, 0, lqpad_size_in_h * scale),
                                    mode='reflect')

        # start test
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.gt, is_diffusion=self.is_diffusion)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.gt, is_diffusion=self.is_diffusion)
            self.net_g.train()

        # crop lq and output to restore original size
        if is_lqpad is True:
            self.lq = self.lq[:, :, 0:old_h_lq, 0:old_w_lq]
            self.output = self.output[:, :, 0:old_h_lq * scale, 0:old_w_lq * scale]
            if self.gt is not None:
                self.gt = self.gt[:, :, 0:old_h_lq * scale, 0:old_w_lq * scale]

        torch.cuda.empty_cache()

    def tile_test(self):
        print('use tile test:')
        # pad the lq image size to an integer multiple of the window_size for the test
        is_lqpad = False
        old_h_lq, old_w_lq = None, None  # original lq height and width size
        scale = self.opt['scale']
        if self.opt['network_g'].get('window_size') is not None:
            b, c, old_h_lq, old_w_lq = self.lq.size()
            window_size = self.opt['network_g']['window_size']
            lqpad_size_in_h = window_size - old_h_lq % window_size if old_h_lq % window_size != 0 else 0
            lqpad_size_in_w = window_size - old_w_lq % window_size if old_w_lq % window_size != 0 else 0
            is_lqpad = True if lqpad_size_in_h != 0 or lqpad_size_in_w != 0 else False
            if is_lqpad:
                self.lq = F.pad(input=self.lq, pad=(0, lqpad_size_in_w, 0, lqpad_size_in_h), mode='reflect')
                if self.gt is not None:
                    self.gt = F.pad(input=self.gt, pad=(0, lqpad_size_in_w * scale, 0, lqpad_size_in_h * scale),
                                    mode='reflect')

        # start tile test
        b, c, h_lq, w_lq = self.lq.size()  # batch size, channel size, paded lq height, paded lq width
        b, c, h_output, w_output = b, c, h_lq * scale, w_lq * scale
        self.output = self.lq.new_zeros(size=(b, c, h_output, w_output))  # initialize output
        tile_size, tile_pad = self.opt['tile']['tile_size'], self.opt['tile']['tile_pad']
        range_list_h_lq = list(range(0, h_lq - tile_size, tile_size))
        range_list_w_lq = list(range(0, w_lq - tile_size, tile_size))
        if len(range_list_h_lq) != 0:
            range_list_h_lq.append(h_lq - tile_size)
        else:
            range_list_h_lq.append(0)
        if len(range_list_w_lq) != 0:
            range_list_w_lq.append(w_lq - tile_size)
        else:
            range_list_w_lq.append(0)
        count = 0
        for h in range_list_h_lq:
            for w in range_list_w_lq:
                count += 1
                print(f'deal with the tile {count} / {len(range_list_h_lq) * len(range_list_w_lq)}')

                # lq with tile padding
                tilepad_h_start_lq, tilepad_h_end_lq = max(h - tile_pad, 0), min(h + tile_size + tile_pad, h_lq)
                tilepad_w_start_lq, tilepad_w_end_lq = max(w - tile_pad, 0), min(w + tile_size + tile_pad, w_lq)

                # get input tile
                tile_input = self.lq[:, :, tilepad_h_start_lq:tilepad_h_end_lq, tilepad_w_start_lq:tilepad_w_end_lq]
                h_tile_input, w_tile_input = tile_input.shape[-2], tile_input.shape[-1]
                if self.gt is not None:
                    tile_input_gt = self.gt[:, :, tilepad_h_start_lq * scale:tilepad_h_end_lq * scale,
                    tilepad_w_start_lq * scale:tilepad_w_end_lq * scale]
                else:
                    tile_input_gt = None

                # get output tile
                if hasattr(self, 'net_g_ema'):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        tile_output = self.net_g_ema(tile_input, tile_input_gt, is_diffusion=self.is_diffusion)
                else:
                    self.net_g.eval()
                    with torch.no_grad():
                        tile_output = self.net_g(tile_input, tile_input_gt, is_diffusion=self.is_diffusion)
                    self.net_g.train()

                # lq without tile padding
                h_start_lq, h_end_lq = h, min(h + tile_size, h_lq)
                w_start_lq, w_end_lq = w, min(w + tile_size, w_lq)

                # TODO: debug
                if h == range_list_h_lq[-1]:
                    assert h + min(tile_size, h_lq) == h_lq, 'error'
                if w == range_list_w_lq[-1]:
                    assert w + min(tile_size, w_lq) == w_lq, 'error'

                # compute the relative position is between `lq with padding` and `lq without padding`
                h_start_output, h_end_output = \
                    (h_start_lq - tilepad_h_start_lq) * scale, (h_tile_input + h_end_lq - tilepad_h_end_lq) * scale
                w_start_output, w_end_output = \
                    (w_start_lq - tilepad_w_start_lq) * scale, (w_tile_input + w_end_lq - tilepad_w_end_lq) * scale

                # TODO: debug
                assert h_start_output + min(tile_size, h_lq) * scale == h_end_output, 'error'
                assert w_start_output + min(tile_size, w_lq) * scale == w_end_output, 'error'

                # result write to output
                self.output[:, :, h_start_lq * scale:h_end_lq * scale, w_start_lq * scale:w_end_lq * scale] = \
                    tile_output[:, :, h_start_output:h_end_output, w_start_output:w_end_output]

        print(f'\tdone')

        # crop lq and output to restore original size
        if is_lqpad is True:
            self.lq = self.lq[:, :, 0:old_h_lq, 0:old_w_lq]
            self.output = self.output[:, :, 0:old_h_lq * scale, 0:old_w_lq * scale]
            if self.gt is not None:
                self.gt = self.gt[:, :, 0:old_h_lq * scale, 0:old_w_lq * scale]

        torch.cuda.empty_cache()
