import logging
import numpy as np
import argparse
import torch
from os import path as osp
import random
from torch.nn import functional as F

import sys
sys.path.append("/media/jiangaiwen/Datesets/zjt-ssh-data/HAT-all/BasicSR-master/")
sys.path.append("/media/jiangaiwen/Datesets/zjt-ssh-data/HAT-all/HAT-main/")

sys.path.append("/home/thinkstation03/zjt/HAT/BasicSR-master/")
sys.path.append("/home/thinkstation03/zjt/HAT/hat_git/")

from hatbasicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, set_random_seed
from hatbasicsr.utils.options import dict2str, _postprocess_yml_value
from hatbasicsr.utils.options import yaml_load
from hatbasicsr.utils.dist_util import get_dist_info, init_dist, master_only

import yaml
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from hat.archs.hat_arch import HAT
# import hat.archs.hat_arch

class create_hat():
    def __init__(self, root_path=None, opt_path=None):
        super(create_hat, self).__init__()
        # self.net, self.opt = self.hat_net()
        
        if root_path==None:
            root_path = "/home/thinkstation03/zjt/HAT/hat_git"
        else:
            root_path = root_path
        if opt_path==None:
            opt_path = "/home/thinkstation03/zjt/HAT/hat_git/options/testA5k/HAT-L_SRx4_ImageNet-finetune.yml"
        else:
            opt_path = opt_path
        self.opt, _ = self.parse_options(root_path, opt_path, is_train=False)
        self.net = self.model_refine(root_path, self.opt)
        print('success load hat_net, and load checkpoints!!!')

    def get_bare_model(self, net):
            """Get bare model, especially under wrapping with
            DistributedDataParallel or DataParallel.
            """
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net = net.module
            return net

    def parse_options(self, root_path, opt_path, is_train=True):
        parser = argparse.ArgumentParser()
        # parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
        parser.add_argument('--auto_resume', action='store_true')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument(
            '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
        args = parser.parse_args()

        # parse yml to dict
        opt = yaml_load(opt_path)
        # opt = yaml_load(args.opt)

        # distributed settings
        if args.launcher == 'none':
            opt['dist'] = False
            print('Disable distributed.', flush=True)
        else:
            opt['dist'] = True
            if args.launcher == 'slurm' and 'dist_params' in opt:
                init_dist(args.launcher, **opt['dist_params'])
            else:
                init_dist(args.launcher)
        opt['rank'], opt['world_size'] = get_dist_info()

        # random seed
        seed = opt.get('manual_seed')
        if seed is None:
            seed = random.randint(1, 10000)
            opt['manual_seed'] = seed
        set_random_seed(seed + opt['rank'])

        # force to update yml options
        if args.force_yml is not None:
            for entry in args.force_yml:
                # now do not support creating new keys
                keys, value = entry.split('=')
                keys, value = keys.strip(), value.strip()
                value = _postprocess_yml_value(value)
                eval_str = 'opt'
                for key in keys.split(':'):
                    eval_str += f'["{key}"]'
                eval_str += '=value'
                # using exec function
                exec(eval_str)

        opt['auto_resume'] = args.auto_resume
        opt['is_train'] = is_train

        # debug setting
        if args.debug and not opt['name'].startswith('debug'):
            opt['name'] = 'debug_' + opt['name']

        if opt['num_gpu'] == 'auto':
            opt['num_gpu'] = torch.cuda.device_count()

        # datasets
        for phase, dataset in opt['datasets'].items():
            # for multiple datasets, e.g., val_1, val_2; test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

        # paths
        for key, val in opt['path'].items():
            if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
                opt['path'][key] = osp.expanduser(val)

        if is_train:
            experiments_root = opt['path'].get('experiments_root')
            if experiments_root is None:
                experiments_root = osp.join(root_path, 'experiments')
            experiments_root = osp.join(experiments_root, opt['name'])

            opt['path']['experiments_root'] = experiments_root
            opt['path']['models'] = osp.join(experiments_root, 'models')
            opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
            opt['path']['log'] = experiments_root
            opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

            # change some options for debug mode
            if 'debug' in opt['name']:
                if 'val' in opt:
                    opt['val']['val_freq'] = 8
                opt['logger']['print_freq'] = 1
                opt['logger']['save_checkpoint_freq'] = 8
        else:  # test
            results_root = opt['path'].get('results_root')
            if results_root is None:
                results_root = osp.join(root_path, 'results')
            results_root = osp.join(results_root, opt['name'])

            opt['path']['results_root'] = results_root
            opt['path']['log'] = results_root
            opt['path']['visualization'] = osp.join(results_root, 'visualization')

        return opt, args

    def model_to_device(self, net, opt, device=None):
            """Model to device. It also warps models with DistributedDataParallel
            or DataParallel.

            Args:
                net (nn.Module)
            """
            if device!=None:
                device = device
            else:
                device = torch.device("cuda")
            net = net.to(device)
            if opt['dist']:
                find_unused_parameters = opt.get('find_unused_parameters', False)
                net = DistributedDataParallel(
                    net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            elif opt['num_gpu'] > 1:
                net = DataParallel(net)
            return net

    def load_network(self, net, load_path, strict=True, param_key='params'):
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
            print('loading ', net.__class__.__name__, 'model from', load_path, 'with param key:', param_key)
            # remove unnecessary 'module.'
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            self._print_different_keys_loading(net, load_net, strict)
            net.load_state_dict(load_net, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
            """Print keys with different name or different size when loading models.

            1. Print keys with different names.
            2. If strict=False, print the same key but with different tensor size.
                It also ignore these keys with different sizes (not load).

            Args:
                crt_net (torch model): Current network.
                load_net (dict): Loaded network.
                strict (bool): Whether strictly loaded. Default: True.
            """
            crt_net = self.get_bare_model(crt_net)
            crt_net = crt_net.state_dict()
            crt_net_keys = set(crt_net.keys())
            load_net_keys = set(load_net.keys())

            logger = get_root_logger()
            if crt_net_keys != load_net_keys:
                logger.warning('Current net - loaded net:')
                for v in sorted(list(crt_net_keys - load_net_keys)):
                    logger.warning(f'  {v}')
                logger.warning('Loaded net - current net:')
                for v in sorted(list(load_net_keys - crt_net_keys)):
                    logger.warning(f'  {v}')

            # check the size for the same keys
            if not strict:
                common_keys = crt_net_keys & load_net_keys
                for k in common_keys:
                    if crt_net[k].size() != load_net[k].size():
                        logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                    f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                        load_net[k + '.ignore'] = load_net.pop(k)

    def pre_process(self, img, opt):
            # pad to multiplication of window_size
            # 根据给定的window_size对图像进行填充，使其高度和宽度都成为window_size的倍数，
            # 以便后续网络能够处理特定大小的图像。填充的方法是使用反射填充（reflect padding）
            window_size = opt['network_g']['window_size']
            scale = opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            return img, mod_pad_h, mod_pad_w

    def post_process(self, output, mod_pad_h, mod_pad_w, scale):
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
            return output

    def model_refine(self, root_path, opt):
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

        # create model
        # model = build_model(opt)
        opt_m = deepcopy(opt['network_g'])
        model = HAT(**opt_m)

        # define network
        # self.net_g = build_network(opt['network_g'])
        net_g = self.model_to_device(model, opt)
        # self.print_network(self.net_g)

        # # load pretrained models
        load_path = opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = opt['path'].get('param_key_g', 'params')
            self.load_network(net_g, load_path, opt['path'].get('strict_load_g', True), param_key)
        print('success')
        return net_g

    def process(self, img):
        pad_img, mod_pad_h, mod_pad_w = self.pre_process(img, self.opt)
        output = self.net(pad_img)
        scale = self.opt['scale']
        output = self.post_process(output, mod_pad_h, mod_pad_w, scale)
        return output

    def hat_net(self, root_path=None, opt_path=None):
        if root_path==None:
            root_path = "/home/thinkstation03/zjt/HAT/hat_git"
        else:
            root_path = root_path
        if opt_path==None:
            opt_path = "/home/thinkstation03/zjt/HAT/hat_git/options/testA5k/HAT-L_SRx4_ImageNet-finetune.yml"
        else:
            opt_path = opt_path
        opt, _ = self.parse_options(root_path, opt_path, is_train=False)
        net = self.model_refine(root_path, opt)

        return net, opt
     

if __name__ == '__main__':
    device = torch.device("cuda")  # 选择GPU设备
    
    # root_path = "/media/jiangaiwen/Datesets/zjt-ssh-data/HAT-all/HAT-main"
    # opt_path = "options/test/HAT-L_SRx4_ImageNet-finetune.yml"
    # root_path = "/home/thinkstation03/zjt/HAT/hat_git"
    # opt_path = "/home/thinkstation03/zjt/HAT/hat_git/options/testA5k/HAT-L_SRx4_ImageNet-finetune.yml"
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    # parse options, set distributed setting, set ramdom seed
    # opt, _ = parse_options(root_path, opt_path, is_train=False)
    # net = model_refine(root_path, opt)

    hat = create_hat()
    img = torch.rand(1, 3, 30, 90)
    img = img.to(device)  # 将图像移到GPU上

    output = hat.process(img)

    print(output.size())
    print('finished')