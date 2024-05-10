# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize, resize
import torch

import sys
sys.path.append("/home/thinkstation03/zjt/NAFNet-ALL/NAFNet-main/")
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_hw, withhat_paired_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import os
import numpy as np
import h5py


sys.path.append("/media/jiangaiwen/Datesets/zjt-ssh-data/HAT-all/BasicSR-master/")
sys.path.append("/media/jiangaiwen/Datesets/zjt-ssh-data/HAT-all/HAT-main/")
sys.path.append("/home/thinkstation03/zjt/HAT/BasicSR-master/")
sys.path.append("/home/thinkstation03/zjt/HAT/hat_git/")
# from hat.createmodel_hat import create_hat


class PairedImageSRLRDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageSRLRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            import os
            nums_lq = len(os.listdir(self.lq_folder))
            nums_gt = len(os.listdir(self.gt_folder))

            # nums_lq = sorted(nums_lq)
            # nums_gt = sorted(nums_gt)

            # print('lq gt ... opt')
            # print(nums_lq, nums_gt, opt)
            assert nums_gt == nums_lq

            self.nums = nums_lq
            # {:04}_L   {:04}_R


            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']

        gt_path_L = os.path.join(self.gt_folder, '{:04}_L.png'.format(index + 1))
        gt_path_R = os.path.join(self.gt_folder, '{:04}_R.png'.format(index + 1))


        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))


        lq_path_L = os.path.join(self.lq_folder, '{:04}_L.png'.format(index + 1))
        lq_path_R = os.path.join(self.lq_folder, '{:04}_R.png'.format(index + 1))

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))



        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path_L)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # if scale != 1:
        #     c, h, w = img_lq.shape
        #     img_lq = resize(img_lq, [h*scale, w*scale])
            # print('img_lq .. ', img_lq.shape, img_gt.shape)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f'{index+1:04}',
            'gt_path': f'{index+1:04}',
        }

    def __len__(self):
        return self.nums // 2


class PairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)

    def crop_image(self, image, scale):
        H, W, C = image.shape

        new_H = (H // scale) * scale
        new_W = (W // scale) * scale

        cropped_image = image[:new_H, :new_W, :]

        return cropped_image

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')

        assert self.lq_files[index]==self.gt_files[index], 'Error, lr and hr not match'

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))


        scale = self.opt['scale']
            

        
        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:  # RGB 通道变换
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()

            if img_gt.shape == img_lq.shape:
                 real_scale = 1
            else: 
                real_scale = scale
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
            #                                     'gt_path_L_and_R')
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, real_scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)
    
            img_gt, img_lq = imgs
        else:
            print(end='')
            pass

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


class abaPairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    修改原生的代码, 读取真的lr
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)

    def crop_image(self, image, scale):
        H, W, C = image.shape

        new_H = (H // scale) * scale
        new_W = (W // scale) * scale

        cropped_image = image[:new_H, :new_W, :]

        return cropped_image

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'reallr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'reallr1.png')

        assert self.lq_files[index]==self.gt_files[index], 'Error, lr and hr not match'

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))


        scale = self.opt['scale']
            

        
        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:  # RGB 通道变换
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()

            if img_gt.shape == img_lq.shape:
                 real_scale = 1
            else: 
                real_scale = scale
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
            #                                     'gt_path_L_and_R')
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, real_scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)
    
            img_gt, img_lq = imgs
        else:
            print(end='')
            pass

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


class hatPassPairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)

    '''
    def __init__(self, opt):
        super(hatPassPairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if opt['dataroot_reallq'] is not None:
            # self.device = torch.device("cuda")  # 选择GPU设备
            self.reallq_folder = opt['dataroot_reallq']
            # self.hat_net = create_hat(device=self.device)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)
        self.reallq_files = os.listdir(self.reallq_folder)

        self.nums = len(self.gt_files)

    def crop_image(self, image, scale):
        H, W, C = image.shape

        new_H = (H // scale) * scale
        new_W = (W // scale) * scale

        cropped_image = image[:new_H, :new_W, :]

        return cropped_image

    def get_imgarry(self, path_root, img_class):
        img_bytes = self.file_client.get(path_root, img_class)
        try:
            img_array = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(path_root))
        return img_array

    def get_imgarry_form_featH5(self, path):
        # 加载
        res = 0
        with h5py.File(path, 'r') as hf:
            loaded_data = hf['data'][()]
            loaded_tensor = torch.from_numpy(loaded_data)
            _, c, h, w = loaded_tensor.shape

            # 转换为NumPy数组
            loaded_array = loaded_tensor.numpy()

            loaded_array = loaded_array.squeeze(0).transpose((1, 2, 0))
            #  将(1, c, h, w)转为( h, w, c)

            if c==1:
                loaded_array = np.tile(loaded_array, (1, 1, 3))
                # 将(h, w, c)转为(h, w, 3*c)
        
            res = loaded_array
        return res

    def define_cat(self, img, feat):
        imgs = img.chunk(2, dim=0)
        # imgs_a = [x for x in imgs]

        feats = feat.chunk(2, dim=0)
        # feats_a = [x for x in feats]
        out_L = torch.cat((imgs[0], feats[0]), dim=0)
        out_R = torch.cat((imgs[1], feats[1]), dim=0)
        out = torch.cat((out_L, out_R), dim=0)
        # torch.cat

        # now = out.chunk(2, dim=1)

        # img_L = out_L[:, :3, :, :]
        # feat_L = out_L[:, 3:, :, :]

        # img_R = out_R[:, :3, :, :]
        # feat_R = out_R[:, 3:, :, :]
        return out

    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # 图片路径
        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')


        if self.opt['phase'] == 'train':
            lq_path_real_L = os.path.join(self.reallq_folder, self.reallq_files[index], 'real3lr0.h5')
            lq_path_real_R = os.path.join(self.reallq_folder, self.reallq_files[index], 'real3lr1.h5')
        else:
            lq_path_real_L = os.path.join(self.reallq_folder, self.reallq_files[index], '3lr0.h5')
            lq_path_real_R = os.path.join(self.reallq_folder, self.reallq_files[index], '3lr1.h5')



        assert self.lq_files[index]==self.gt_files[index], 'Error, lr and hr not match'

        # 加载图片
        img_gt_L = self.get_imgarry(gt_path_L, 'gt')  # img array
        img_gt_R = self.get_imgarry(gt_path_R, 'gt')

        img_lq_L = self.get_imgarry(lq_path_L, 'lq')
        img_lq_R = self.get_imgarry(lq_path_R, 'lq')

        # img_lq_realL = self.get_imgarry(lq_path_real_L, 'lq')
        # img_lq_realR = self.get_imgarry(lq_path_real_R, 'lq')
        img_lq_realL = self.get_imgarry_form_featH5(lq_path_real_L)
        img_lq_realR = self.get_imgarry_form_featH5(lq_path_real_R)

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)
        img_reallq = np.concatenate([img_lq_realL, img_lq_realR], axis=-1)



        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:  # RGB 通道变换
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]
                # img_reallq = img_reallq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_reallq = img_reallq.copy()

            if img_gt.shape == img_lq.shape:
                 real_scale = 1
            else: 
                real_scale = scale
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
            #                                     'gt_path_L_and_R')
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, real_scale,
            #                                     'gt_path_L_and_R')   # 随机裁剪
            img_gt, img_lq, img_reallq = withhat_paired_random_crop_hw(img_gt, img_lq, img_reallq, gt_size_h, gt_size_w, real_scale,
                                                'gt_path_L_and_R')   # 随机裁剪

            # flip, rotation
            # imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
            #                         self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)
            tmpreal_imgs, status = augment([img_gt,img_lq, img_reallq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)  # 翻转等

    
            # img_gt, img_lq = imgs
            img_gt, img_lq, img_reallq = tmpreal_imgs
        else:
            print(end='')
            pass

        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        img_gt, img_lq, img_reallq = img2tensor([img_gt, img_lq, img_reallq],
                                    bgr2rgb=True,
                                    float32=True)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_reallq, self.mean, self.std, inplace=True)


        # img_reallq_with_batch = img_reallq.unsqueeze(0).to(self.device)  # 在第0维度（最前面）添加一个维度
        # output_L, feat_L = self.hat_net.process(img_reallq_with_batch[:,:3,:,:])
        # output_R, feat_R = self.hat_net.process(img_reallq_with_batch[:,3:,:,:])

        # img_reallq_with_batch.to('cpu')

        if 1==0: # 单层特征图
            res_feat = torch.concat((img_reallq[0, :, :].unsqueeze(0), img_reallq[3, :, :].unsqueeze(0)), dim=0)
        else:  # 多层特征图
            res_feat = img_reallq
            # res_feat = torch.concat((img_reallq[:, :, :], img_reallq[:, :, :]), dim=0)

        img_lq = self.define_cat(img_lq, res_feat)

        return {
            'lq': img_lq,
            'gt': img_gt,
            # 'reallq': res_feat,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


class LeftTwice_hatPassPairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)

    '''
    def __init__(self, opt):
        super(LeftTwice_hatPassPairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if opt['dataroot_reallq'] is not None:
            # self.device = torch.device("cuda")  # 选择GPU设备
            self.reallq_folder = opt['dataroot_reallq']
            # self.hat_net = create_hat(device=self.device)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)
        self.reallq_files = os.listdir(self.reallq_folder)

        self.nums = len(self.gt_files)

        self.Left_twice = True

    def crop_image(self, image, scale):
        H, W, C = image.shape

        new_H = (H // scale) * scale
        new_W = (W // scale) * scale

        cropped_image = image[:new_H, :new_W, :]

        return cropped_image

    def get_imgarry(self, path_root, img_class):
        img_bytes = self.file_client.get(path_root, img_class)
        try:
            img_array = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(path_root))
        return img_array

    def get_imgarry_form_featH5(self, path):
        # 加载
        res = 0
        with h5py.File(path, 'r') as hf:
            loaded_data = hf['data'][()]
            loaded_tensor = torch.from_numpy(loaded_data)
            _, c, h, w = loaded_tensor.shape

            # 转换为NumPy数组
            loaded_array = loaded_tensor.numpy()

            loaded_array = loaded_array.squeeze(0).transpose((1, 2, 0))
            #  将(1, c, h, w)转为( h, w, c)

            if c==1:
                loaded_array = np.tile(loaded_array, (1, 1, 3))
                # 将(h, w, c)转为(h, w, 3*c)
        
            res = loaded_array
        return res

    def define_cat(self, img, feat):
        imgs = img.chunk(2, dim=0)
        # imgs_a = [x for x in imgs]

        feats = feat.chunk(2, dim=0)
        # feats_a = [x for x in feats]
        out_L = torch.cat((imgs[0], feats[0]), dim=0)
        out_R = torch.cat((imgs[1], feats[1]), dim=0)
        out = torch.cat((out_L, out_R), dim=0)
        # torch.cat

        # now = out.chunk(2, dim=1)

        # img_L = out_L[:, :3, :, :]
        # feat_L = out_L[:, 3:, :, :]

        # img_R = out_R[:, :3, :, :]
        # feat_R = out_R[:, 3:, :, :]
        return out

    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # 图片路径
        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        if self.Left_twice == False:
            gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')
        else:
            gt_path_R = gt_path_L

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        if self.Left_twice == False:
            lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')
        else:
            lq_path_R = lq_path_L

        if self.opt['phase'] == 'train':
            lq_path_real_L = os.path.join(self.reallq_folder, self.reallq_files[index], 'real3lr0.h5')
            lq_path_real_R = os.path.join(self.reallq_folder, self.reallq_files[index], 'real3lr1.h5')
        else:
            lq_path_real_L = os.path.join(self.reallq_folder, self.reallq_files[index], '3lr0.h5')
            if self.Left_twice == False:
                lq_path_real_R = os.path.join(self.reallq_folder, self.reallq_files[index], '3lr1.h5')
            else:
                lq_path_real_R = lq_path_real_L



        assert self.lq_files[index]==self.gt_files[index], 'Error, lr and hr not match'

        # 加载图片
        img_gt_L = self.get_imgarry(gt_path_L, 'gt')  # img array
        img_gt_R = self.get_imgarry(gt_path_R, 'gt')

        img_lq_L = self.get_imgarry(lq_path_L, 'lq')
        img_lq_R = self.get_imgarry(lq_path_R, 'lq')

        # img_lq_realL = self.get_imgarry(lq_path_real_L, 'lq')
        # img_lq_realR = self.get_imgarry(lq_path_real_R, 'lq')
        img_lq_realL = self.get_imgarry_form_featH5(lq_path_real_L)
        img_lq_realR = self.get_imgarry_form_featH5(lq_path_real_R)

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)
        img_reallq = np.concatenate([img_lq_realL, img_lq_realR], axis=-1)



        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:  # RGB 通道变换
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]
                # img_reallq = img_reallq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_reallq = img_reallq.copy()

            if img_gt.shape == img_lq.shape:
                 real_scale = 1
            else: 
                real_scale = scale
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
            #                                     'gt_path_L_and_R')
            # img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, real_scale,
            #                                     'gt_path_L_and_R')   # 随机裁剪
            img_gt, img_lq, img_reallq = withhat_paired_random_crop_hw(img_gt, img_lq, img_reallq, gt_size_h, gt_size_w, real_scale,
                                                'gt_path_L_and_R')   # 随机裁剪

            # flip, rotation
            # imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
            #                         self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)
            tmpreal_imgs, status = augment([img_gt,img_lq, img_reallq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)  # 翻转等

    
            # img_gt, img_lq = imgs
            img_gt, img_lq, img_reallq = tmpreal_imgs
        else:
            print(end='')
            pass

        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        img_gt, img_lq, img_reallq = img2tensor([img_gt, img_lq, img_reallq],
                                    bgr2rgb=True,
                                    float32=True)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_reallq, self.mean, self.std, inplace=True)


        # img_reallq_with_batch = img_reallq.unsqueeze(0).to(self.device)  # 在第0维度（最前面）添加一个维度
        # output_L, feat_L = self.hat_net.process(img_reallq_with_batch[:,:3,:,:])
        # output_R, feat_R = self.hat_net.process(img_reallq_with_batch[:,3:,:,:])

        # img_reallq_with_batch.to('cpu')

        if 1==0: # 单层特征图
            res_feat = torch.concat((img_reallq[0, :, :].unsqueeze(0), img_reallq[3, :, :].unsqueeze(0)), dim=0)
        else:  # 多层特征图
            res_feat = img_reallq
            # res_feat = torch.concat((img_reallq[:, :, :], img_reallq[:, :, :]), dim=0)

        img_lq = self.define_cat(img_lq, res_feat)

        return {
            'lq': img_lq,
            'gt': img_gt,
            # 'reallq': res_feat,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}',)
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# # 示例用法
# total_iterations = 100
# for i in range(total_iterations + 1):
#     time.sleep(0.1)  # 模拟某个任务
    # print_progress_bar(i, total_iterations, prefix='Progress:', suffix='Complete', length=40)


# from basicsr.utils.options import dict2str, parse

def check_h5Open(): # # 检查训练集的h5文件是否能正常读取
    def get_imgarry_form_featH5(path): # 读取h5文件
        # 加载
        res = 0
        with h5py.File(path, 'r') as hf:
            loaded_data = hf['data'][()]
            loaded_tensor = torch.from_numpy(loaded_data)
            _, c, h, w = loaded_tensor.shape

            # 转换为NumPy数组
            loaded_array = loaded_tensor.numpy()

            loaded_array = loaded_array.squeeze(0).transpose((1, 2, 0))
            #  将(1, c, h, w)转为( h, w, c)

            if c==1:
                loaded_array = np.tile(loaded_array, (1, 1, 3))
                # 将(h, w, c)转为(h, w, 3*c)
        
            res = loaded_array
        return res
    if 1==1:  
        err_list = []
        print('process start')
        train_root = "/home/thinkstation03/zjt/NAFNet-data/train/v2finetuneHAT-L_sisr_withreallr_patches_x2"
        folders = os.listdir(train_root)
        print('len(folders)', len(folders))
        for i,folder in enumerate(folders):
            print_progress_bar(i, len(folders), prefix='Progress:', suffix='Complete', length=100)
            try:
                fileh5L = os.path.join(train_root, folder, 'real3lr0.h5')
                res_L = get_imgarry_form_featH5(fileh5L)
            except:
                print('error:', fileh5L)
                err_list.append(fileh5L)

            try:
                fileh5R = os.path.join(train_root, folder, 'real3lr1.h5')
                res_R = get_imgarry_form_featH5(fileh5R)
            except:
                print('error:', fileh5R)
                err_list.append(fileh5R)
    print('end!!!')
    print('\n\n')
    print(err_list)

if __name__=="__main__":
    device = torch.device("cuda")  # 选择GPU设备
    print('start!!!')
    if 1==0:
        from hat.createmodel_hat import create_hat
        hat = create_hat()


        # pairDataset = PairedStereoImageDataset(opt_dateset)
        img = torch.rand(1, 3, 30, 90)
        img = img.to(device)  # 将图像移到GPU上
        output, feat = hat.process(img)
        print("img-size:",output.size())
        print("feat-size:",feat.size())
        print('here')
    
    
    
