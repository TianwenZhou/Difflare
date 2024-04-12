import cv2
import os
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.utils as vutils
import random
from PIL import Image
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.utils.img_util import tensor2img
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from pathlib import Path
from basicsr.data.flare_dataloader import generate_flare
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)

@DATASET_REGISTRY.register()
class FlareCorruptedDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            gt_path (str): Data root path for gt.
            dataroot_depth(str): Data root path for depth
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(FlareCorruptedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.crop_size = opt['crop_size'] 
        if 'image_type' not in opt:
            opt['image_type'] = 'png'
        # support multiple type of data: file path and meta data, remove support of lmdb
        self.gt_paths = []
        self.scattering_flare_paths = []
        self.reflective_flare_paths = []
        self.light_source_paths = []
        self.io_backend_opt = opt['io_backend']


       
        if 'gt_root' in opt:
            if isinstance(opt['gt_root'], str):
                self.gt_paths.extend(sorted([str(x) for x in Path(opt['gt_root']).glob('*.'+'jpg')]))
            else:
                self.gt_paths.extend(sorted([str(x) for x in Path(opt['gt_root'][0]).glob('*.'+'jpg')]))
                if len(opt['gt_root']) > 1:
                    for i in range(len(opt['gt_root'])-1):
                        self.gt_paths.extend(sorted([str(x) for x in Path(opt['gt_root'][i+1]).glob('*.'+'jpg')]))
        if 'scattering_flare_path' in opt:
            if isinstance(opt['scattering_flare_path'], str):
                self.scattering_flare_paths.extend(sorted([str(x) for x in Path(opt['scattering_flare_path']).glob('*.'+opt['image_type'])]))
            else:
                self.scattering_flare_paths.extend(sorted([str(x) for x in Path(opt['scattering_flare_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['scattering_flare_path']) > 1:
                    for i in range(len(opt['scattering_flare_path'])-1):
                        self.scattering_flare_paths.extend(sorted([str(x) for x in Path(opt['scattering_flare_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'reflective_flare_path' in opt:
            if isinstance(opt['reflective_flare_path'], str):
                self.reflective_flare_paths.extend(sorted([str(x) for x in Path(opt['reflective_flare_path']).glob('*.'+opt['image_type'])]))
            else:
                self.reflective_flare_paths.extend(sorted([str(x) for x in Path(opt['reflective_flare_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['reflective_flare_path']) > 1:
                    for i in range(len(opt['reflective_flare_path'])-1):
                        self.reflective_flare_paths.extend(sorted([str(x) for x in Path(opt['reflective_flare_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'light_source_path' in opt:
            if isinstance(opt['light_source_path'], str):
                self.light_source_paths.extend(sorted([str(x) for x in Path(opt['light_source_path']).glob('*.'+opt['image_type'])]))
            else:
                self.light_source_paths.extend(sorted([str(x) for x in Path(opt['light_source_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['light_source_path']) > 1:
                    for i in range(len(opt['light_source_path'])-1):
                        self.light_source_paths.extend(sorted([str(x) for x in Path(opt['light_source_path'][i+1]).glob('*.'+opt['image_type'])]))
        # limit number of pictures for test
       

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt and depth images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        scattering_index = random.randint(0,len(self.scattering_flare_paths) - 1)
        #print(scattering_index)
        reflective_index = random.randint(0,len(self.reflective_flare_paths) - 1)
        scattering_flare_path = self.scattering_flare_paths[scattering_index]
        reflective_flare_path = self.reflective_flare_paths[reflective_index]
        light_source_path = self.light_source_paths[scattering_index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                scattering_flare_bytes = self.file_client.get(scattering_flare_path)
                reflective_flare_bytes = self.file_client.get(reflective_flare_path)
                light_source_bytes = self.file_client.get(light_source_path)
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.gt_paths[index]
                scattering_flare_path = self.scattering_flare_paths[scattering_index]
                reflective_flare_path = self.reflective_flare_paths[reflective_index]
                light_source_path = self.light_source_paths[scattering_index]
                light_source_bytes = self.file_client.get(light_source_path)
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
       

        img_gt = imfrombytes(img_bytes, float32=True)
        scattering_flare = imfrombytes(scattering_flare_bytes, float32=True)
        reflective_flare = imfrombytes(reflective_flare_bytes, float32=True)
        light_source = imfrombytes(light_source_bytes, float32=True)


        img_with_light_source, img_lq = generate_flare(img_gt, scattering_flare, reflective_flare, light_source)

        h, w, c = img_gt.shape[0:3]
        
        crop_pad_size = self.crop_size
        # # pad
        # if h != crop_pad_size or w != crop_pad_size:
        #     img_lq = cv2.resize(img_lq, (crop_pad_size,crop_pad_size))
        #     img_with_light_source = cv2.resize(img_with_light_source, (crop_pad_size,crop_pad_size))
        # img_lq, img_with_light_source = tensor2img([img_lq, img_with_light_source], rgb2bgr=True)
        # img_lq, img_with_light_source = augment([img_lq, img_with_light_source], self.opt['use_hflip'], self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        # img_lq = torch.clamp(img_lq,min=0,max=1)
        #img_with_light_source = img2tensor([img_with_light_source], bgr2rgb=True, float32=True)[0]
        # img_with_light_source = torch.clamp(img_with_light_source,min=0,max=1)
        


        return_d = {'gt': img_with_light_source, 'lq': img_lq}
        return return_d

    def __len__(self):
        return len(self.gt_paths)
