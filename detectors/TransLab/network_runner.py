from __future__ import print_function
import os
import sys

from pathlib import Path

from common.network_runner_base import NetworkRunnerBase
from segmentron.utils import options
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from IPython import embed
from collections import OrderedDict
from segmentron.utils.filesystem import makedirs
import cv2
import numpy as np


class NetworkRunner(NetworkRunnerBase):
    def __init__(
            self,
            input_dir: Path,
            output_dir: Path,
            log_path: Path,
            model_path: Path,
            segmentron_args
    ):
        super().__init__(input_dir, output_dir, log_path, model_path)

        root_path = os.path.abspath(os.path.dirname(__file__))
        sys.path.append(root_path)

        cfg.TEST.TEST_MODEL_PATH = str(model_path)
        cfg.DEMO_DIR = str(input_dir)
        cfg.update_from_file('configs/trans10K/translab.yaml')
        cfg.PHASE = 'test'
        cfg.ROOT_PATH = root_path
        cfg.DATASET.NAME = 'trans10k_extra'
        cfg.check_and_freeze()

        sys.argv = [__file__] + segmentron_args
        args = options.parse_args()
        default_setup(args)

        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                               root=cfg.DEMO_DIR,
                                               split='val',
                                               mode='val',
                                               transform=input_transform,
                                               base_size=cfg.TRAIN.BASE_SIZE)

        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model(self.device).to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)
        self.count_easy = 0
        self.count_hard = 0

        self.model.eval()

        if self.args.distributed:
            self.model = self.model.module

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def run(self):
        for (image, _, filename) in tqdm(self.val_loader):
            image = image.to(self.device)
            filename = filename[0]
            save_name = os.path.basename(filename).replace('.jpg', '').replace('.png', '')

            ori_img, size = self._read_img(filename)

            glass_res = self._predict(image, size)
            self._write_img(save_name, glass_res)

    def _predict(self, img, size):
        with torch.no_grad():
            output, output_boundary = self.model.evaluate(img)

            glass_res = output.argmax(1)[0].data.cpu().numpy().astype('uint8') * 127
            glass_res = cv2.resize(glass_res, size, interpolation=cv2.INTER_NEAREST)
        return glass_res

    def _read_img(self, img_name):
        ori_img = cv2.imread(img_name)
        h, w, _ = ori_img.shape
        return ori_img, (w, h)

    def _write_img(self, img_name, prediction):
        save_path = self.output_dir
        cv2.imwrite(os.path.join(save_path, '{}_glass.png'.format(img_name)), prediction)

    def _load_model(self, model_path):
        pass
