# Copyright (c) 2022, Mikhail Kiselyov, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import cv2
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import sys

from pathlib import Path
from torchvision import transforms

from common.network_runner_base import NetworkRunnerBase
from common.input_image import InputImage
from segmentron.utils import options
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import (
    make_data_sampler,
    make_batch_data_sampler,
)
from segmentron.config import cfg
from segmentron.utils.default_setup import default_setup


class NetworkRunner(NetworkRunnerBase):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        log_path: Path,
        model_path: Path,
        segmentron_args,
    ):
        root_path = Path(__file__).parent
        sys.path.append(str(root_path))

        cfg.TEST.TEST_MODEL_PATH = str(model_path)
        cfg.DEMO_DIR = str(input_dir)
        cfg.update_from_file("configs/trans10K/translab.yaml")
        cfg.PHASE = "test"
        cfg.ROOT_PATH = str(root_path)
        cfg.DATASET.NAME = "trans10k_extra"
        cfg.check_and_freeze()

        sys.argv = [__file__] + segmentron_args
        args = options.parse_args()
        default_setup(args)

        self.args = args
        self.device = torch.device(args.device)

        super().__init__(input_dir, output_dir, log_path, model_path)

        self._init_dataset()

    def _image_gen(self) -> InputImage:
        for img, _, img_name in self.val_loader:
            img_path = Path(img_name[0])
            ori_img = cv2.imread(str(img_path))
            h, w, _ = ori_img.shape
            yield InputImage(img, img_path, w, h)

    def _predict(self, img, shape):
        with torch.no_grad():
            output, output_boundary = self.model.evaluate(img)

            glass_res = output.argmax(1)[0].data.cpu().numpy().astype("uint8") * 127
            glass_res = cv2.resize(glass_res, shape, interpolation=cv2.INTER_NEAREST)
        return glass_res

    def _write_img(self, img_name, prediction):
        save_path = self.output_dir
        cv2.imwrite(str(save_path / img_name), prediction)

    def _init_dataset(self):
        input_dir = self.input_dir
        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            cfg.DATASET.NAME,
            root=input_dir,
            split="val",
            mode="val",
            transform=input_transform,
            base_size=cfg.TRAIN.BASE_SIZE,
        )

        val_sampler = make_data_sampler(
            val_dataset, shuffle=False, distributed=self.args.distributed
        )
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False
        )

        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=cfg.DATASET.WORKERS,
            pin_memory=True,
        )
        self.classes = val_dataset.classes

    def __set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def _load_model(self, model_path):
        self.model = get_segmentation_model(self.device).to(self.device)

        if hasattr(self.model, "encoder") and cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info(
                "set bn custom eps for bn in encoder: {}".format(
                    cfg.MODEL.BN_EPS_FOR_ENCODER
                )
            )
            self._set_batch_norm_attr(
                self.model.encoder.named_modules(), "eps", cfg.MODEL.BN_EPS_FOR_ENCODER
            )

        if self.args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        self.model.to(self.device)
        self.count_easy = 0
        self.count_hard = 0

        self.model.eval()

        if self.args.distributed:
            self.model = self.model.module
