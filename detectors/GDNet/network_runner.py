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

import logging
import numpy as np
import torch.cuda

from pathlib import Path
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from typing import Tuple, Any
from tqdm import tqdm

from gdnet import GDNet
from misc import crf_refine
from common.network_runner_base import NetworkRunnerBase
from common.input_image import InputImage


class NetworkRunner(NetworkRunnerBase):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        log_path: Path,
        model_path: Path,
        do_crf_refine: bool,
        scale: int,
        calculate_secondary: bool,
    ):
        super().__init__(input_dir, output_dir, log_path, model_path)
        self.do_crf_refine = do_crf_refine
        self.calculate_secondary = calculate_secondary

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((scale, scale)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.to_pil = transforms.ToPILImage()

    def _predict(self, img, size):
        with torch.no_grad():
            img_var = Variable(self.img_transform(img).unsqueeze(0)).to(self.device)
            f1, f2, f3 = self.net(img_var)
            if self.calculate_secondary:
                f1 = f1.data.squeeze(0).cpu()
                f2 = f2.data.squeeze(0).cpu()
                f1 = np.array(transforms.Resize(size)(self.to_pil(f1)))
                f2 = np.array(transforms.Resize(size)(self.to_pil(f2)))
            else:
                f1 = f2 = None
            f3 = f3.data.squeeze(0).cpu()
            f3 = np.array(transforms.Resize(size)(self.to_pil(f3)))

            if self.do_crf_refine:
                if self.calculate_secondary:
                    f1 = crf_refine(np.array(img), f1)
                    f2 = crf_refine(np.array(img), f2)
                f3 = crf_refine(np.array(img), f3)
        return f1, f2, f3

    def _load_model(self, model_path):
        with_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if with_gpu else "cpu")
        logging.info(
            "CUDA is available, device = 'gpu'"
            if with_gpu
            else "CUDA is unavailable, device = 'cpu'"
        )
        self.net = GDNet().to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        logging.info("Loading model succeeded.")
        self.net.eval()

    def _image_gen(self) -> InputImage:
        for img_path in sorted(self.input_dir.iterdir()):
            img, (h, w) = self._read_img(img_path)
            yield InputImage(img, img_path, w, h)

    def _read_img(self, img_path):
        logging.info(f"Image {img_path.name} read")
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            logging.info(f"Image {img_path.name} is a gray image. Converting to RGB.")

        return img, img.size

    def _write_img(self, img_name, prediction):
        f1, f2, f3 = prediction
        logging.info(f"Image {img_name} processed. Writing results.")
        if self.calculate_secondary:
            Image.fromarray(f1).save(
                self.output_dir / Path(img_name.name + "_h" + img_name.suffix)
            )
            Image.fromarray(f2).save(
                self.output_dir / Path(img_name.name + "_l" + img_name.suffix)
            )
        Image.fromarray(f3).save(self.output_dir / Path(img_name))
