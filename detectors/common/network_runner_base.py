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
import time

from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

from common.input_image import InputImage


class NetworkRunnerBase(ABC):
    def __init__(
        self, input_dir: Path, output_dir: Path, log_path: Path, model_path: Path
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_path = log_path

        output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(filename=log_path, level=logging.INFO)

        self._load_model(model_path)

    def run(self):
        for input_img in tqdm(self._image_gen()):
            self._write_img(
                input_img.path.name,
                self._predict(input_img.img, (input_img.w, input_img.h)),
            )
        logging.info("evaluation done")

    @abstractmethod
    def _predict(self, img, meta):
        pass

    @abstractmethod
    def _image_gen(self) -> InputImage:
        pass

    @abstractmethod
    def _write_img(self, img_name, prediction):
        pass

    @abstractmethod
    def _load_model(self, model_path):
        pass

    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start
