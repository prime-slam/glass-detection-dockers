#  @Date    : 2023-03-19
#  @Editor  : Mikhail Kiselyov
#  @E-mail  : kiselev.0353@gmail.com
#  Provided as is

import logging
import time
from typing import Tuple, Any

from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm


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
        for img, img_path, shape in tqdm(self._image_gen()):
            self._write_img(img_path.name, self._predict(img, shape))
        logging.info("evaluation done")

    @abstractmethod
    def _predict(self, img, meta):
        pass

    @abstractmethod
    def _image_gen(self) -> Tuple[Any, Path, Tuple[int, int]]:
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
