import os
import sys

from pathlib import Path

from common.network_runner_base import NetworkRunnerBase
from tools.test_demo import Evaluator
from segmentron.utils import options
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


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
        self.evaluator = Evaluator(args, self)

    def run(self):
        self.evaluator.eval()

    def _predict(self, img, meta):
        pass

    def _read_img(self, img_name):
        pass

    def _write_img(self, img_name, prediction):
        pass

    def _load_model(self, model_path):
        pass
