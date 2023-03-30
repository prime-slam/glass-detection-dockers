from pathlib import Path

from common.network_runner_base import NetworkRunnerBase
from tools.test_demo import Evaluator


class NetworkRunner(NetworkRunnerBase):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        log_path: Path,
        model_path: Path,
        evaluator: Evaluator
    ):
        super().__init__(input_dir, output_dir, log_path, model_path)
        self.evaluator = evaluator

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
