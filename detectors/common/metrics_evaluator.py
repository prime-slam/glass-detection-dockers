#  @Date    : 2023-03-19
#  @Editor  : Mikhail Kiselyov
#  @E-mail  : kiselev.0353@gmail.com
#  Provided as is

import json
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    accuracy_score,
    mean_absolute_error,
    balanced_accuracy_score,
)
from PIL import Image
from tqdm import tqdm


class MetricsEvaluator:
    def __init__(self, prediction_dir: Path, ground_truth_dir: Path, output_path: Path):
        self.prediction_dir = prediction_dir
        self.ground_truth_dir = ground_truth_dir
        self.metrics_output_path = output_path

        self.pred_files = sorted(prediction_dir.iterdir())
        self.gt_files = sorted(ground_truth_dir.iterdir())
        assert all(
            [
                pred_name.name == gt_name.name
                for pred_name, gt_name in zip(self.pred_files, self.gt_files)
            ]
        )

    def _evaluate_pair(self, pred_file, gt_file):
        pred = np.array(Image.open(pred_file).convert("1")).flatten()
        gt = np.array(Image.open(gt_file).convert("1")).flatten()
        return {
            "iou": jaccard_score(gt, pred),
            "f1_score": f1_score(gt, pred),
            "accuracy": accuracy_score(gt, pred),
            "mean_absolute_error": mean_absolute_error(
                gt.astype(np.int8), pred.astype(np.int8)
            ),
            "balanced_error_rate": 1 - balanced_accuracy_score(gt, pred),
        }

    def evaluate(self):
        result = {}

        for pred_file, gt_file in tqdm(zip(self.pred_files, self.gt_files)):
            result[pred_file.name] = self._evaluate_pair(pred_file, gt_file)

        with open(self.metrics_output_path, "w") as output_file:
            json.dump(result, output_file)
