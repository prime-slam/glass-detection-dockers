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

import argparse

from pathlib import Path

from common.metrics_evaluator import MetricsEvaluator
from network_runner import NetworkRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to the .pth file",
        default=Path(__file__).parent / "16.pth",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory that contains the images",
        default=Path(__file__).parent / "input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory in which the result images are written",
        default=Path(__file__).parent / "output",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=Path,
        default=None,
        help="Directory containing ground truth masks for input images."
        "If specified, metrics for predictions are calculated.",
    )

    args, segmentron_args = parser.parse_known_args()

    NetworkRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir / "masks",
        log_path=args.output_dir / "log.txt",
        model_path=args.pretrained_model_path,
        segmentron_args=segmentron_args,
    ).run()

    if args.ground_truth_dir:
        MetricsEvaluator(
            prediction_dir=args.output_dir / "masks",
            ground_truth_dir=args.ground_truth_dir,
            output_path=args.output_dir / "metrics.json",
        ).evaluate()
