import argparse

from pathlib import Path

from network_runner import NetworkRunner
from common.metrics_evaluator import MetricsEvaluator

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
        output_dir=args.output_dir,
        log_path=args.output_dir / "log.txt",
        model_path=args.pretrained_model_path,
        segmentron_args=segmentron_args
    ).run()

    if args.ground_truth_dir:
        MetricsEvaluator(
            prediction_dir=args.output_dir / "masks",
            ground_truth_dir=args.ground_truth_dir,
            output_path=args.output_dir / "metrics.json",
        ).evaluate()
