import os
import sys
import argparse

from pathlib import Path

from network_runner import NetworkRunner
from common.metrics_evaluator import MetricsEvaluator
from segmentron.utils import options
from segmentron.config import cfg
from segmentron.utils.default_setup import default_setup
from tools.test_demo import Evaluator

if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to the .pth file",
        default=Path(__file__).parent / "200.pth",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory that contains the images",
        default=Path(__file__).parent / "demo" / "imgs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory in which the result images are written",
        default=Path(__file__).parent / "output",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=416,
        help="Scale parameter for resizing images. Default: 416",
    )
    parser.add_argument(
        "--do-crf-refine",
        action="store_true",
        help="Optional CRF refinement. Default: False",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=Path,
        default=None,
        help="Directory containing ground truth masks for input images."
             "If specified, metrics for predictions are calculated.",
    )
    parser.add_argument(
        "--calculate_secondary",
        action="store_true",
        help="records secondary images generated by the algorithm, their meaning not explicitly clear for now :D",
    )

    args, segmentron_args = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir / "masks"
    log_path = args.output_dir / "log.txt"
    pretrained_model_path = args.pretrained_model_path
    ground_truth_dir = args.ground_truth_dir


    s = '''--config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ./demo/imgs'''
    sys.argv = ['run.py'] + s.split()

    root_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(root_path)

    args = options.parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.DATASET.NAME = 'trans10k_extra'
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    # evaluator.eval()

    NetworkRunner(
        input_dir=input_dir,
        output_dir=output_dir / "masks",
        log_path=output_dir / "log.txt",
        model_path=pretrained_model_path,
        evaluator=evaluator
    ).run()

    if ground_truth_dir:
        MetricsEvaluator(
            prediction_dir=args.output_dir / "masks",
            ground_truth_dir=args.ground_truth_dir,
            output_path=args.output_dir / "metrics.json",
        ).evaluate()
