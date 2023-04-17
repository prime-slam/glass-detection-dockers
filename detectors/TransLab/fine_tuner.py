import sys

from pathlib import Path

from tools.train import Trainer
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg

root_path = str(Path.cwd().resolve())
sys.path.append(root_path)


class FineTuner(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.model.train(mode=False)
        self.model.head.train()
        self.model.head_b.train()


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    fine_tuner = FineTuner(args)
    fine_tuner.train()
