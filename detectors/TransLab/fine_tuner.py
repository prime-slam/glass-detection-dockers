import os
import sys

from tools.train import Trainer
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = '/'.join(os.path.split(cur_path))
sys.path.append(root_path)


class FineTuner(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.model.train(mode=False)
        self.model.head.train()
        self.model.head_b.train()
        print('model loaded')



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
    trainer = FineTuner(args)
    trainer.train()