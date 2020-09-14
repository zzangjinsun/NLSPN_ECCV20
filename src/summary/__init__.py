"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    BaseSummary implementation

    If you want to implement a new summary interface,
    it should inherit from the BaseSummary class.
"""


from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def get(args):
    summary_name = args.model_name + 'Summary'
    module_name = 'summary.' + summary_name.lower()
    module = import_module(module_name)

    return getattr(module, summary_name)


class BaseSummary(SummaryWriter):
    def __init__(self, log_dir, mode, args):
        super(BaseSummary, self).__init__(log_dir=log_dir + '/' + mode)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.f_loss = '{}/loss_{}.txt'.format(log_dir, mode)
        self.f_metric = '{}/metric_{}.txt'.format(log_dir, mode)

        f_tmp = open(self.f_loss, 'w')
        f_tmp.close()
        f_tmp = open(self.f_metric, 'w')
        f_tmp.close()

    def add(self, loss=None, metric=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            self.loss.append(loss.data.cpu().numpy())
        if metric is not None:
            self.metric.append(metric.data.cpu().numpy())

    def update(self, global_step, sample, output):
        self.loss = np.concatenate(self.loss, axis=0)
        self.metric = np.concatenate(self.metric, axis=0)

        self.loss = np.mean(self.loss, axis=0)
        self.metric = np.mean(self.metric, axis=0)

        # Do update

        self.reset()
        pass

    def make_dir(self, epoch, idx):
        pass

    def save(self, epoch, idx, sample, output):
        pass
