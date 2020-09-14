"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    BaseLoss implementation

    If you want to implement a new loss function interface,
    it should inherit from the BaseLoss class.
"""


from importlib import import_module
from .submodule import *
import torch
import torch.nn as nn


def get(args):
    loss_name = args.model_name + 'Loss'
    module_name = 'loss.' + loss_name.lower()
    module = import_module(module_name)

    return getattr(module, loss_name)


class BaseLoss:
    def __init__(self, args):
        self.args = args

        self.loss_dict = {}
        self.loss_module = nn.ModuleList()

        # Loss configuration : w1*l1+w2*l2+w3*l3+...
        # Ex : 1.0*L1+0.5*L2+...
        for loss_item in args.loss.split('+'):
            weight, loss_type = loss_item.split('*')

            module_name = 'loss.submodule.' + loss_type.lower() + 'loss'
            module = import_module(module_name)
            loss_func = getattr(module, loss_type + 'Loss')(args)

            loss_tmp = {
                'weight': float(weight),
                'func': loss_func
            }

            self.loss_dict.update({loss_type: loss_tmp})
            self.loss_module.append(loss_func)

        self.loss_dict.update({'Total': {'weight': 1.0, 'func': None}})

    def __call__(self, sample, output):
        return self.compute(sample, output)

    def cuda(self, gpu):
        self.loss_module.cuda(gpu)

    def compute(self, sample, output):
        loss_val = []
        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is not None:
                loss_tmp = loss['weight'] * loss_func(sample, output)
                loss_val.append(loss_tmp)

        loss_val = torch.cat(loss_val, dim=1)
        loss_sum = torch.sum(loss_val)

        return loss_sum, loss_val
