"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    Some of useful functions are defined here.
"""


import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val


def make_optimizer_scheduler(args, target):
    # optimizer
    if hasattr(target, 'param_groups'):
        # NOTE : lr for each group must be set by the network
        trainable = target.param_groups
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSProp
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    # scheduler
    decay = convert_str_to_num(args.decay, 'int')
    gamma = convert_str_to_num(args.gamma, 'float')

    assert len(decay) == len(gamma), 'decay and gamma must have same length'

    calculator = LRFactor(decay, gamma)
    scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)

    return optimizer, scheduler


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))
