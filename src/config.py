"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    All of the parameters are defined here.
"""


import time
import argparse


parser = argparse.ArgumentParser(description='NLSPN')


# Dataset
parser.add_argument('--dir_data',
                    type=str,
                    default='/HDD/dataset/NYUDepthV2_HDF5',
                    # default='/HDD/dataset/KITTIDepthCompletion',
                    help='path to dataset')
parser.add_argument('--data_name',
                    type=str,
                    default='NYU',
                    # default='KITTIDC',
                    choices=('NYU', 'KITTIDC'),
                    help='dataset name')
parser.add_argument('--split_json',
                    type=str,
                    default='../data_json/nyu.json',
                    # default='../data_json/kitti_dc.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    default=228,
                    # default=240,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=304,
                    # default=1216,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    default=0,
                    # default=100,
                    help='top crop size for KITTI dataset')


# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=7240,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0,1,2,3",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='multiprocessing port')
parser.add_argument('--num_threads',
                    type=int,
                    default=1,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')


# Network
parser.add_argument('--model_name',
                    type=str,
                    default='NLSPN',
                    choices=('NLSPN',),
                    help='model name')
parser.add_argument('--network',
                    type=str,
                    default='resnet34',
                    choices=('resnet18', 'resnet34'),
                    help='network name')
parser.add_argument('--from_scratch',
                    action='store_true',
                    default=False,
                    help='train from scratch')
parser.add_argument('--prop_time',
                    type=int,
                    default=18,
                    help='number of propagation')
parser.add_argument('--prop_kernel',
                    type=int,
                    default=3,
                    help='propagation kernel size')
parser.add_argument('--preserve_input',
                    action='store_true',
                    default=False,
                    help='preserve input points by replacement')
parser.add_argument('--affinity',
                    type=str,
                    default='TGASS',
                    choices=('AS', 'ASS', 'TC', 'TGASS'),
                    help='affinity type (dynamic pos-neg, dynamic pos, '
                         'static pos-neg, static pos, none')
parser.add_argument('--affinity_gamma',
                    type=float,
                    default=0.5,
                    help='affinity gamma initial multiplier '
                         '(gamma = affinity_gamma * number of neighbors')
parser.add_argument('--conf_prop',
                    action='store_true',
                    default=True,
                    help='confidence for propagation')
parser.add_argument('--no_conf',
                    action='store_false',
                    dest='conf_prop',
                    help='no confidence for propagation')
parser.add_argument('--legacy',
                    action='store_true',
                    default=False,
                    help='legacy code support for pre-trained models')


# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*L1+1.0*L2',
                    help='loss function configuration')
parser.add_argument('--opt_level',
                    type=str,
                    default='O0',
                    choices=('O0', 'O1', 'O2', 'O3'))
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='ckpt path')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training')
parser.add_argument('--test_only',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    help='number of epochs to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=12,
                    help='input batch size for training')
parser.add_argument('--max_depth',
                    type=float,
                    default=10.0,
                    # default=90.0,
                    help='maximum depth')
parser.add_argument('--augment',
                    type=bool,
                    default=True,
                    help='data augmentation')
parser.add_argument('--no_augment',
                    action='store_false',
                    dest='augment',
                    help='no augmentation')
parser.add_argument('--num_sample',
                    type=int,
                    default=500,
                    # default=0,
                    help='number of sparse samples')
parser.add_argument('--test_crop',
                    action='store_true',
                    default=False,
                    help='crop for test')


# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')


# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--decay',
                    type=str,
                    default='10,15,20',
                    help='learning rate decay schedule')
parser.add_argument('--gamma',
                    type=str,
                    default='1.0,0.2,0.04',
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

# Logs
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--save_full',
                    action='store_true',
                    default=False,
                    help='save optimizer, scheduler and amp in '
                         'checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=False,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=False,
                    help='save result images only with submission format')


args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = '../experiments/' + current_time + args.save
args.save_dir = save_dir
