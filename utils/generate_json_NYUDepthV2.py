"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    This script generates a json file for the NYUDepthV2 HDF5 dataset.
"""


import os
import random

import argparse
import json
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="NYUDepthV2 HDF5 json generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to NYUDepthV2 HDF dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='nyu.json', help='Output file name')
parser.add_argument('--val_ratio', type=float, required=False,
                    default=0.05, help='Validation data ratio')
parser.add_argument('--csv_train', type=str, required=False,
                    default='nyudepth_hdf5_train.csv',
                    help='Train data csv file')
parser.add_argument('--csv_test', type=str, required=False,
                    default='nyudepth_hdf5_val.csv',
                    help='Test data csv file')
parser.add_argument('--num_train', type=int, required=False, default=1e8,
                    help='Maximum number of train data')
parser.add_argument('--num_val', type=int, required=False, default=1e8,
                    help='Maximum number of val data')
parser.add_argument('--num_test', type=int, required=False, default=1e8,
                    help='Maximum number of test data')
parser.add_argument('--seed', type=int, required=False, default=7240,
                    help='Random seed')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def main():
    check_dir_existence(args.path_root)
    check_dir_existence(args.path_out)

    assert (args.val_ratio >= 0.0) and (args.val_ratio <= 1.0), \
        "Validation set ratio should be in [0, 1] but {}".format(args.val_ratio)

    csv_train = pd.read_csv(args.csv_train)
    csv_test = pd.read_csv(args.csv_test)

    num_train = len(csv_train)
    num_test = len(csv_test)

    idx = np.arange(0, num_train)
    random.shuffle(idx)

    dict_json = {}

    num_val = int(num_train*args.val_ratio)
    num_train = num_train - num_val

    num_train = min(num_train, args.num_train)
    num_val = min(num_val, args.num_val)
    num_test = min(num_test, args.num_test)

    idx_train = idx[0:num_train]
    idx_val = idx[num_train:num_train+num_val]

    # Train
    list_data = []
    for i in idx_train:
        file_name = csv_train.iloc[i, 0]
        file_name = file_name[19:]

        dict_sample = {'filename': file_name}

        list_data.append(dict_sample)

    dict_json['train'] = list_data

    print('Training data : {}'.format(len(list_data)))

    # Val
    list_data = []
    for i in idx_val:
        file_name = csv_train.iloc[i, 0]
        file_name = file_name[19:]

        dict_sample = {'filename': file_name}

        list_data.append(dict_sample)

    dict_json['val'] = list_data

    print('Validation data : {}'.format(len(list_data)))

    # Test
    list_data = []
    list_files = os.listdir('{}/val/official'.format(args.path_root))
    list_files.sort()
    for f_name in list_files:
        file_name = 'val/official/{}'.format(f_name)

        dict_sample = {'filename': file_name}

        list_data.append(dict_sample)

    list_data = list_data[:num_test]

    dict_json['test'] = list_data

    print('Test data : {}'.format(len(list_data)))

    # Write to json files
    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('')

    main()
