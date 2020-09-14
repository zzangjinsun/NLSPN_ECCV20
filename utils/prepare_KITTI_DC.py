"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    This script copies stereo RGB images, poses and calibrations
    from KITTI Raw dataset to KITTI Depth Completion dataset.
"""

import os
import sys
import shutil
import argparse


parser = argparse.ArgumentParser(
    description="KITTI Depth Completion dataset preparer")

parser.add_argument('--path_root_dc', type=str, required=True,
                    help="Path to the Depth completion dataset")
parser.add_argument('--path_root_raw', type=str, required=True,
                    help="Path to the Raw dataset")

args = parser.parse_args()


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


# Reorganization functions
def prepare_reorganization():
    # Check directory existence
    check_dir_existence(args.path_root_dc)
    check_dir_existence(args.path_root_raw)

    print("Preparation of reorganization is done.")


def reorganize_train_val():
    for split in ['train', 'val']:
        path_dc = args.path_root_dc + '/' + split

        check_dir_existence(path_dc)

        # Get the list of sequences
        list_seq = os.listdir(path_dc)
        list_seq.sort()

        for seq in list_seq:
            path_raw_seq_src = args.path_root_raw + '/' + seq[0:10] + '/' + seq

            path_seq_dst = path_dc + '/' + seq

            try:
                # Copy stereo images and poses
                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_02', path_seq_dst + '/image_02'))
                shutil.copytree(path_raw_seq_src + '/image_02',
                                path_seq_dst + '/image_02')

                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_03', path_seq_dst + '/image_03'))
                shutil.copytree(path_raw_seq_src + '/image_03',
                                path_seq_dst + '/image_03')

                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/oxts', path_seq_dst + '/oxts'))
                shutil.copytree(path_raw_seq_src + '/oxts',
                                path_seq_dst + '/oxts')

                # Copy calibrations
                for calib in ['calib_cam_to_cam.txt',
                              'calib_imu_to_velo.txt',
                              'calib_velo_to_cam.txt']:
                    shutil.copy2(args.path_root_raw + '/' + seq[0:10] + '/'
                                 + calib, path_seq_dst + '/' + calib)
            except OSError:
                print("Failed to copy files for {}".format(seq))
                sys.exit(-1)

        print("Reorganization for {} split finished".format(split))


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    prepare_reorganization()
    reorganize_train_val()
