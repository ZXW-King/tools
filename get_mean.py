#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: get_mean.py
@time: 2022/10/27 下午12:24
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.file import Walk
import cv2
import numpy as np
from tqdm import tqdm


def image_mean_channel_value(image):
    value_mean = []
    value_mean.append(np.mean(image[:, :, 0]))
    value_mean.append(np.mean(image[:, :, 1]))
    value_mean.append(np.mean(image[:, :, 2]))
    return value_mean


def image_list_mean_value(image_list):
    B_value = []
    G_value = []
    R_value = []
    calculate_images_size = 0
    for image_file in tqdm(image_list):
        if not os.path.exists(image_file):
            print("image file:", image_file, "not exist, continue")
        else:

            image = cv2.imread(image_file)
            if isinstance(image, np.ndarray):
                calculate_images_size += 1
                B_value.append(np.mean(image[:, :, 0]))
                G_value.append(np.mean(image[:, :, 1]))
                R_value.append(np.mean(image[:, :, 2]))
            else:
                print("image: ", image_file, "read error by opencv!")
            # print(image_file)
            # print(B_value[-1], G_value[-1], R_value[-1])
    print("total images: ", len(image_list), "calcualte mean value images: ", calculate_images_size)
    return [np.mean(B_value), np.mean(G_value), np.mean(R_value), np.std(B_value), np.std(G_value), np.std(R_value)]


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default="", help="")

    args = parser.parse_args()
    return args


def main():
    args = GetArgs()

    files = Walk(args.input, ['jpg', 'jpeg', 'png'])

    ret = image_list_mean_value(files)
    ret = np.asarray(ret)

    print("mean, var(BGR): ", ret)
    print("mean, var(BGR) normal: ", ret / 255)


if __name__ == '__main__':
    main()
