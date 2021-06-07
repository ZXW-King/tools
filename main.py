#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: main.py
@time: 2021/3/29 下午3:14
@desc: 
'''
import sys, os
import cv2
import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.imageprocess import Remap, ReadPara
from utils.file import MkdirSimple, Walk

CONFIG_FILE = 'config.yaml'

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file or dir")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--flip", type=bool, default=False, help="flip up-down")

    args = parser.parse_args()
    return args


def RemapFile(file, flip, fisheye_x, fisheye_y):
    image = cv2.imread(file)
    if image is None:
        print(file)
        return
    imageRemap = Remap(image, fisheye_x, fisheye_y)
    cv2.namedWindow("remap")
    cv2.imshow("remap", np.hstack((image, imageRemap)))
    cv2.waitKey(0)
    if flip:
        imageRemap = imageRemap[::-1, :, :]

    return imageRemap

def WriteImage(image, file, output_dir, root_len):
    if output_dir is not None:
        sub_path = file[root_len+1:]
        output_file = os.path.join(output_dir, sub_path)
        MkdirSimple(output_file)
        cv2.imwrite(output_file, image)

def main():
    args = GetArgs()

    if os.path.isfile(args.input):
        root = len(os.path.dirname(args.input))
        fisheye_x, fisheye_y = ReadPara(os.path.join(os.path.dirname(args.input), CONFIG_FILE))
        imageRemap = RemapFile(args.input, args.flip, fisheye_x, fisheye_y)
        WriteImage(imageRemap, args.input, args.output_dir , root)
    else:
        root = len(os.path.dirname(args.input))
        dirs = os.listdir(args.input)
        for d in dirs:
            print("in dir: ", d)
            files = Walk(os.path.join(args.input, d), ['jpg', 'png'])
            config_file = os.path.join(os.path.join(args.input, d), CONFIG_FILE)
            fisheye_x, fisheye_y = ReadPara(config_file)

            for f in tqdm(files):
                imageRemap = RemapFile(f, args.flip, fisheye_x, fisheye_y)
                if imageRemap is None:
                    print(f)
                    continue
                WriteImage(imageRemap, f, args.output_dir , root)


if __name__ == '__main__':
    main()