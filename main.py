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

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.imageprocess import Remap
from utils.file import MkdirSimple


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")

    args = parser.parse_args()
    return args


def main():
    args = GetArgs()
    image = cv2.imread(args.input)
    imageRemap = Remap(image)
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, os.path.basename(args.input))
        MkdirSimple(output_file)
        cv2.imwrite(output_file, imageRemap)

    cv2.namedWindow("remap")
    cv2.imshow("remap", np.hstack((image, imageRemap)))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()