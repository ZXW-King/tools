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
import shutil

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.imageprocess import Remap, ReadPara
from utils.file import MkdirSimple, Walk
import shutil

CONFIG_FILE = ['config.yaml', 'MODULE.yaml']

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file or dir")
    parser.add_argument("--output", type=str, default=None, help="output dir")
    parser.add_argument("--flip", action="store_true", help="flip up-down")
    parser.add_argument("--module", action="store_true", help="data capture module")

    args = parser.parse_args()
    return args


def Filp(image):
    imageFilp = image[::-1, ::-1, :]
    return  imageFilp

def RemapFile(image, fisheye_x, fisheye_y):
    imageRemap = Remap(image, fisheye_x, fisheye_y)
    # view = np.hstack((image, imageRemap))

    # cv2.namedWindow("remap")
    # cv2.imshow("remap", view)
    # cv2.waitKey(0)

    return imageRemap

def WriteImage(image, file, output, root_len):
    if output is not None:
        sub_path = file[root_len+1:]
        output_file = os.path.join(output, sub_path)
        MkdirSimple(output_file)
        cv2.imwrite(output_file, image)

def GetConfigFile(path):
    for file_name in CONFIG_FILE:
        file = os.path.join(path, file_name)
        if os.path.exists(file):
           break
    return file

def main():
    args = GetArgs()

    if os.path.isfile(args.input):
        root = len(os.path.dirname(args.input))
        config_file = GetConfigFile(os.path.dirname(args.input))
        fisheye_x, fisheye_y = ReadPara(config_file, args.module)
        imageRemap = RemapFile(args.input, args.flip, fisheye_x, fisheye_y)
        WriteImage(imageRemap, args.input, args.output , root)
    else:
        root = len(args.input.rstrip("/"))
        dirs = os.listdir(args.input)

        for d in dirs:
            config_file = GetConfigFile(args.input)
            d = os.path.join(args.input, d)
            if not os.path.isdir(d):
                continue
            print("in dir: ", d)
            files = Walk(d, ['jpg', 'png'])

            if len(files) == 0:
                continue

            if not os.path.exists(config_file):
                config_file = GetConfigFile(d)
            fisheye_x_l, fisheye_y_l, fisheye_x_r, fisheye_y_r = ReadPara(config_file, args.module)

            config_file_dst = config_file.replace(args.input, args.output)
            MkdirSimple(config_file_dst)
            shutil.copyfile(config_file, config_file_dst)

            count = 0
            for f in tqdm(files):
                image = cv2.imread(f)
                if image is None:
                    print("image is empty :", f)
                    continue

                if 'rgb' in f:
                    imageRemap = image
                elif 'cam0' in f:
                    imageRemap = RemapFile(image, fisheye_x_l, fisheye_y_l)
                elif 'cam1' in f:
                    imageRemap = RemapFile(image, fisheye_x_r, fisheye_y_r)
                else:
                    imageRemap = RemapFile(image, fisheye_x_l, fisheye_y_l)

                if imageRemap is None:
                    print(f)
                    continue

                if args.flip:
                    imageFilp = Filp(imageRemap)
                else:
                    imageFilp = imageRemap

                WriteImage(imageFilp, f, args.output , root)
                count += 1
                if count > 1000:
                    os.system('sync')
                    count = 0



if __name__ == '__main__':
    main()