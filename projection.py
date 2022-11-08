#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: projection.py
@time: 2022/10/11 下午3:51
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.file import Walk, MkdirSimple
from utils.module import GetConfigFile
from utils.imageprocess import GetR, GetP
import cv2
import numpy as np

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="dataset include cam0 & cam1")
    parser.add_argument("--output", type=str, help="output dir")

    args = parser.parse_args()
    return args

def Projection(img_file, config_file):
    if 'cam0' in img_file:
        right_file = img_file.replace('/cam0/', '/cam1/')
        left_img = cv2.imread(img_file)
        right_img = cv2.imread(right_file)

        rl, rr = GetR(config_file)
        pl, pr = GetP(config_file)

        R = rr.T * rl
        baseline = - pr[0, 3] / pr[0, 0]
        T = rr.T * np.array([baseline, 1e-10, 1e-10]).T

        concat = np.hstack([left_img, right_img])
        cv2.imshow('concat', concat)
        cv2.waitKey(0)

def main():
    args = GetArgs()

    valid_files = [] # todo
    root = ''

    if args.input.endswith('txt'):
        if not os.path.exists(args.filelist):
            print("file {} not exist.".format(args.input))
        else:
            valid_files = [f.strip('\n') for f in open(args.filelist, 'r').readlines()]
    elif os.path.isfile(args.input):
        root = len(os.path.dirname(args.input))

        current_dir = os.path.dirname(args.input)
        config_file = GetConfigFile(os.path.dirname(args.input))
        while not os.path.exists(config_file):
            current_dir = os.path.dirname(current_dir)
            config_file = GetConfigFile(current_dir)
        Projection(args.input, config_file)
    else:
        root = len(args.input.rstrip("/"))
        dirs = os.listdir(args.input)

        for d in dirs:
            config_file = GetConfigFile(args.input)
            path = os.path.join(args.input, d)
            if not os.path.isdir(path):
                continue

            if not os.path.exists(config_file):
                config_file = GetConfigFile(path)
                if not os.path.exists(config_file): # wanderful
                    dirs.extend([os.path.join(d, p) for p in os.listdir(path)])
                    continue

            print("in dir: ", path)
            files = Walk(path, ['jpg', 'png'])

            for f in files:
                Projection(f, config_file)


if __name__ == '__main__':
    main()