#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: paras_dir.py
@time: 2023/3/14 下午4:02
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import numpy as np
from tqdm import tqdm


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default="", help="")

    args = parser.parse_args()
    return args

def main():
    args = GetArgs()

    with open(args.input) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        arrays = [l.strip().split('/') for l in lines]

        root = set([a[0] for a in arrays])

        dataset = dict()
        for r in root:
            dataset[r] = dict()

        for i, arr in  enumerate(tqdm(arrays)):
            current = dataset[arr[0]]
            for j, a in enumerate(arr):
                if j == len(arr) - 1:
                    current.append(lines[i])
                elif j > 0:
                    if j + 1 == len(arr) - 1:
                        if a not in current.keys():
                            current[a] = []
                    else:
                        if a not in current.keys():
                            current[a] = dict()

                    current = current[a]

        print(dataset)

if __name__ == '__main__':
    main()
