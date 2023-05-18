#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: image_diff.py
@time: 2023/5/17 下午6:41
@desc: 
'''
import sys, os

import argparse
from utils.file import Walk, WriteTxt
import cv2
import numpy as np

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=str, default="", help="")

    args = parser.parse_args()
    return args


def main():
    args = GetArgs()

    files = Walk(args.root, ['jpg', 'jpeg', 'png'])
    files = [f for f in files if "imgs.L" in f]

    length = len(files)
    start = 0
    item = []

    for i in range(length):
        if i < start:
            continue

        f1 = files[i]
        img1 = cv2.imread(f1)

        for j in range(length - i - 1):
            id = j + i + 1
            f2 = files[id]
            img2 = cv2.imread(f2)

            diff = cv2.absdiff(img1, img2)
            mean_diff = np.mean(diff)

            if mean_diff < 30:
                continue

            item.append("{}, {}".format(f1, f2))
            print("{}, {}".format(f1, f2))
            start = id + 1
            # img_show = np.concatenate([img1, img2])
            # cv2.putText(img_show, "{}".format(mean_diff), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            # cv2.imshow("img", img_show)
            # cv2.waitKey(0)
            break

    WriteTxt("\n".join(item), os.path.join(args.root, "sync_diff.txt"), 'w')


if __name__ == '__main__':
    main()
