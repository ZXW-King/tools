#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: depth_view.py
@time: 2022/8/24 下午2:27
@desc: put depth image on the origin image
'''
import sys, os

import argparse
from utils.file import Walk, MkdirSimple
import cv2
import numpy as np
from tqdm import tqdm


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, help="image dir")
    parser.add_argument("--depth", type=str, help="depth image dir")
    parser.add_argument("--output", type=str, help="output image dir")

    args = parser.parse_args()
    return args


def main():
    args = GetArgs()
    image_list = Walk(args.image, ["jpg", "png", "jpeg"])
    fps = 24

    # video_file = os.path.join(args.output, "result.avi")
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    video_file = os.path.join(args.output, "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videoWriter = None

    for image_file in tqdm(image_list):
        image = cv2.imread(image_file)
        dpeth_file = image_file.replace(args.image, args.depth)

        for suffix in ['.jpg', '.jpeg', '.png']:
            if not os.path.exists(dpeth_file):
                dpeth_file = os.path.splitext(dpeth_file)[0] + suffix
            else:
                break
        if not os.path.exists(dpeth_file):
            print("not exist {}".format(dpeth_file))
            continue
        depth = cv2.imread(dpeth_file)
        color = cv2.applyColorMap(depth, cv2.COLORMAP_HSV)
        # out = cv2.applyColorMap(depth, cv2.COLORMAP_WINTER)

        weighted = cv2.addWeighted(image, 0.3, color, 0.7, 2)

        concat = np.vstack((np.hstack((image, depth)), np.hstack((weighted, color))))
        size = (concat.shape[1], concat.shape[0])
        if videoWriter is None:
            videoWriter = cv2.VideoWriter(video_file, fourcc, fps, size)

        cv2.imshow("result", concat)
        cv2.waitKey(1)

        output_color_file = image_file.replace(args.image, os.path.join(args.output, "color/"))
        output_cover_file = image_file.replace(args.image, os.path.join(args.output, "cover/"))
        output_compare_file = image_file.replace(args.image, os.path.join(args.output, "compare/"))
        MkdirSimple(output_color_file)
        MkdirSimple(output_cover_file)
        MkdirSimple(output_compare_file)

        videoWriter.write(concat)

        cv2.imwrite(output_color_file, color)
        cv2.imwrite(output_cover_file, weighted)
        cv2.imwrite(output_compare_file, concat)

    videoWriter.release()


if __name__ == '__main__':
    main()
