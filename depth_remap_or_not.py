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
    parser.add_argument("--image_org", type=str, help="origin image dir")
    parser.add_argument("--depth_org", type=str, help="origin depth image dir")
    parser.add_argument("--image_remap", type=str, help="remap image dir")
    parser.add_argument("--depth_remap", type=str, help="remap depth image dir")
    parser.add_argument("--output", type=str, help="output image dir")

    args = parser.parse_args()
    return args

def Check(file):
    if not os.path.exists(file):
        print("not exist {}".format(file))

def main():
    args = GetArgs()
    image_list = Walk(args.image_org, ["jpg", "png", "jpeg"])
    fps = 24

    # video_file = os.path.join(args.output, "result.avi")
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    video_file = os.path.join(args.output, "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videoWriter = None

    for image_org_file in tqdm(image_list):
        Check(image_org_file)
        image_org = cv2.imread(image_org_file)

        depth_org_file = image_org_file.replace(args.image_org, args.depth_org)
        image_remap_file = image_org_file.replace(args.image_org, args.image_remap)
        depth_remap_file = image_org_file.replace(args.image_org, args.depth_remap)

        Check(depth_org_file)
        Check(image_remap_file)
        Check(depth_remap_file)

        depth_org = cv2.imread(depth_org_file)
        image_remap = cv2.imread(image_remap_file)
        depth_remap = cv2.imread(depth_remap_file)

        color_org = cv2.applyColorMap(depth_org, cv2.COLORMAP_HSV)
        color_remap = cv2.applyColorMap(depth_remap, cv2.COLORMAP_HSV)
        # out = cv2.applyColorMap(depth, cv2.COLORMAP_WINTER)

        concat = np.vstack((np.hstack((image_org, image_remap)), np.hstack((color_org, color_remap))))
        size = (concat.shape[1], concat.shape[0])
        if videoWriter is None:
            videoWriter = cv2.VideoWriter(video_file, fourcc, fps, size)

        cv2.imshow("result", concat)
        cv2.waitKey(1)

        output_color_file = image_org_file.replace(args.image_org, os.path.join(args.output, "compare_remap"))
        MkdirSimple(output_color_file)

        videoWriter.write(concat)

        cv2.imwrite(output_color_file, concat)

    videoWriter.release()


if __name__ == '__main__':
    main()
