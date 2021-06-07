#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: imageprocess.py
@time: 2021/3/29 下午3:16
@desc: 
'''
import cv2
import numpy as np
import os

W, H  = 640, 400

def ReadPara(file):
    if not os.path.exists(file):
        print("not exist file <", file, ">.")
        exit(0)

    config = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)

    r = config.getNode('Rl').mat()
    p = config.getNode('Pl').mat()
    p[p>1] /= 2
    k = config.getNode('Kl').mat()
    k[k>1] /= 2
    d = config.getNode('Dl').mat()

    config.release()

    fisheye_x, fisheye_y = np.ndarray((H, W), np.float32), np.ndarray((H, W), np.float32)

    cv2.fisheye.initUndistortRectifyMap(k, d, r, p[0:3, 0:3], (W, H), cv2.CV_32FC1, fisheye_x, fisheye_y )

    return fisheye_x, fisheye_y


def Remap(image, fisheye_x, fisheye_y):
    imgaeRemap =  cv2.remap(image, fisheye_x, fisheye_y, cv2.INTER_LINEAR)
    return imgaeRemap

