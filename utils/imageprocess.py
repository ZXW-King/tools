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

K = [(7.1261170761727135e+02)/2, 0., (6.4968264752036998e+02)/2
    , 0.,(7.1253440977996888e+02)/2, (4.1337075038430754e+02)/2, 0., 0., 1.]
R = [(9.9985787572528106e-01), (5.3832163198578422e-03)
    , (-1.5976524408707376e-02), (-5.4883259985329599e-03)
    , (9.9996353651168690e-01), (-6.5424708462575197e-03)
    , (1.5940722313064530e-02), (6.6292253766121224e-03)
    , (9.9985096226539882e-01)]
D = [(-5.6028484984777541e-02), (-9.3122797502125758e-03)
    , (-1.4479074254791148e-02), (1.0911557039419725e-02)]
P = [620.138/2, 0., 698.277/2, 0, 0, 620.1377/2
    , 433.271/2, 0., 0., 0., 1., 0]

W, H  = 640, 400

def ReadPara():
    k = np.reshape(K, (3, 3))
    d = np.reshape(D, (4, 1))
    r = np.reshape(R, (3, 3))
    p = np.reshape(P, (3, 4))

    fisheye_x, fisheye_y = np.ndarray((H, W), np.float32), np.ndarray((H, W), np.float32)

    cv2.fisheye.initUndistortRectifyMap(k, d, r, p[0:3, 0:3], (W, H), cv2.CV_32FC1, fisheye_x, fisheye_y )

    return fisheye_x, fisheye_y

fisheye_x, fisheye_y = ReadPara()

def Remap(image):
    imgaeRemap =  cv2.remap(image, fisheye_x, fisheye_y, cv2.INTER_LINEAR)
    return imgaeRemap

