#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: transpose_reshape.py
@time: 2023/3/17 下午3:38
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import numpy as np


def MyTRTR2(array, size):
    print(array.shape)
    # print(array)
    H = size[0] * size[2]
    W = size[1] * size[2]
    cell = size[2]
    Hc, Wc = size[0:2]
    Cc = cell * cell
    output = np.zeros((H, W), dtype=int)
    for c in range(Cc):
        for h in range(Hc):
            row = h * cell + c // cell
            for w in range(Wc):
                v = array[c][h * Wc +w]
                col = w * cell + c % cell
                output[row][col] = v

    return output


def MyTRTR(array, size):
    print(array.shape)
    # print(array)
    H = size[0] * size[2]
    W = size[1] * size[2]
    cell = size[2]
    Hc, Wc = size[0:2]
    Cc = cell * cell
    output = np.zeros((H, W), dtype=int)
    for c in range(Cc):
        for h in range(Hc):
            row = h * cell + c // cell
            for w in range(Wc):
                v = array[c][h][w]
                col = w * cell + c % cell
                output[row][col] = v

    return output


def main():
    Hc, Wc, cell = 5, 3, 5
    array = np.arange(1, Hc * Wc * cell * cell + 1)
    nodust = array.reshape((cell * cell, Hc, Wc))
    nodust2 = array.reshape((cell * cell, Hc * Wc))
    nodust_trans = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust_trans, [Hc, Wc, cell, cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])

    print(heatmap.shape)
    print(heatmap)
    # result = MyTRTR(nodust, [Hc, Wc, cell])
    result = MyTRTR2(nodust2, [Hc, Wc, cell])
    print(result)


if __name__ == '__main__':
    main()
