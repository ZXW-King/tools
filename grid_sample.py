#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: grid_sample.py
@time: 2023/4/3 下午9:11
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import numpy as np
import torch

W, H = 320, 200
cell = 8
c, h, w = 5, H // cell, W // cell


def torch_grid_sample(data, pts):
    data = torch.from_numpy(data.copy()).float()
    pts = torch.from_numpy(pts.copy()).float()

    pts[0, :] = (pts[0, :] / (W / 2.)) - 1
    pts[1, :] = (pts[1, :] / (H / 2.)) - 1
    pts = pts.transpose(0, 1).contiguous()
    pts = pts.view(1, 1, -1, 2)
    data = data.unsqueeze(dim=0)
    pts = pts.float()

    desc = torch.nn.functional.grid_sample(data, pts)
    desc = desc.data.cpu().numpy().reshape(c, -1)

    return desc


def grid_sample(data, pts):
    pts_c = pts
    pts_c = pts_c.astype(np.float16)

    pts_c[0, :] = (pts_c[0, :] + 1) / W
    pts_c[1, :] = (pts_c[1, :] + 1) / H

    pts_c[0, :] = np.clip(pts_c[0, :] * w - 1, 0, w - 1)
    pts_c[1, :] = np.clip(pts_c[1, :] * h - 1, 0, h - 1)

    floor = np.floor(pts_c).astype(np.uint8)
    ceil = np.ceil(pts_c).astype(np.uint8)

    a = (pts_c - floor)[0, :]
    b = (pts_c - floor)[1, :]

    x = data[:, floor[1], floor[0]]
    y = data[:, floor[1], ceil[0]]
    t = data[:, ceil[1], floor[0]]
    z = data[:, ceil[1], ceil[0]]

    out = (a * b * z) + ((1 - a) * (1 - b) * x) + ((1 - a) * b * t) + ((1 - b) * a * y)

    return out


def main():
    array = np.arange(1, c * h * w + 1)
    data = array.reshape((c, h, w))
    # print(data)

    point_count = 5
    pts_x = np.random.rand(point_count) * W
    pts_y = np.random.rand(point_count) * H
    pts = np.vstack([pts_x, pts_y])

    pts = pts.astype(np.int16)

    torch_out = torch_grid_sample(data, pts)
    np_out = grid_sample(data, pts)

    np.set_printoptions(suppress=True)

    print(torch_out)
    print(np_out)

    norm = np.linalg.norm(torch_out, axis=0)[np.newaxis, :]
    torch_out /= norm
    print(torch_out)

    norm = np.linalg.norm(np_out, axis=0)[np.newaxis, :]
    np_out /= norm
    print(np_out)


if __name__ == '__main__':
    main()
