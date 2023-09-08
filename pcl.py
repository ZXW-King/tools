#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: pcl.py
@time: 2023/9/6 下午12:36
@desc: 
'''
import sys, os

import cv2

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../DL/relocation/'))

import argparse

import numpy as np
import open3d as o3d
from utils.file import Walk


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", type=str, default="", help="")
    parser.add_argument("--bf", type=float, default="-1", help="")

    args = parser.parse_args()
    return args



def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

def Project(array, axis=0):
    axis1 = (axis + 1) % 3
    axis2 = (axis + 2) % 3
    plane_array = array[:, :, (axis1, axis2)]

    min_val = np.min(plane_array)
    max_val = np.max(plane_array)

    # 归一化数组到 [0, 1] 范围
    normalized_arr = (plane_array - min_val) #  / (max_val - min_val)

    return normalized_arr


def DrawPoint(array):
    W = 1280
    H = W
    image = np.zeros([W, H])
    array = array * (W - 1)
    array = array.astype(np.int)
    image[array] = 255
    su = np.sum(image) / 255
    su2 = W * H

    cv2.imshow("name", image)
    cv2.waitKey(0)

def main():
    args = GetArgs()

    files = Walk(args.file, ['jpg', 'png'])


    i = 0
    for f in files:
        if i < 18:
            i += 1
            continue

        max_distance = 5
        depth_map = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        depth_map = depth_map.astype(float)
        depth_map /= 256
        if args.bf > 0:
            depth_map = args.bf / depth_map
            depth_map[depth_map > max_distance] = 0
            depth_map *= 100
        else:
            depth_map /= 100

        depth_cam_matrix = np.array([[302, 0, 300],
                                     [0, 302, 187],
                                     [0, 0, 1]])
        pc = depth2xyz(depth_map, depth_cam_matrix, depth_scale = 1)

        # 0: 左右 2: 前后 1： 上下
        axis = 1
        pc[:, :, axis] = -pc[:, :, axis]
        axis = 2
        pc[:, :, axis] = -pc[:, :, axis]

        print("Load a ply point cloud, print it, and render it")
        # 创建一个 Open3D 点云对象并加载数据
        pc_flatten = pc.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_flatten)

        # pcd = o3d.io.read_point_cloud("cat.ply")  # 这里的cat.ply替换成需要查看的点云文件
        # print(pcd)
        # print(np.asarray(pcd.points))

        FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([FOR, pcd], window_name=f)

    return
    # cv2.projectPoints()# 此时pc即为点云(point cloud)
    points = Project(pc, axis=2)
    DrawPoint(points)
    pc_flatten = pc.reshape(-1, 3)  # 等价于 pc = depth2xyz(depth_map, depth_cam_matrix, flatten=True)

    '''
    ################### 相机测距 ##################
    置 flatten=False 此时的pc是具有二维信息的 既shape为(720, 1280, 3) 否则为(720 * 1280, 3)

    此时 rgb 图片和点云的shape是一样的, 都为(720, 1280, 3)

    假设此时欲想测图片中的一个箱子的在真实世界中的长度, 箱子长边的一角a在rgb图片中的像素坐标为 (500, 100) -> (纵坐标, 横坐标), 长边的另一角b的像素坐标为(600, 200) -> (纵坐标, 横坐标)
    则a, b两点在 pc 中的xyz坐标就是已知的, 因此此时只需求取a, b两点对应的xyz坐标的欧氏距离就是该箱子的长度. 至于a, b两像素点的来源可能是你在rgb图片上对着箱子手动标出的, 也可能是算法得出的比如角点检测
    note1: 基于点云的测距不受相机角度影响, 只要能保证像素点对应的xyz精度够高, 相机可以从任何角度拍摄并得到正确的距离
    note2: 相机测距如果想测得准则对深度相机的精度要求比较高, 如果深度相机精度不高 测出来的长度会十分糟糕(在depth2xyz中可以看出xy的计算均与z有关), 本人在realense 435, 415, 515 上使用过测距功能, 精度上 515 > 415 > 435, 因此测距功能 515 > 415 > 435
    当然 工业级深度相机的精度大部分是要远高于realsense这类消费级深度相机的, 价钱也更贵一些
    '''
    '''
    # 代码如下:
    a = pc[500, 100] # (500, 100)为上面提到的像素坐标
    b = pc[600, 200] # (600, 200)为上面提到的像素坐标
    ax, ay, az = a
    bx, by, bz = b
    # 此时 distance 就是箱子的长度
    distance = ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
    # distance = np.linalg.norm(a - b)
    '''

if __name__ == '__main__':
    main()
