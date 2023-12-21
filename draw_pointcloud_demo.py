import cv2
import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt
import yaml

RADIAN_2_ANGLE = 180 / math.pi


def pointRotationAngle(point,cameraAngle):
    angle = cameraAngle / RADIAN_2_ANGLE
    y = point[1]
    z = point[2]
    point[1] = y * math.cos(angle) - z * math.sin(angle)
    point[2] = z * math.cos(angle) + y * math.sin(angle)
    return point


def getpoint(depth_image):
    depth_cam_matrix = np.array([[286.555, 0, 324.5],
                                 [0, 286.555, 197.339],
                                 [0, 0, 1]])
    # depth_cam_matrix = np.array([[283.808, 0, 340.495],
    #                              [0, 283.808, 210.246],
    #                              [0, 0, 1]])

    # 相机参数（根据你的相机设置进行调整）
    focal_length = 100  # 焦距
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

    # 转换为点云
    rows, cols = depth_image.shape
    y = []
    points = []
    # for u in range(296,296+54):
    for u in range(158,158+308):
        # for v in range(212,212+56):
        for v in range(80,80+198):
            Z = depth_image[v, u] / focal_length / 100
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            p = [X, Y, Z]
            point = pointRotationAngle(p, 10)
            if point[1] > -0.544 or point[1] < -1.11:
                continue
            points.append(point)
            y.append(-point[1])


    # arr = np.array(y)
    # # 最大值
    # max_val = np.max(arr)
    # print("最大值:", max_val)
    #
    # # 最小值
    # min_val = np.min(arr)
    # print("最小值:", min_val)
    #
    # # 均值
    # mean_val = np.mean(arr)
    # print("均值:", mean_val)
    # #
    # # 四分位数
    # quartiles = np.percentile(arr, [25, 50, 75,80,90,95])
    # print("第一四分位数 (Q1):", quartiles[0])
    # print("中位数 (Q2):", quartiles[1])
    # print("第三四分位数 (Q3):", quartiles[2])
    # print("第80%位数 (Q3):", quartiles[3])
    # print("第90%位数 (Q3):", quartiles[4])
    # print("第95%位数 (Q3):", quartiles[5])



    # 使用open3d绘制点云图
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0, 0, 0]))  # 确定坐标轴方向
    coord_frame.transform(np.eye(4))  # 将坐标系与点云对齐
    o3d.visualization.draw_geometries([pcd, coord_frame], mesh_show_wireframe=True)


if __name__ == '__main__':
    # 读取深度图像
    paker_cre_image = cv2.imread(
        '/media/xin/data1/data/parker_data_2023_08_22/result/CREStereo_MiDaS/CREStereo_big_object_100_tof/scale_tof/louti/data_2023_0822_2/20210223_1355/cam0/15_1614045311054993.png',
        cv2.IMREAD_UNCHANGED)
    paker_hitnet_image = cv2.imread(
        '/media/xin/data1/data/parker_data_2023_08_22/result/CREStereo_MiDaS/HitNet13.x/louti/data_2023_0822_2/20210223_1355/cam0/15_1614045311054993.png',
        cv2.IMREAD_UNCHANGED)
    rubby_cre_image = cv2.imread('/media/xin/data1/data/rubby/20220913/depth/cre/base/data_0912_2256_0/20220912_1639/cam0/20_1663000760493835.png',cv2.IMREAD_UNCHANGED)
    # rubby_hitnet_image = cv2.imread('/media/xin/data1/data/rubby/20220913/depth/hitnet/base/data_0912_2256_0/20220912_1639/cam0/20_1663000760493835.png',cv2.IMREAD_UNCHANGED)
    rubby_madnet_image = cv2.imread('/media/xin/data1/data/rubby/20220913/depth/madnet10.2.37/base/20220912_1639/cam0/20_1663000760493835.png',cv2.IMREAD_UNCHANGED)
    paker_madnet10_7_0_image = cv2.imread(
        '/media/xin/data1/data/parker_data_2023_08_22/result/CREStereo_MiDaS/MADNet10.7.0/louti/data_2023_0822_2/20210223_1355/cam0/15_1614045311054993.png',
        cv2.IMREAD_UNCHANGED)
    paker_madnet10_2_34_image = cv2.imread(
        '/media/xin/data1/data/parker_data_2023_08_22/result/CREStereo_MiDaS/MADNet10.2.34/louti/data_2023_0822_2/20210223_1355/cam0/15_1614045311054993.png',
        cv2.IMREAD_UNCHANGED)

    #################test####################
    paker_madnet10_2_37 = cv2.imread(
        '/media/xin/data1/test_data/depth_test/paker/madnet_10.2.37/data_2023_0822_2/20210223_1355/cam0/15_1614045311054993.png',
        cv2.IMREAD_UNCHANGED)
    rubby_madnet10_2_37 = cv2.imread(
        '/media/xin/data1/test_data/depth_test/rubby/madnet_10.2.37/20220912_1639/cam0/20_1663000760493835.png',
        cv2.IMREAD_UNCHANGED)
    rubby_madnet10_7_0 = cv2.imread(
        '/media/xin/data1/test_data/depth_test/rubby/madnet_10.7.0/20220912_1639/cam0/20_1663000760493835.png',
        cv2.IMREAD_UNCHANGED)
    getpoint(rubby_madnet10_7_0)
