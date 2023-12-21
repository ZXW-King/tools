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

    # 相机参数（根据你的相机设置进行调整）
    focal_length = 1000  # 焦距
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

    # 转换为点云
    rows, cols = depth_image.shape
    y = []
    points = []
    z = []
    x = []
    for u in range(cols):
        for v in range(rows):
    # for u in range(236,236+66):
    #     for v in range(168,168+86):
    # 狗
    # for u in range(386,386+94):
    #     for v in range(172,172+90):
            Z = depth_image[v, u]
            Z = Z / focal_length
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            p = [X, Y, Z]
            point = pointRotationAngle(p, 10)
            if -point[1] < 0.058:
                continue
            points.append(point)
            y.append(-point[1])
            z.append(point[2])
            x.append(point[0])

    z_arr = np.array(z)
    y_arr = np.array(y)
    x_arr = np.array(x)
    print(f"z_mean:{z_arr.mean()},z_min:{z_arr.min()},z_max:{z_arr.max()}")
    print(f"y_mean:{y_arr.mean()},y_min:{y_arr.min()},y_max:{y_arr.max()}")
    # print(f"x_mean:{x_arr.mean()},x_min:{x_arr.min()},x_max:{x_arr.max()}")
    # 使用open3d绘制点云图
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=np.array([0, 0, 0]))  # 确定坐标轴方向
    coord_frame.transform(np.eye(4))  # 将坐标系与点云对齐
    o3d.visualization.draw_geometries([pcd, coord_frame], mesh_show_wireframe=True)
    return points


def drow_image(points):
    # 创建一个1000x1000的黑色背景图
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # 绘制点
    for point in points:
        x,y,z = point
        if 0 < -x*100 < 1000 and 0 < z*100 < 1000:
            background[int(-x*100)+500,int(z*100)+500] = (255, 255, 255)

    # 显示图像
    cv2.imshow("Points", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 读取深度图像
    rubby_bin = cv2.imread(
        '/home/xin/zhang/c_project/detection/psl/perception_standard_library/cmake-build-debug/result/depth_denoise/1703125626595.png',
        cv2.IMREAD_UNCHANGED)
    rubby_dog = cv2.imread(
        '/home/xin/zhang/c_project/detection/psl/perception_standard_library/cmake-build-debug/result_dog_after/depth_denoise/1703060809332.png',
        cv2.IMREAD_UNCHANGED)
    rubby_bin3 = cv2.imread(
        '/home/xin/zhang/c_project/detection/psl/perception_standard_library/cmake-build-debug/irDepth/1703128810520bin.png',
        cv2.IMREAD_UNCHANGED)
    points = getpoint(rubby_bin3)
    # drow_image(points)
