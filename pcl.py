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
from utils.file import Walk, MkdirSimple
from tqdm import tqdm


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", type=str, default="", help="")
    parser.add_argument("--bf", type=float, default="-1", help="")
    parser.add_argument("--scale", type=int, default="256")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--xml", type=str, default="")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--show_pcl", action="store_true")

    args = parser.parse_args()
    return args



def Depth2XYZ(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

def Crop(array, axis, min = 0, max = 300):
    array_flatten = array.reshape(-1, 3)

    less = array_flatten[:, axis] > min
    great = array_flatten[:, axis] < max

    plane_index = np.logical_and(less, great)

    if not any(plane_index):
        return None, None

    crop_pcl = array_flatten[plane_index, :]

    return crop_pcl

def Project(array, axis=0, min = 0, max = 300):
    axis1 = (axis + 1) % 3
    axis2 = (axis + 2) % 3
    select_index = [axis1, axis2] if axis1 < axis2 else [axis2, axis1]

    plane_array = array[:, select_index]

    min_val = np.min(plane_array)
    max_val = np.max(plane_array)

    # 归一化数组到 [0, 1] 范围
    normalized_arr = (plane_array - min_val) # / (max_val - min_val)

    return normalized_arr


def DrawPoint(array, name):
    W = 1280
    H = W
    radius = 2
    image = np.zeros([W, H])

    if array is not None:
        # array = array * (W - 1)
        array = array.astype(int)

        for a in array:
            cv2.circle(image, a, radius, (255, 0, 0), -1, lineType=cv2.LINE_4)

    cv2.circle(image, array[-1], radius * 3, (255, 0, 0), -1, lineType=cv2.LINE_4)

    left, right = np.min(array[:, 0]), np.max(array[:, 0])
    top, bottom= np.min(array[:, 1]), np.max(array[:, 1])
    image = image[top:bottom, left:right]
    return image

def GetColor(points):
    # 选择一个参考点（这里选择点云的第一个点）
    reference_point = points[0]

    # 计算每个点到参考点的距离
    distances = np.linalg.norm(points - reference_point, axis=1)

    # 将距离映射到颜色空间，这里使用一个简单的映射，可以根据需要进行调整
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)
    colors = np.stack([1 - normalized_distances, normalized_distances, np.zeros_like(distances)], axis=1)

    return colors

def ScaleRecovery(array, scale, bf):
    max_distance = 5
    array = array.astype(float)
    array /= scale
    if bf > 0:
        array = bf / array
        array[array > max_distance] = 0
        array *= 100
    else:
        pass

    return array

def GetKl(config):
    depth_cam_matrix = np.array([[334.6, 0, 319.7],
                                 [0, 334.5, 206.9],
                                 [0, 0, 1]])

    if config != "":
        fs = cv2.FileStorage(config, cv2.FILE_STORAGE_READ)
        depth_cam_matrix = fs.getNode("Kl").mat()

    return depth_cam_matrix

def Reflect4Show(pc):
    # 0: 左右 2: 前后 1： 上下
    axis = 1
    pc[:, :, axis] = -pc[:, :, axis]
    axis = 2
    pc[:, :, axis] = -pc[:, :, axis]

    return pc

def GetBox(W, H, name, xml_path):
    boxes = []
    if xml_path == "":
        return boxes

    # name = os.path.basename(name)
    file = os.path.join(xml_path, name)
    file = os.path.splitext(file)[0] + '.txt'
    if not os.path.exists(file):
        return boxes

    with open(file, 'r') as f:
        lines = f.readlines()
        data = [l.strip().split()[1:] for l in lines]
        for d in data:
            d = [float(i) for i in d]
            boxes.append([d[0] * W, d[1] * H, d[2] * W, d[3] * H])

    return boxes

def CropByBox(depth_map, name, xml_path):
    H, W = depth_map.shape
    boxes = GetBox(W, H, name, xml_path)

    crop_depth = np.zeros_like(depth_map)
    if len(boxes) < 1:
        return depth_map

    box = boxes[0]
    top, bottom = int(box[1] - box[3] / 2), int(box[1] + box[3] / 2)
    left, right = int(box[0] - box[2] / 2), int(box[0] + box[2] / 2)
    crop_depth[top:bottom, left:right] = depth_map[top:bottom, left:right]

    return  crop_depth, [left, right, top, bottom]

def GetImage(name, path):
    image = None
    if path == "":
        return image

    file = os.path.join(path, name)
    file = os.path.splitext(file)[0] + '.jpg'
    if not os.path.exists(file):
        return image

    image = cv2.imread(file)

    return image

def ResizePadding(W, H, C, img):
    # todo : C check, org and dst
    point2 = np.zeros((H, W, C))
    shape = img.shape
    if len(shape) < 3:
        h, w =  shape
        c = 1
    else:
        h, w, c = shape

    scale = min(H / h, W / w)
    newH, newW = round(h * scale), round(w * scale)
    image_resize = cv2.resize(img, (newW, newH))
    if 1 == c:
        point2[:newH, :newW, 0] = image_resize
    else:
        point2[:newH, :newW, :] = image_resize

    return point2

def PutText(image, text, left, bottom):
    margin = 5
    (text_width, text_height), _ = cv2.getTextSize(text, 2, 1, 1)
    left_box = left - margin
    bottom_box = bottom + margin
    right, top = left + text_width + margin, max(0, bottom - text_height - margin)
    image = cv2.rectangle(image, (left_box, top), (right, bottom_box), (0, 0, 0), -1)
    image = cv2.putText(image, text, (left, bottom), 2, 1, (255, 255, 255), 1)

    return image

def ShowAllImage(name, depth_map, image_rgb, image_point, box, show = True, pcl=None):
    depth_color = depth_map
    depth_color = (depth_color - np.min(depth_color)) * 255 / (np.max(depth_color) - np.min(depth_color))
    depth_color = depth_color.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_HOT)

    depth_color = cv2.rectangle(depth_color, (box[0], box[2]), (box[1], box[3]), (255, 255, 255), 2)
    depth_color = PutText(depth_color, "depth", 500, 50)

    if image_rgb is None:
        stack = depth_color
    else:
        image_rgb = cv2.rectangle(image_rgb, (box[0], box[2]), (box[1], box[3]), (255, 255, 255), 2)
        image_rgb = PutText(image_rgb, "left image", 450, 50)
        stack = np.vstack([image_rgb, depth_color])

    H, W, C = depth_color.shape
    image_point_resize = ResizePadding(W, H, C, image_point)
    image_point_resize = PutText(image_point_resize, "bird's eye view", 400, 50)
    if pcl is not None:
        pcl_resize = ResizePadding(W, H, C, pcl)
    else:
        pcl_resize = np.zeros((H, W, C))
    pcl_resize = PutText(pcl_resize, "point cloud", 400, 50)

    stack_right = np.vstack([image_point_resize, pcl_resize])
    stack = np.hstack([stack, stack_right])

    if show:
        cv2.imshow(name, stack)
        cv2.waitKey(100)

    return stack

def GetLine():
    # 创建一个坐标线的起点和终点
    h_range = 200
    starts = [[0, 2, 0], [0, 2, -h_range]]
    ends = [[0, 50, 0], [0, 50, -h_range]]
    line_sets = []
    for start_point, end_point in zip(starts, ends):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack((start_point, end_point)))
        lines = np.array([[0, 1]])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_sets.append(line_set)

    return line_sets

def main():
    args = GetArgs()

    files = Walk(args.file, ['jpg', 'png'])
    root_len = len(args.file.strip().rstrip('/'))

    for f in tqdm(files):
        file_name = f[root_len+1:]
        array = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        depth_map = ScaleRecovery(array, args.scale, args.bf)
        array_box, box = CropByBox(depth_map, file_name, args.xml)
        KL = GetKl(args.config)
        pc = Depth2XYZ(depth_map, KL, depth_scale = 1)
        pc = Reflect4Show(pc)
        pc_box = Depth2XYZ(array_box, KL, depth_scale = 1)
        pc_box = Reflect4Show(pc_box)

        # 创建一个 Open3D 点云对象并加载数据
        pc_flatten = pc.reshape(-1, 3)

        pc_crop = Crop(pc_box, axis=1, min = 10, max = 70)



        # todo : not skip
        if pc_crop is None:
            continue

        try:
            # todo : why
            pc_crop[-1, :] = [0, 0, 0] ## add origin point
        except:
            continue
        project_points = Project(pc_crop, axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_flatten)


        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc_crop)

        # pcd.paint_uniform_color([1, 0.706, 0])  # 黄色
        colors = GetColor(pc_crop)
        pcd2.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
        pcd2.colors = o3d.utility.Vector3dVector(colors)

        name = "flatten"
        cv2.namedWindow(name)
        image_point = DrawPoint(project_points, name)
        image_rgb = GetImage(file_name, args.image)

        if args.show_pcl:
            FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35, origin=[0, 0, 0])
            line_sets = GetLine()
            point_cloud = [FOR, pcd, pcd2] + line_sets
            o3d.visualization.draw_geometries(point_cloud)

            from PIL import ImageGrab
            screenshot = ImageGrab.grab(bbox=(600, 350, 1240, 900))  # 指定截图的区域
            screenshot = np.array(screenshot)
            cv2.imshow(name, screenshot)
            cv2.waitKey(100)
            image_show = ShowAllImage(name, depth_map, image_rgb, image_point, box, show=False, pcl=screenshot)

        else:
            image_show = ShowAllImage(name, depth_map, image_rgb, image_point, box, show=False)


        if "" != args.save_dir:
            save_file = os.path.join(args.save_dir, file_name)
            MkdirSimple(save_file)
            cv2.imwrite(save_file, image_show)

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
