from ruamel import yaml
import numpy as np
import open3d as o3d


# with open('/media/xin/data1/data/data_2023_11_21/data_2023_11_21_0/20231121_1145/slam_depth/40_1700538340127855_slam_depth.yaml', 'r') as f:
with open('/media/xin/data1/data/data_2023_11_21/data_2023_11_21_1/20231121_1145/slam_depth/53_1700538353693783_slam_depth.yaml', 'r') as f:
    # f.readline()
    yamls = yaml.YAML()
    result = yamls.load(f)
    data = result["depth"]["data"]
    data_arr = np.array(data)
    data_arr[data_arr == ".Nan"] = 0
    points = data_arr.astype(np.float64).reshape(-1,3)
    print(points.shape)
    # 使用open3d绘制点云图
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0, 0, 0]))  # 确定坐标轴方向
    coord_frame.transform(np.eye(4))  # 将坐标系与点云对齐
    o3d.visualization.draw_geometries([pcd, coord_frame], mesh_show_wireframe=True)

