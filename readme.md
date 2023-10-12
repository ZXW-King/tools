# 说明
date @2021   
@ 孙昊


## 运行
### 去畸变
```
python3 main.py --input  /media/hao/U393/BASE/dir1 --output_dir /media/hao/U393/REMAP
```

最终会遍历 /media/hao/U393/BASE/dir1 下的所有文件，去畸变并放置在 /media/hao/U393/REMAP/dir1 下；
要求 config.yaml 文件在 /media/hao/U393/BASE/dir1 下，或者 /media/hao/U393/BASE/dir1 下的每个子文件夹都有一个 config.yaml 文件；


### 深度图显示
显示点云
```shell
# MobileLightStereo
/WORK/LIB/Python/flask/bin/python /WORK/CODE/Python/remap_python/pcl.py --file /media/hao/U393/madnet_test_data/Parker/TRAIN/data_2023_08_23/DEPTH/D10.7.0/disp --scale 256 --config /MVS/Parker/REMAP/TRAIN/data_2023_08_23/data_2023_0823_1/config.yaml --image /media/hao/U393/madnet_test_data/Parker/TRAIN/data_2023_08_23/REMAP/cam0 --bf 34.24 --show_pcl
# LightStereo
/WORK/LIB/Python/flask/bin/python /WORK/CODE/Python/remap_python/pcl.py --file /media/hao/U393/madnet_test_data/Parker/TRAIN/data_2023_08_23/DEPTH/D13.1.1/disp_scaleX256_uint16/cam0 --scale 256 --config /MVS/Parker/REMAP/TRAIN/data_2023_08_23/data_2023_0823_1/config.yaml --image /media/hao/U393/madnet_test_data/Parker/TRAIN/data_2023_08_23/REMAP/cam0 --bf 34.24 --show_pcl
```

数字键 1 和 2 可以切换颜色；

```shell script
python3 depth_view.py --image img_remap/left --depth depth/origin/left/gray --output depth/origin/
```
### 去畸变前后深度图对比
```shell script
python3 depth_remap_or_not.py --image_org img/left --depth_org depth/origin/left/gray --image_remap img_remap/left --depth_remap depth/remap/gray/left --output depth
```
