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


### 单目深度图对比

```shell script
python3 depth_view.py --image img_remap/left --depth depth/origin/left/gray --output depth/origin/
```
### 去畸变前后深度图对比
```shell script
python3 depth_remap_or_not.py --image_org img/left --depth_org depth/origin/left/gray --image_remap img_remap/left --depth_remap depth/remap/gray/left --output depth
```

### 深的估计列表图像前后帧保存
```shell script
python  save_mono_image.py --file_name ./temp/diff_minute.txt --data_path /work/data/ABBY/REMAP/TRAIN/EVT/ --dest_path ./image_concat
```

### 深度估计训练用图像列表过滤
```shell script
python  save_mono_image.py --file_name ./temp/diff_minute.txt --data_path /work/data/ABBY/REMAP/TRAIN/EVT/ --dest_path ./image_concat
```