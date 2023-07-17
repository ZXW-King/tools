#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: gather.py
@time: 2023/7/17 下午3:38
@desc: 
'''
import torch

batch_size = 2
channel = 2
height = 3
width = 3
max_disp = 3

import inspect

def print_current_function_name():
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    print("-" * len(function_name)*10)
    print(function_name)
    print("-" * len(function_name)*10)

def Gather():
    print_current_function_name()

    # 创建示例的 right 张量和 x_index 张量
    d_range = torch.arange(max_disp)
    d_range = d_range.view(1, 1, -1, 1, 1)
    print("drange: ", d_range)

    right = torch.arange(batch_size * channel * height * width).reshape(batch_size, channel, height, width)
    print("right： ", right)
    print("-" * 8, right.shape)

    x_index = torch.arange(width)
    print("index1： ", x_index)
    print("-" * 8, x_index.shape)
    x_index_org = torch.clip(4 * x_index - d_range + 1, 0, max_disp - 1)
    print("index2： ", x_index_org)
    print("-" * 8, x_index_org.shape)

    x_index_gather = x_index_org.repeat(batch_size, channel, 1, height, 1)
    print("index3： ", x_index_gather)
    print("-" * 8, x_index_gather.shape)

    # 调整 right 张量的形状以匹配 x_index 张量
    right_repeat = right.unsqueeze(2).repeat(1, 1, max_disp, 1, 1)
    print("right_repeat： ", right_repeat)
    print("-" * 8, right_repeat.shape)

    # index for slice

    index_slice = x_index_org.squeeze(0).squeeze(0).repeat(1, height, 1)
    print("index slice： ", index_slice)
    print("-" * 8, index_slice.shape)

    index_last = torch.arange(right.shape[-2])[:, None]
    print("index_last: ", index_last)
    print("-" * 8, index_last.shape)


    # 使用 torch.gather 进行索引操作
    gathered = torch.gather(right_repeat, dim=-1, index=x_index_gather)
    print("gather： ", gathered)  # 输出: torch.Size([2, 3, 2, 5])
    print("-" * 8, gathered.shape)

    # slice
    tensor_slice = right[:, :, index_last, index_slice]
    print("tensor_slice: ", tensor_slice)
    print("-" * 8, tensor_slice.shape)

    return


    # 输出每个样本的结果
    for i in range(batch_size):
        print("Sample", i + 1)
        print(gathered[i])


def Slice4D():
    print_current_function_name()

    import torch


    a = torch.arange(batch_size * channel * height * width).reshape(batch_size, channel, height, width)
    print("tensor: ", a)
    print("-" * 8, a.shape)
    index_1 = torch.LongTensor([[[[0, 1, 1], [0, 1, 0], [0, 1, 0]]]])
    print("index: ", index_1)
    print("-" * 8, index_1.shape)

    index_slice = index_1.squeeze(0).squeeze(0)
    print("index slice: ", index_slice)
    print("-" * 8, index_slice.shape)

    index_last = torch.arange(a.shape[-2])[:, None]
    print("index_last: ", index_last)
    print("-" * 8, index_last.shape)
    new_tensor = a[:, :, index_last, index_slice]


    index_gather = index_1.repeat(batch_size, channel, 1, 1)
    print("index_gather: ", index_gather)
    print("-" * 8, index_gather.shape)
    gathered = torch.gather(a, dim=-1, index=index_gather)
    # print(torch.gather(a, dim=1, index=index_2))

    print("slice: ", new_tensor)
    print("-" * 8, new_tensor.shape)
    print("gather: ", gathered)
    print("-" * 8, gathered.shape)


def Slice3D():
    print_current_function_name()

    import torch


    a = torch.arange(channel * height * width).reshape(channel, height, width)
    print("tensor: ", a)
    print("-" * 8, a.shape)
    index_1 = torch.LongTensor([[[0, 1, 1], [0, 1, 0]]])
    print("index: ", index_1)
    print("-" * 8, index_1.shape)

    index_2 = torch.arange(index_1.shape[-2])[:, None]
    print("index_last: ", index_2)
    print("-" * 8, index_2.shape)
    new_tensor = a[:, index_2, index_1].squeeze(1)
    gathered = torch.gather(a, dim=-1, index=index_1.repeat(channel, 1, 1))
    # print(torch.gather(a, dim=1, index=index_2))

    print("slice: ", new_tensor)
    print("-" * 8, new_tensor.shape)
    print("gather: ", gathered)
    print("-" * 8, gathered.shape)

def Slice2D():
    print_current_function_name()
    import torch

    a = torch.arange(height * width).reshape(height, width)
    print("tensor: ", a)
    print(a.shape)
    index_1 = torch.LongTensor([[0, 1, 1], [0, 1, 0]])
    print("index: ", index_1)

    index_2 = torch.arange(index_1.shape[0])[:, None]
    print("index_last: ", index_2)
    new_tensor = a[index_2, index_1]
    gathered = torch.gather(a, dim=-1, index=index_1)
    # print(torch.gather(a, dim=1, index=index_2))

    print("slice: ", new_tensor)
    print("gather: ", gathered)



def main():
    Slice2D()
    Slice3D()
    Slice4D()
    Gather()


if __name__ == '__main__':
    main()
