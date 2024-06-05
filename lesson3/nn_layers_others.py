# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from common_tools import transform_invert, set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs/lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
# transforms.Compose用于将多个图像转换操作组合在一起。
# [transforms.ToTensor()]是一个列表，其中包含一个转换操作transforms.ToTensor()。
# transforms.ToTensor()将PIL图像或NumPy数组转换为PyTorch张量，并将像素值从0-255缩放到0-1之间。

img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img) # 将RGB图像转化为Pytorch张量
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# unsqueeze_是PyTorch张量的一个原地操作（in-place operation），用于在指定维度上增加一个大小为1的维度。
# dim=0指定在第0维增加一个维度。
# 原始张量的形状为C*H*W，增加维度后的形状为1*C*H*W，即B*C*H*W，其中B表示批次大小。

# ================================= create convolution layer ==================================

# ================ maxpool
flag = 1
# flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

# ================ avgpool
flag = 1
# flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# ================ avgpool divisor_override
flag = 1
# flag = 0
if flag:
    img_tensor = torch.ones((1, 1, 4, 4)) # size=(1, 1, 4, 4)表示生成形状为1*1*4*4的张量，其中1是批次大小，1是通道数，4是高度和宽度。
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{}\npooling_img:\n{}".format(img_tensor, img_pool))


# ================ max unpool
flag = 1
# flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor) # 将最大池化层应用到随机图像张量img_tensor上，得到池化后的输出张量img_pool和位置索引indices。

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# ================ linear
flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    # 权重矩阵的形状为(4, 3)，每行代表一个输出特征，每列代表一个输入特征。

    linear_layer.bias.data.fill_(0.5)
    # 将线性层的偏置设置为0.5。
    # 偏置是一个长度为4的向量，对应每个输出特征。
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)

# ================================= visualization ==================================
print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_pool)
plt.subplot(121).imshow(img_raw)
plt.show()

# transform_invert是一个自定义的反变换函数，用于将张量转换回图像。
# img_pool[0, 0:3, ...]表示选取池化后张量的第一张图像（第一个批次）和前三个通道（RGB）。
# img_transform是之前定义的图像变换操作。
# img_tensor.squeeze()去除多余的维度，将张量从形状(1, C, H, W)变为(C, H, W)。
# plt.subplot(122)表示创建一个1行2列的图，第2个子图。
# plt.imshow(img_pool)显示池化后的图像。
# plt.subplot(121)表示创建一个1行2列的图，第1个子图。
# plt.imshow(img_raw)显示池化前的图像。










