# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
'''
这段代码的用途是验证Deformable DETR模型中的可变形注意力机制的前向传播和梯度计算的正确性。通过比较CUDA实现和PyTorch实现的结果，以及检查梯度计算的准确性，可以确保模型的稳定性和准确性。

'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


# 定义输入参数
N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
# 定义形状
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
# 计算每个级别的起始索引
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
# 计算总像素数
S = sum([(H*W).item() for H, W in shapes])


# 设置随机种子
torch.manual_seed(3)


# 检查前向传播与PyTorch是否相等（双精度）
@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    # 生成随机值
    value = torch.rand(N, S, M, D).cuda() * 0.01
    # 生成随机采样位置
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    # 生成随机注意力权重
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    # 归一化注意力权重
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    # 设置im2col_step
    im2col_step = 2
    # 使用PyTorch计算前向传播
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    # 使用CUDA计算前向传播
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
    # 检查前向传播是否相等
    fwdok = torch.allclose(output_cuda, output_pytorch)
    # 计算最大绝对误差
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    # 计算最大相对误差
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


# 检查前向传播与PyTorch是否相等（单精度）
@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    # 生成随机值
    value = torch.rand(N, S, M, D).cuda() * 0.01
    # 生成随机采样位置
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    # 生成随机注意力权重
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    # 归一化注意力权重
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    # 设置im2col_step
    im2col_step = 2
    # 使用PyTorch计算前向传播
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    # 使用CUDA计算前向传播
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).detach().cpu()
    # 检查前向传播是否相等
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    # 计算最大绝对误差
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    # 计算最大相对误差
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


# 检查梯度数值
def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    # 生成随机值
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    # 生成随机采样位置
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    # 生成随机注意力权重
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    # 归一化注意力权重
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    # 设置im2col_step
    im2col_step = 2
    # 定义函数
    func = MSDeformAttnFunction.apply

    # 设置梯度
    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    # 检查梯度
    gradok = gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    # 检查前向传播与PyTorch是否相等（双精度）
    check_forward_equal_with_pytorch_double()
    # 检查前向传播与PyTorch是否相等（单精度）
    check_forward_equal_with_pytorch_float()

    # 检查梯度数值
    for channels in [30, 32, 64, 71]:
        check_gradient_numerical(channels, True, True, True)
