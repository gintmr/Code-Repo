# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    """
    判断一个数是否是2的幂
    :param n: 待判断的数
    :return: 如果是2的幂，返回True，否则返回False
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    # 初始化MSDeformAttn类
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        # 检查d_model是否能被n_heads整除
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        # 计算每个头的维度
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 定义采样偏移量
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 定义注意力权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 定义值投影
        self.value_proj = nn.Linear(d_model, d_model)
        # 定义输出投影
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化参数
        """
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 对输入进行线性变换
        value = self.value_proj(input_flatten)
        # 如果有padding mask，则将padding部分置为0
        if input_padding_mask is not None: # maskd_fill函数的作用是将mask为True的位置填充为value
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # 将value reshape为(N, Len_in, n_heads, d_model // n_heads)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads) ## 多头
        # 对query进行线性变换，得到sampling_offsets
        # sampling_offsets的shape为(N, Len_q, n_heads, n_levels, n_points, 2),含义是每个query在每个level的每个point的偏移量
        ## 计算每个查询向量在每个尺度和每个点的偏移量。
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # 对query进行线性变换，得到attention_weights
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 对attention_weights进行softmax
        # 将注意力权重进行softmax归一化，并调整维度
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # 如果reference_points的最后一个维度为2，则进行以下操作
        if reference_points.shape[-1] == 2: 
            # reference_points的作用是将query的参考点映射到输入特征图
            # 计算offset_normalizer
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            ## 实际上就是将最后一个维度的值交换顺序(3*2==>还是3*2,只不过顺序变了)

            # 计算sampling_locations
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # 如果reference_points的最后一个维度为4，则进行以下操作
        elif reference_points.shape[-1] == 4:
            # 计算sampling_locations
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # for amp
        if value.dtype == torch.float16:
            # for mixed precision
            output = MSDeformAttnFunction.apply(
            value.to(torch.float32), input_spatial_shapes, input_level_start_index, sampling_locations.to(torch.float32), attention_weights, self.im2col_step)
            # 将value转换为float32类型
            output = output.to(torch.float16)
            # 将output转换为float16类型
            output = self.output_proj(output)
            # 将output通过output_proj进行投影
            return output

        # 对value进行MSDeformAttnFunction操作
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        # 对output进行线性变换
        output = self.output_proj(output)
        return output
