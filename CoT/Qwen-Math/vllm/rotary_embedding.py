# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp
from vllm.utils import is_tpu


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    # 将输入张量x的最后一个维度分成两部分
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    # 将x2取负数，然后将x1和x2在最后一个维度上拼接
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        # 初始化函数，传入参数包括head_size、rotary_dim、max_position_embeddings、base、is_neox_style和dtype
        super().__init__()
        # 调用父类的初始化函数
        self.head_size = head_size
        # 设置head_size
        self.rotary_dim = rotary_dim
        # 设置rotary_dim
        self.max_position_embeddings = max_position_embeddings
        # 设置max_position_embeddings
        self.base = base
        # 设置base
        self.is_neox_style = is_neox_style
        # 设置is_neox_style
        self.dtype = dtype
        
        ###?
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        self.inv_freq = self.inv_freq.to('cuda')  # 移动到 GPU
        ###?
        
        # 设置dtype
        cache = self._compute_cos_sin_cache()
        # 调用_compute_cos_sin_cache函数，计算cos_sin_cache
        self.use_native2 = is_tpu() and is_neox_style
        # 判断是否使用TPU和is_neox_style
        if not self.use_native2:
            # 如果不使用TPU和is_neox_style
            cache = cache.to(dtype)
            # 将cache转换为dtype类型
            self.register_buffer("cos_sin_cache", cache, persistent=False)
            # 将cache注册为buffer，命名为cos_sin_cache，不持久化
        else:
            # 如果使用TPU和is_neox_style
            cos, sin = cache.chunk(2, dim=-1)
            # 将cache分成cos和sin两部分
            freqs_cis = cos + 1j * sin
            # 将cos和sin转换为复数形式
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        # 计算逆频率
        inv_freq = self._compute_inv_freq(self.base)
        # 生成一个从0到max_position_embeddings的序列
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        # 计算频率
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        # 计算频率的余弦值
        cos = freqs.cos()
        # 计算频率的正弦值
        sin = freqs.sin()
        # 将余弦值和正弦值拼接在一起
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation equivalent to forward().

        This method mimics the implementation of the custom CUDA kernel
        used in `forward_cuda()`.
        """
        # 将query和key的最后一个维度展平，并按照head_size进行分割
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        # 将query和key的前rotary_dim个维度提取出来
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        # 如果rotary_dim小于head_size，则将query和key的剩余维度提取出来
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        # 将cos_sin_cache移动到positions所在的设备，并转换为query的数据类型
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device, dtype=query.dtype)
        # 根据positions和offsets计算cos和sin
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                     if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 如果是neox_style，则将cos和sin重复两次，并在最后一个维度上增加一个维度
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        # 否则，将cos和sin在最后一个维度上重复两次，并在最后一个维度上增加一个维度
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        # 根据is_neox_style选择旋转函数
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
        # 如果rotary_dim小于head_size，则将query_rot和query_pass拼接起来，将key_rot和key_pass拼接起来
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
        # 否则，将query_rot和key_rot赋值给query和key
            query = query_rot
            key = key_rot
        query = query.flatten(-2)
        # 将query和key的最后一个维度展平
        key = key.flatten(-2)
        return query, key

    def forward_native2(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Another PyTorch-native implementation of forward().

        This method might perform better than `forward_native()` when compiled.
        """
        # 如果positions的维度为1，则batch_size为1，seq_len为positions的长度
        if positions.dim() == 1:
            batch_size = 1
            seq_len = positions.shape[0]
        # 否则，batch_size和seq_len分别为positions的维度
        else:
            batch_size, seq_len = positions.shape
        # 如果offsets不为空，则将positions加上offsets
        if offsets is not None:
            positions = positions + offsets
        # 根据positions从freqs_cis中选取对应的频率
        freqs_cis = self.freqs_cis.index_select(0, positions.flatten())
        # 将频率reshape为(batch_size, 1, seq_len, -1)
        freqs_cis = freqs_cis.view(batch_size, 1, seq_len, -1)

        # 获取query的形状
        query_shape = query.shape
        # 将query reshape为(batch_size, seq_len, -1, head_size)
        query = query.view(batch_size, seq_len, -1, self.head_size)
        # 获取query的rotary_dim部分
        query_rot = query[..., :self.rotary_dim]
        # 获取query的pass部分
        query_pass = query[..., self.rotary_dim:]
        # 对query的rotary_dim部分应用rotary_emb
        query_rot = _apply_rotary_emb(query_rot, freqs_cis)
        # 将query的rotary_dim部分和pass部分拼接起来，并reshape为原来的形状
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # 获取key的形状
        key_shape = key.shape
        # 将key reshape为(batch_size, seq_len, -1, head_size)
        key = key.view(batch_size, seq_len, -1, self.head_size)
        # 获取key的rotary_dim部分
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, freqs_cis)
        # 对key的rotary_dim部分应用rotary_emb
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        # 将key的rotary_dim部分和pass部分拼接起来，并reshape为原来的形状
        return query, key
        # 返回query和key
        

###########################################!
###########################################!
###########################################!
###########################################!    
###########################################!    
###########################################!    



        
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with support for default, reverse, and hybrid modes."""
        import os
        from vllm import _custom_ops as ops

        # 获取环境变量
        mode = os.getenv("PE_MODE", "default")
        budget = os.getenv("BUDGET", str(self.max_position_embeddings))

        # 处理 budget 参数
        if budget == "POS":
            budget = self.max_position_embeddings
        else:
            budget = int(budget)
        ####
        self.budget = budget
        # print(f"Budget = {budget}")
        # print(f"mode = {mode}")
        
        ####
        # 将 cos_sin_cache 移动到 positions 所在的设备，并转换为 query 的数据类型
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device, dtype=query.dtype)

        # 根据 mode 选择不同的实现方式
        if mode == "default":
            # 默认的 RoPE 实现
            if offsets is not None:
                ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                            self.cos_sin_cache, self.is_neox_style,
                                            self.rotary_dim, offsets)
            else:
                ops.rotary_embedding(positions, query, key, self.head_size,
                                    self.cos_sin_cache, self.is_neox_style)
        elif mode == "reverse":
            # 反向位置编码实现
            query, key = self._apply_reverse_position_emb_cuda(positions, query, key, offsets)
        elif mode == "hybrid":
            # 混合位置编码实现
            query, key = self._apply_hybrid_position_emb_cuda(positions, query, key, offsets)
        elif mode == "ldpe":
            # LDPE 模式实现
            query, key = self._apply_ldpe_position_emb_cuda(positions, query, key, offsets)
        elif mode == "lrpe":
            # LRPE 模式实现
            query, key = self._apply_lrpe_position_emb_cuda(positions, query, key, offsets)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return query, key
    
########### reverse
    def _apply_reverse_position_emb_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply reverse position embedding using CUDA."""
        from vllm import _custom_ops as ops

        # 调用默认的 RoPE 实现
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                        self.cos_sin_cache, self.is_neox_style,
                                        self.rotary_dim, offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                self.cos_sin_cache, self.is_neox_style)

        # 生成 delta_array
        delta_array = self._get_delta_array(self.budget)

        # 应用反向位置编码
        # print(f"delta_array_expand.shape = {delta_array.shape}")
        # print(f"query.shape = {query.shape}") 

        delta_array_q = delta_array.repeat(1, (query.shape[1]//delta_array.shape[1]))
        delta_array_k = delta_array.repeat(1, (key.shape[1]//delta_array.shape[1]))
        delta_array_q = delta_array_q.expand(query.shape[0], -1)
        delta_array_k = delta_array_k.expand(key.shape[0], -1) 
              
        delta_array_q = delta_array_q.to(query.device)
        delta_array_k = delta_array_k.to(key.device)
        
        query = query * delta_array_q
        key = key * delta_array_k

        query = query.to(dtype=torch.bfloat16)
        key = key.to(dtype=torch.bfloat16)

        return query, key
    
  

    def _get_delta_array(self, length: int) -> torch.Tensor:
        """Generate a delta array for reverse position embedding."""
        delta_array = torch.zeros(1, length)  # 创建一个全为0的数组
        for i in range(length):
            delta_array[0, i] = 1 / ((1 + (1 / length) * i) ** 0.3)  # 多项式衰减
        return delta_array  # 返回delta_array
################## reverse



########### hybrid

    def _apply_hybrid_position_emb_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply hybrid position embedding (RoPE + Reverse Position Embedding) using CUDA."""
        from vllm import _custom_ops as ops

        # 将 query 和 key 分割为两部分
        query_part1, query_part2 = query[..., :64], query[..., 64:]
        key_part1, key_part2 = key[..., :64], key[..., 64:]

        # 对第一部分应用默认的 RoPE
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query_part1, key_part1, self.head_size,
                                        self.cos_sin_cache[..., :64], self.is_neox_style,
                                        self.rotary_dim, offsets)
        else:
            ops.rotary_embedding(positions, query_part1, key_part1, self.head_size,
                                self.cos_sin_cache[..., :64], self.is_neox_style)

        # 对第二部分应用反向位置编码 
        query_part2, key_part2 = self._apply_reverse_position_emb_cuda(positions, query_part2, key_part2, offsets)

        # 合并两部分
        query = torch.cat((query_part1, query_part2), dim=-1)
        key = torch.cat((key_part1, key_part2), dim=-1)

        return query, key            

##########hybrid



################ldpe
    def _apply_ldpe_position_emb_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LDPE (Length-Dependent Positional Embedding) using CUDA."""
        from vllm import _custom_ops as ops

        # 调用默认的 RoPE 实现
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                        self.cos_sin_cache, self.is_neox_style,
                                        self.rotary_dim, offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                self.cos_sin_cache, self.is_neox_style)

        # 计算 LDPE
        ldpe = self._compute_ldpe(positions, self.budget, self.rotary_dim)
        # print(f"ldpe.shape = {ldpe.shape}") ## ([32768, 128])
        # 将ldpe在第1个维度上扩展1个维度，在第-1个维度上扩展1个维度，并将结果移动到query所在的设备上
        # ldpe_expand = ldpe.unsqueeze(1).unsqueeze(-1).to(query.device) 
        # print(f"ldpe_expand.shape = {ldpe.shape}") ## ([32768, 1, 128, 1])
        # print(f"query.shape = {query.shape}") 
        ## query.shape = ([32768, 2048])
        ## key.shape = ({32768, 256})
        # 叠加 LDPE
        
        
        ###ddd 由于多头注意力的参与，需要将维度repeat
        nhead_q = query.shape[1] // 128
        nhead_k = key.shape[1] // 128
        nhead_ldpe = ldpe.shape[1] // 128
        n_q = nhead_q // nhead_ldpe
        n_k = nhead_k // nhead_ldpe
    
        ldpe_q = ldpe.repeat(1, n_q)
        ldpe_k = ldpe.repeat(1, n_k)
        
        query = query + ldpe_q
        key = key + ldpe_k

        ###ddd 由于计算精度的限制，需要检查精度,全部设置为 bfloat16
        # print(f"query.dtype = {query.dtype}")  ### float32

        # 检查并转换数据类型
        if query.dtype != torch.float16 and query.dtype != torch.bfloat16:
            query = query.half()  # 转换为 fp16
        if key.dtype != torch.float16 and key.dtype != torch.bfloat16:
            key = key.half()  # 转换为 fp16
        # print(f"query.dtype = {query.dtype}")  ### flat16
        
        query = query.to(dtype=torch.bfloat16)
        
        key = key.to(dtype=torch.bfloat16)
        
        return query, key



    def _compute_ldpe(
        self,
        positions: torch.Tensor,
        length: int,
        dim: int,
    ) -> torch.Tensor:
        """Compute LDPE (Length-Dependent Positional Embedding)."""
        # 计算逆频率
        # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # inv_freq = inv_freq.to(positions.device)  # 将 inv_freq 移动到 GPU

        # 计算位置与长度的差值
        t = length - positions  # [batch_size, seq_len]
        t = t.to(positions.device)  # 将 t 移动到 GPU
        # 计算频率
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)  # [batch_size, seq_len, dim/2]
        # print(f"freqs.shape = {freqs.shape}") ## ([32768, 64])
        cos = freqs.cos()  # [batch_size, seq_len, dim/2]
        sin = freqs.sin()  # [batch_size, seq_len, dim/2]
        # ldpe = torch.cat((sin, cos), dim=-1)  # [batch_size, seq_len, dim]
        stacked = torch.stack((sin, cos), dim=-1)
        ldpe = stacked.reshape(freqs.shape[0], -1)
        return ldpe


###########ldpe


#########lrpe

    def _apply_lrpe_position_emb_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LRPE (Length-Reversed Positional Embedding) using CUDA."""
        from vllm import _custom_ops as ops

        # 调用默认的 RoPE 实现
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                        self.cos_sin_cache, self.is_neox_style,
                                        self.rotary_dim, offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)

        # 计算 LRPE
        lrpe = self._compute_lrpe(positions, self.budget, self.rotary_dim)
        
        # lrpe_expand = lrpe.unsqueeze(1).unsqueeze(-1).to(query.device)

        
        ###ddd 由于多头注意力的参与，需要将维度repeat
        nhead_q = query.shape[1] // 128
        nhead_k = key.shape[1] // 128
        nhead_ldpe = lrpe.shape[1] // 128
        n_q = nhead_q // nhead_ldpe
        n_k = nhead_k // nhead_ldpe
    
        ldpe_q = lrpe.repeat(1, n_q)
        ldpe_k = lrpe.repeat(1, n_k)
        
        query = query + ldpe_q
        key = key + ldpe_k
        ###ddd 由于计算精度的限制，需要检查精度,全部设置为 bfloat16

        query = query.to(dtype=torch.bfloat16)
        
        key = key.to(dtype=torch.bfloat16)
          

        return query, key


    def _compute_lrpe(
        self,
        positions: torch.Tensor,
        length: int,
        dim: int,
    ) -> torch.Tensor:
        """Compute LRPE (Length-Reversed Positional Embedding)."""
        # inv_freq = 1.0 / ((length * 10000) ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # inv_freq = inv_freq.to(positions.device) 
        t = positions  # [batch_size, seq_len]
        t = t.to(positions.device)  # 将 t 移动到 GPU
        
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)  # [batch_size, seq_len, dim/2]
        cos = freqs.cos()  # [batch_size, seq_len, dim/2]
        sin = freqs.sin()  # [batch_size, seq_len, dim/2]
        # lrpe = torch.cat((sin, cos), dim=-1)  # [batch_size, seq_len, dim]
        stacked = torch.stack((sin, cos), dim=-1)
        # print(f"stacked.shape = {stacked.shape}") ## ([32768, 64, 2])
        lrpe = stacked.reshape(freqs.shape[0], -1)
        # print(f"lrpe.shape = {lrpe.shape}") ## ([32768, 128])

        return lrpe
 
    
#######lrpe
    
    

###########################################!
###########################################!
###########################################!
###########################################!    
###########################################!    
###########################################!    
    
    
    
    
    
    
    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm._ipex_ops import ipex_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_tpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 根据self.use_native2的值选择调用self.forward_native2或self.forward_native
        forward_fn = (self.forward_native2
                      if self.use_native2 else self.forward_native)
        # 调用forward_fn函数，传入positions, query, key, offsets参数
        return forward_fn(positions, query, key, offsets)

    def extra_repr(self) -> str:
        # 返回模型的额外表示
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        # 添加头大小和旋转维度
        s += f", max_position_embeddings={self.max_position_embeddings}"
        # 添加最大位置嵌入
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        # 添加基数和是否为neox风格
        return s


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings) -
            (self.scaling_factor - 1))**(self.rotary_dim /
                                         (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding.

    Based on the original RotaryEmbedding implementation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        short_factor: List[float],
        long_factor: List[float],
        short_mscale: float = 1.0,
        long_mscale: float = 1.0,
    ):
        super().__init__()

        if rotary_dim != head_size:
            raise ValueError(
                f"`Phi3LongRoPEScaledRotaryEmbedding` does not support \
                    rotary_dim != head_size ({rotary_dim}!={head_size}).")
        if is_neox_style is False:
            raise ValueError(
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."
            )

        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        scale = (self.max_position_embeddings /
                 self.original_max_position_embeddings)

        if scale <= 1.0:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))

        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings, short_factor, short_mscale)
        short_cache = short_cache.to(dtype)
        self.register_buffer("short_cos_sin_cache",
                             short_cache,
                             persistent=False)

        long_cache = self._compute_cos_sin_cache(max_position_embeddings,
                                                 long_factor, long_mscale)
        long_cache = long_cache.to(dtype)
        self.register_buffer("long_cos_sin_cache",
                             long_cache,
                             persistent=False)

        long_short_cache = torch.cat(
            [self.short_cos_sin_cache, self.long_cos_sin_cache], dim=0)
        self.register_buffer("long_short_cos_sin_cache",
                             long_short_cache,
                             persistent=False)

    def _compute_inv_freq(self, rescale_factors: List[float]) -> torch.Tensor:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (rescale_factors * (self.base**(torch.arange(
            0, self.head_size, 2, dtype=torch.float) / self.head_size)))
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_position_embeddings: int,
        rescale_factors: List[float],
        mscale: float,
    ) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale * self.scaling_factor
        sin = freqs.sin() * mscale * self.scaling_factor
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        k = self.original_max_position_embeddings
        long_prompt_offset = (torch.any(positions > k).float() *
                              torch.full_like(positions, k)).long()
        idx = (torch.add(positions, long_prompt_offset)
               if long_prompt_offset is not None else positions)
        self.long_short_cos_sin_cache: torch.Tensor = (
            self.long_short_cos_sin_cache.to(idx.device))
        idx = torch.add(idx, offsets) if offsets is not None else idx
        cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)

        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)

        query = query * cos + _rotate_neox(query) * sin
        key = key * cos + _rotate_neox(key) * sin

        return query.flatten(-2), key.flatten(-2)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
                                self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device="cuda",
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        # print("Cache shape", cache.shape)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                     if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


class GemmaRotaryEmbedding(RotaryEmbedding):

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        # https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/gemma/modeling_gemma.py#L107
        inv_freq = 1.0 / (base**(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float() /
            self.rotary_dim))
        return inv_freq


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
) -> RotaryEmbedding:
    
    # print("################################################################################")
    # print(f"rope_scaling = {rope_scaling}") ## NONE
    # print("################################################################################")
    # print(f"head_size = {head_size}") ## 128
    
    
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dtype)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    
    
    
    
    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                     is_neox_style, dtype)
        
        
    else:
        scaling_type = rope_scaling["type"]
        # The correct one should be "longrope" but keep "su" here
        # for backward compatible
        if scaling_type != "su" and scaling_type != "longrope":
            scaling_factor = rope_scaling["factor"]
        if scaling_type == "linear":
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "dynamic":
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style,
                scaling_factor, dtype)
        elif scaling_type == "yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
                                                    original_max_position,
                                                    base, is_neox_style,
                                                    scaling_factor, dtype,
                                                    **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        # The correct one should be "longrope" but keep "su" here
        # for backward compatible
        elif scaling_type == "su" or scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size, rotary_dim, max_position, original_max_position,
                base, is_neox_style, dtype, short_factor, long_factor,
                **extra_kwargs)
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
