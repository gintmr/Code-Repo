# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch, os
from torch import nn
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):

    """
    匈牙利算法
    这个类计算目标和网络预测之间的匹配
    出于效率原因，目标不包括no_object。因此，通常情况下，预测的数量多于目标。在这种情况下，我们进行1对1的最佳匹配，
    而其他预测则未匹配（因此被视为非对象）。
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """创建匹配器
        参数:
            cost_class: 这是匹配成本中分类误差的相对权重
            cost_bbox: 这是匹配成本中边界框坐标L1误差的相对权重
            cost_giou: 这是匹配成本中边界框giou损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class  # 分类误差的相对权重
        self.cost_bbox = cost_bbox  # 边界框坐标L1误差的相对权重
        self.cost_giou = cost_giou  # 边界框giou损失的相对权重
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha  # 焦点损失的alpha参数

    # 在不计算梯度的情况下执行前向传播
    @torch.no_grad()
    def forward(self, outputs, targets, label_map):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0

        # 将目标标签转换为新的标签映射
        new_label_map=label_map[tgt_ids.cpu()]

        # 计算负样本的分类成本
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # 计算正样本的分类成本
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # 将新的标签映射移动到pos_cost_class所在的设备上
        new_label_map=new_label_map.to(pos_cost_class.device)
        
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # cost_class=(pos_cost_class @ new_label_map.T - neg_cost_class@ new_label_map.T)
        cost_class=[]
        for idx_map in new_label_map:
            idx_map = idx_map / idx_map.sum()
            cost_class.append(pos_cost_class @ idx_map - neg_cost_class@ idx_map)
        if cost_class:
            cost_class=torch.stack(cost_class,dim=0).T
        else:
            cost_class=torch.zeros_like(cost_bbox)
        # Compute the L1 cost between boxes
        

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # import pdb;pdb.set_trace()
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        C[torch.isnan(C)] = 0.0 # 处理可能的 NaN 和 Inf 值。
        C[torch.isinf(C)] = 0.0

        # 获取targets中每个元素的"boxes"字段的长度，并存储在sizes列表中
        """
        这段代码执行以下步骤：
        1. 获取批次中每个元素的目标框数量，并将这些计数存储在 `sizes` 列表中。
        2. 根据目标框的大小，沿最后一个维度分割成本矩阵 `C`。
        3. 对每个分割后的成本矩阵应用线性求和分配算法（匈牙利算法），以找到最优分配。
        4. 将最优分配的索引收集到 `indices` 列表中。
        """
        sizes = [len(v["boxes"]) for v in targets]
        # 尝试执行以下代码
        try:
            # 对C进行split操作，按照sizes列表中的元素进行分割，最后一个维度为-1，即最后一个维度
            # 对分割后的每个元素c，使用linear_sum_assignment函数进行计算，得到indices列表
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        except:
            print("warning: use SimpleMinsumMatcher")
            indices = []
            device = C.device
            for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
                weight_mat = c[i]
                idx_i = weight_mat.min(0)[1]
                idx_j = torch.arange(_size).to(device)
                indices.append((idx_i, idx_j))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


    
class SimpleMinsumMatcher(nn.Module):
    """
    简单最小和匹配器
    这个类计算目标和网络预测之间的匹配
    出于效率原因，目标不包括no_object。因此，通常情况下，预测的数量多于目标。在这种情况下，我们进行1对1的最佳匹配，
    而其他预测则未匹配（因此被视为非对象）。
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """
        创建匹配器
        参数:
            cost_class: 这是匹配成本中分类误差的相对权重
            cost_bbox: 这是匹配成本中边界框坐标L1误差的相对权重
            cost_giou: 这是匹配成本中边界框giou损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class  # 分类误差的相对权重
        self.cost_bbox = cost_bbox  # 边界框坐标L1误差的相对权重
        self.cost_giou = cost_giou  # 边界框giou损失的相对权重
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        
        
        self.focal_alpha = focal_alpha  # 焦点损失的alpha参数


    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行匹配
        参数:
            outputs: 这是一个字典，至少包含以下条目:
                 "pred_logits": 维度为 [batch_size, num_queries, num_classes] 的张量，包含分类对数
                 "pred_boxes": 维度为 [batch_size, num_queries, 4] 的张量，包含预测的边界框坐标
            targets: 这是一个目标列表 (len(targets) = batch_size)，其中每个目标是一个字典，包含:
                 "labels": 维度为 [num_target_boxes] 的张量 (其中 num_target_boxes 是目标中真实对象的数量)，包含类别标签
                 "boxes": 维度为 [num_target_boxes, 4] 的张量，包含目标边界框坐标
        返回:
            一个大小为 batch_size 的列表，包含元组 (index_i, index_j)，其中:
                - index_i 是选定的预测索引 (按顺序)
                - index_j 是相应的选定目标索引 (按顺序)
            对于每个批次元素，它满足:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
                
        """

        # 我们展平以批量计算成本矩阵
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # 计算分类成本

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 计算边界框之间的L1成本

        # Compute the L1 cost between boxes
        # 计算边界框之间的giou成本
            
        # Compute the giou cost betwen boxes            
        # 最终成本矩阵
        # Final cost matrix
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        # 计算每个目标中的boxes数量
        sizes = [len(v["boxes"]) for v in targets]
        # 初始化索引列表
        indices = []
        # 获取设备类型
        device = C.device
        # 遍历每个目标
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            # 获取权重矩阵
            weight_mat = c[i]
            # 获取权重矩阵中每一行的最小值的索引
            idx_i = weight_mat.min(0)[1]
            # 获取目标中的boxes数量
            idx_j = torch.arange(_size).to(device)
            # 将索引添加到索引列表中
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))