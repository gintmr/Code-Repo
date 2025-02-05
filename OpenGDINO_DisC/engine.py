# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import numpy as np

from util.utils import to_device
import torch
import torch.nn as nn
import torch.nn.functional as F

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'


def DisC_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, DisC_model: torch.nn.Module,
                    box_threshold: int,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    """
    训练一个epoch的函数。

    参数:
    - model: 要训练的模型。
    - criterion: 损失函数。
    - data_loader: 数据加载器。
    - optimizer: 优化器。
    - device: 设备（CPU或GPU）。
    - epoch: 当前的epoch数。
    - max_norm: 梯度裁剪的最大范数。
    - wo_class_error: 是否不计算分类错误。
    - lr_scheduler: 学习率调度器。
    - args: 命令行参数。
    - logger: 日志记录器。
    """
    # 创建一个GradScaler对象，用于自动混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 将模型和损失函数设置为训练模式
    # model.train()  # 将模型设置为训练模式
    criterion.train()  # 将损失函数设置为训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")  # 创建一个MetricLogger对象，用于记录训练过程中的各种指标
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加一个用于记录学习率的指标
    if not wo_class_error:  # 如果需要计算分类错误
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))  # 添加一个用于记录分类错误的指标
    header = 'Epoch: [{}]'.format(epoch)  # 设置日志的头部信息，显示当前的epoch数
    print_freq = 10  # 设置打印频率，每10次迭代打印一次日志

    _cnt = 0  # 初始化计数器，用于记录当前的迭代次数


    LOSS = 0
    

    # 遍历数据加载器中的样本和目标
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)  # 将样本数据移动到指定设备
        captions = [t["caption"] for t in targets]  # 从目标中提取caption
        cap_list = [t["cap_list"] for t in targets]  # 从目标中提取cap_list
        
        # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]  # 将目标中的张量移动到指定设备
        
        
        # model = nn.DataParallel(model.to(device), device_ids=[0, 1, 2, 3], output_device=0)  # 将模型包装为分布式数据并行模型
        
        model.eval()
        
        with torch.cuda.amp.autocast(enabled=args.amp):  # 启用自动混合精度
            model.to(device)
            for para in model.parameters():
                para.requires_grad = False
                para.to(device=device)
                
            for param in DisC_model.parameters():
                param.requires_grad = True
                param.to(device=device)
        
        
            model.eval()  # 冻结 model，不需要计算梯度
            DisC_model.train()  # 训练 DisC_model
                
            with torch.no_grad():
                outputs, filt_outputs, pred_phrases, text_dicts, boxes = model(samples, captions=captions)  # 前向传播，获取模型输出
            ###########! filt_output是一个列表，分别将batchsize中的每一个单独图像中的所有检测框标出(根据每张图的数量来列出)
            #####! pred_phrases是一个列表，分别将batchsize中的每一个单独图像中的所有检测框的预测短语标出（全部列出）
            filt_output_reshaped = np.concatenate([output.to('cpu').detach().numpy() for output in filt_outputs], axis=0)           
            # filt_output_reshaped = filt_output.view(filt_output.size(0)*filt_output.size(1), filt_output.size(2))
            LOSS_dict = {}
            LOSS_dict['base_sim'] = 0
            LOSS_dict['neighbors_sim'] = 0
            LOSS_dict['kl_divergence'] = 0
            filt_output_reshaped = torch.from_numpy(filt_output_reshaped)
    
            # 将张量移动到指定的设备
            filt_output_reshaped = filt_output_reshaped.to('cuda:0')
            for feature, word in zip(filt_output_reshaped, pred_phrases):
                DisC_model.cuda()
                feature = DisC_model(feature)
                # print(feature.shape, word)
                loss_dict = criterion(feature, word)
                LOSS_dict['base_sim'] += loss_dict['base_sim']
                LOSS_dict['neighbors_sim'] += loss_dict['neighbors_sim']
                LOSS_dict['kl_divergence'] += loss_dict['kl_divergence']
            
            del outputs, filt_outputs, pred_phrases, filt_output_reshaped
            ############!
            # loss_dict = criterion(outputs, targets, cap_list, captions)  # 计算损失字典

            # weight_dict = criterion.weight_dict  # 获取损失函数的权重字典

            # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # 计算总损失
        losses = custom_loss(LOSS_dict)  # 计算总损失
        
        
        # # 减少所有GPU上的损失以进行日志记录
        # loss_dict_reduced = utils.reduce_dict(loss_dict)  # 减少所有GPU上的损失字典
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v  # 创建一个未缩放的损失字典
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]  # 创建一个缩放的损失字典
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())  # 计算缩放后的总损失

        # loss_value = losses_reduced_scaled.item()  # 获取缩放后的总损失值
        # # 判断loss_value是否为有限值
        # if not math.isfinite(loss_value):
        #     # 如果loss_value不是有限值，则打印提示信息，并退出程序
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)
        # 判断loss_value是否为有限值
        if not math.isfinite(losses):
            # 如果loss_value不是有限值，则打印提示信息，并退出程序
            print("Loss is {}, stopping training".format(losses))
            sys.exit(1)

        # amp反向传播函数
        # 如果使用了amp（自动混合精度）训练
        if args.amp:
            # 梯度清零
            optimizer.zero_grad()
            # 将损失值缩放
            scaler.scale(losses).backward()
            # 如果设置了梯度裁剪
            if max_norm > 0:
                # 取消缩放
                scaler.unscale_(optimizer)
                # 对模型参数进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # 更新优化器
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()
        else:
            # 原始反向传播函数
            optimizer.zero_grad()  # 梯度清零
            losses.backward()  # 反向传播
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 梯度裁剪
            optimizer.step()  # 更新参数

        if args.onecyclelr:
            lr_scheduler.step()

# 更新metric_logger，包括loss、loss_dict_reduced_scaled、loss_dict_reduced_unscaled
        metric_logger.update(loss=losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        # 如果debug参数为True，并且_cnt能被15整除，则打印BREAK!5次，并跳出循环
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

        LOSS += losses


    # 如果criterion对象有loss_weight_decay属性，则调用该属性，并传入epoch参数
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    # 如果criterion对象有tuning_matching属性，则调用该属性
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # 从所有进程中收集统计数据
    # 在进程之间同步metric_logger
    metric_logger.synchronize_between_processes()
    # 打印平均统计信息
    print("Averaged stats:", metric_logger)
    
    
    
    # 计算并打印每个 epoch 的总损失
    total_loss = LOSS
    print(f"Epoch {epoch} Total Loss: {LOSS}")
    
    
    
    # 将metric_logger中的meter转换为字典，并计算全局平均值
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # 如果criterion有loss_weight_decay属性，则将weight_dict中的值更新到resstat中
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    # 返回resstat
    return resstat, DisC_model, total_loss



def custom_loss(loss_dict, base_sim_weight=1.0, neighbors_sim_weight=0.125, kl_divergence_weight=1.0):
    """
    自定义损失函数，要求 base_sim 增大，neighbors_sim 减少，kl_divergence 也增大。

    参数:
    - loss_dict: 包含 base_sim, neighbors_sim, kl_divergence 的字典
    - base_sim_weight: base_sim 的权重
    - neighbors_sim_weight: neighbors_sim 的权重
    - kl_divergence_weight: kl_divergence 的权重

    返回:
    - 总的损失值
    """
    base_sim = loss_dict['base_sim']
    neighbors_sim = loss_dict['neighbors_sim']
    kl_divergence = loss_dict['kl_divergence']

    # 计算损失
    loss = base_sim_weight * base_sim - neighbors_sim_weight * neighbors_sim + kl_divergence_weight * kl_divergence
    
    loss = torch.tensor(loss, requires_grad=True)

    return loss






@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs, _, _ = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator
