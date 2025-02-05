import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    # 获取参数字典类型
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    # 断言参数字典类型是否在指定范围内
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    # import pdb;pdb.set_trace()
   # 如果参数字典类型为'default'，则创建参数字典列表
    if param_dict_type == 'default':
        # 创建参数字典列表，第一个字典包含所有不包含'backbone'的参数，第二个字典包含所有包含'backbone'的参数，并设置学习率为args.lr_backbone
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
        # 返回参数字典列表
        return param_dicts

   # 如果参数字典类型为ddetr_in_mmdet
    if param_dict_type == 'ddetr_in_mmdet':
        # 创建参数字典列表
        param_dicts = [
            # 创建第一个参数字典，包含所有不需要调整学习率的参数
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            # 创建第二个参数字典，包含需要调整学习率的backbone参数
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            # 创建第三个参数字典，包含需要调整学习率的线性投影参数
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr_linear_proj_mult,
            }
        ]        
        # 返回参数字典列表
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]

        # print("param_dicts: {}".format(param_dicts))

    return param_dicts