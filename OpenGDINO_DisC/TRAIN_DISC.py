# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, DisC_train_one_epoch

from groundingdino.util.utils import clean_state_dict

from models.GroundingDINO.DisCutils.utils import Adjust_Module




import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'



DisC_model = Adjust_Module()





def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default="/data2/wuxinrui/OpenGDINO_DisC/config/cfg_coco.py", type=str, required=False)
# 添加一个名为options的参数，该参数可以接受多个值，并且使用DictAction来处理这些值
    parser.add_argument('--options',
        nargs='+', ## nargs='+'表示可以接受多个值
        action=DictAction, ## 使用DictAction来处理这些值，DictAction会将这些值解析为字典
        # 帮助信息，说明该参数用于覆盖配置文件中的某些设置，以xxx=yyy的格式键值对将被合并到配置文件中
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    # parser.add_argument("--text_encoder_type", default = "/data2/wuxinrui/OpenGDINO_DisC/bert-base-uncased", type=str, required=False, help='path to datasets json')

    # dataset parameters
    # 添加一个参数，用于指定数据集的json文件路径
    parser.add_argument("--datasets", default = "/data2/wuxinrui/OpenGDINO_DisC/DisC_datasets/coco.json", type=str, required=False, help='path to datasets json')
    # 添加一个参数，用于指定是否移除困难样本
    parser.add_argument('--remove_difficult', action='store_true')
    # 添加一个参数，用于指定是否固定图像大小
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='/data2/wuxinrui/OpenGDINO_DisC/outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', default="/data2/wuxinrui/OpenGDINO_DisC/weights/groundingdino_swint_ogc.pth", help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default = False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--distributed', default=True, action='store_true')
    parser.add_argument('-gpu', default=[0, 1, 2, 3])
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args, DisC_model):
    

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    # 根据args.rank的值，让程序休眠0.02秒
    time.sleep(args.rank * 0.02)
    # 从指定的配置文件中加载配置，并将其存储在cfg变量中。
    # 如果提供了选项参数，则使用这些选项来覆盖配置文件中的设置。
    cfg = SLConfig.fromfile(args.config_file)  # 从配置文件中加载配置
    if args.options is not None:  # 如果提供了选项参数
        cfg.merge_from_list(args.options)  # 使用选项参数覆盖配置文件中的设置
        cfg.merge_from_dict(args.options)
# 如果当前进程的rank为0，则执行以下操作
    if args.rank == 0:
        # 将配置文件保存到指定路径
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        # 将参数保存为json文件
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
# 将配置文件转换为字典
    cfg_dict = cfg._cfg_dict.to_dict()
# 将参数转换为字典
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")


# 如果使用分布式训练，则将模型转换为分布式数据并行模型
    model_without_ddp = model
    if args.distributed:
    # 将模型转换为分布式数据并行模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
        
    if args.distributed:
    # 将模型转换为分布式数据并行模型
        DisC_model = torch.nn.parallel.DistributedDataParallel(DisC_model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        DisC_without_ddp = DisC_model.module
        
    # 获取模型参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 打印模型参数数量
    logger.info('number of params:'+str(n_parameters))
    # 打印模型参数信息
    logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))



    # 获取参数字典
    param_dicts = get_param_dict(args, DisC_model)
    

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)



    logger.debug("build dataset ... ...")
    # 如果不进行评估
    if not args.eval:
        # 获取训练集的数量
        num_of_dataset_train = len(dataset_meta["train"])
        # 如果训练集数量为1
        if num_of_dataset_train == 1:
            # 构建训练集
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            # 否则，导入ConcatDataset
            from torch.utils.data import ConcatDataset
            # 创建一个空列表，用于存储训练集
            dataset_train_list = []
            # 遍历训练集
            for idx in range(len(dataset_meta["train"])):
                # 构建训练集
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            # 将训练集列表合并为一个训练集
            dataset_train = ConcatDataset(dataset_train0_list)
        # 记录构建训练集完成
        logger.debug("build dataset, done.")
        # 记录训练集的数量和样本数量
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')
    
    subeset_size = len(dataset_train) // 10
    subset_indices = list(range(subeset_size))
    dataset_train = Subset(dataset_train, subset_indices)


    # 构建验证集
    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

# 如果不进行评估
    if not args.eval:
        # 创建训练集的batch_sampler
        # 创建一个批采样器，用于训练数据集
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True) #! 取样，将数据分批
        # 创建训练集的数据加载器
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)

# 创建验证集的数据加载器
    data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

# 如果使用OneCycleLR学习率调度器
    if args.onecyclelr:
        # 创建OneCycleLR学习率调度器，设置最大学习率、每个epoch的步数、总epoch数和起始百分比
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
# 如果使用MultiStepLR学习率调度器
    elif args.multi_step_lr:
        # 创建MultiStepLR学习率调度器，设置里程碑列表
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
# 否则使用StepLR学习率调度器
    else:
        # 创建StepLR学习率调度器，设置学习率下降的步长
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val)

    # 如果有冻结权重，则加载冻结权重
    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    # 获取输出目录
    output_dir = Path(args.output_dir)
    # 如果输出目录下存在checkpoint.pth文件，则将args.resume设置为该文件路径
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    # 如果有恢复训练的参数，则加载恢复训练的模型
    if args.resume:
        # 如果恢复训练的参数是URL，则从URL加载模型
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        # 否则从本地文件加载模型
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 加载模型参数s
        DisC_model.load_state_dict(clean_state_dict(checkpoint['DisC_model']),strict=False)


        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 如果不恢复训练并且有预训练模型路径
    if (not args.resume) and args.pretrain_model_path:
        # 加载预训练模型
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        # 导入OrderedDict
        from collections import OrderedDict
        # 如果有需要忽略的参数，则将其存入_ignorekeywordlist
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        # 初始化ignorelist
        ignorelist = []

        # 检查是否需要保留参数
        def check_keep(keyname, ignorekeywordlist):
            # 遍历需要忽略的参数
            for keyword in ignorekeywordlist:
                # 如果参数名中包含需要忽略的参数，则将其存入ignorelist，并返回False
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            # 如果参数名中不包含需要忽略的参数，则返回True
            return True

        # 打印忽略的键
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        # 清理状态字典，只保留需要保留的键
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        # 加载状态字典，不严格匹配
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # 打印加载结果
        logger.info(str(_load_output))

 
    
    # 如果args.eval为真，则执行以下代码
    if args.eval:
        # 设置环境变量EVAL_FLAG为TRUE
        os.environ['EVAL_FLAG'] = 'TRUE'
        # 调用evaluate函数，传入model、criterion、postprocessors、data_loader_val、base_ds、device、args.output_dir、wo_class_error、args参数
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        # 如果args.output_dir为真，则调用utils.save_on_master函数，传入coco_evaluator.coco_eval["bbox"].eval和output_dir / "eval.pth"参数
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        # 将test_stats中的键值对转换为log_stats字典
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        # 如果args.output_dir为真且utils.is_main_process()为真，则调用with语句，将log_stats写入output_dir / "log.txt"文件中
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # 返回
        return
    
 
    
    print("Start training")
# 记录开始时间
    start_time = time.time()
# 创建一个BestMetricHolder对象，use_ema参数为False
    best_map_holder = BestMetricHolder(use_ema=False)

    min_loss = 0


    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        # 如果是分布式训练，设置当前epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # 训练一个epoch
        train_stats, DisC_model_ , LOSS = DisC_train_one_epoch(
            model, criterion, data_loader_train, optimizer, DisC_model, args.box_threshold, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None))
        # 如果有输出目录
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        # 如果不是onecyclelr，更新学习率
        if not args.onecyclelr:
            lr_scheduler.step()
        # 如果有输出目录
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'DisC_model': DisC_model_.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    
                    'epoch': epoch,
                    'args': args,
                }

                utils.save_on_master(weights, checkpoint_path)
                
        # # eval
        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
        #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        # )
        # # 评估模型，获取测试统计信息和COCO评估器
        # map_regular = test_stats['coco_eval_bbox'][0]
        # # 获取测试统计信息中的COCO评估框的平均精度
        # _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        
        min_loss = min(min_loss, LOSS)
        
        _isbest = False
        
        if LOSS == min_loss:
            _isbest = True
        
        
        # 更新最佳平均精度，并返回是否为最佳
        # if _isbest:
        #     checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
        #     # 如果是最佳，则保存模型
        #     utils.save_on_master({
        #         'model': DisC_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #     }, checkpoint_path)
        
        
        
        
        
        
        
############       
        
        
        # # 保存模型
        # log_stats = {
        #     **{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        # }
        
############
        

        
        
        
        
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()}
        }

        
        # 更新日志统计信息中的当前时间
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        # 计算当前epoch的时间
        epoch_time = time.time() - epoch_start_time
        # 将epoch时间转换为字符串
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        # 将epoch时间添加到日志统计信息中
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")




##############
            # # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)
                        
###############                        
                        
                        
                        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, DisC_model)
