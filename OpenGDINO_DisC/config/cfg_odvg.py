# 数据增强尺度
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
# 数据增强最大尺寸
data_aug_max_size = 1333
# 数据增强尺度2调整大小
data_aug_scales2_resize = [400, 500, 600]
# 数据增强尺度2裁剪
data_aug_scales2_crop = [384, 600]
# 数据增强尺度重叠
data_aug_scale_overlap = None
# 批量大小
batch_size = 4
# 模型名称
modelname = 'groundingdino'
# 主干网络
backbone = 'swin_T_224_1k'
# 位置嵌入
position_embedding = 'sine'
# 位置嵌入温度H
pe_temperatureH = 20
# 位置嵌入温度W
pe_temperatureW = 20
# 返回中间索引
return_interm_indices = [1, 2, 3]
# 编码器层数
enc_layers = 6
# 解码器层数
dec_layers = 6
# 预规范化
pre_norm = False
# 前馈网络维度
dim_feedforward = 2048
# 隐藏维度
hidden_dim = 256
# Dropout
dropout = 0.0
# 多头注意力
nheads = 8
# 查询数量
num_queries = 900
# 查询维度
query_dim = 4
# 模式数量
num_patterns = 0
# 特征级别数量
num_feature_levels = 4
# 编码器点数量
enc_n_points = 4
# 解码器点数量
dec_n_points = 4
# 两阶段类型
two_stage_type = 'standard'
# 两阶段边界框嵌入共享
two_stage_bbox_embed_share = False
# 两阶段类别嵌入共享
two_stage_class_embed_share = False
# 变换器激活函数
transformer_activation = 'relu'
# 解码器预测边界框嵌入共享
dec_pred_bbox_embed_share = True
# dn边界框噪声尺度
dn_box_noise_scale = 1.0
# dn标签噪声比率
dn_label_noise_ratio = 0.5
# dn标签系数
dn_label_coef = 1.0
# dn边界框系数
dn_bbox_coef = 1.0
# 初始化目标
embed_init_tgt = True
# dn标签簿大小
dn_labelbook_size = 91
# 最大文本长度
max_text_len = 256
# 文本编码器类型
text_encoder_type = "bert-base-uncased"
# 使用文本增强器
use_text_enhancer = True
# 使用融合层
use_fusion_layer = True
# 使用检查点
use_checkpoint = True
# 使用变换器检查点
use_transformer_ckpt = True
# 使用文本交叉注意力
use_text_cross_attention = True
# 文本Dropout
text_dropout = 0.0
# 融合Dropout
fusion_dropout = 0.0
# 融合DropPath
fusion_droppath = 0.1
# 子句呈现
sub_sentence_present = True
# 最大标签数
max_labels = 50                               # pos + neg
# 基础学习率
lr = 0.0001                                   # base learning rate
# 仅用于gdino主干的关键词
backbone_freeze_keywords = None               # only for gdino backbone
# 整个模型的关键词，例如['backbone.0', 'bert']用于冻结视觉编码器和文本编码器
freeze_keywords = ['bert']                    # for whole model, e.g. ['backbone.0', 'bert'] for freeze visual encoder and text encoder
# 特定学习率
lr_backbone = 1e-05                           # specific learning rate
# 特定学习率名称
lr_backbone_names = ['backbone.0', 'bert']
# 线性投影乘数
lr_linear_proj_mult = 1e-05
# 线性投影名称
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']
# 权重衰减
weight_decay = 0.0001
# 参数字典类型
param_dict_type = 'ddetr_in_mmdet'
# ddetr学习率参数
ddetr_lr_param = False
# 训练轮数
epochs = 15
# 学习率下降
lr_drop = 4
# 保存检查点间隔
save_checkpoint_interval = 1
# 裁剪最大范数
clip_max_norm = 0.1
# 一周期学习率
onecyclelr = False
# 多步学习率
multi_step_lr = False
# 学习率下降列表
lr_drop_list = [4, 8]
# 冻结权重
frozen_weights = None
# 膨胀
dilation = False
# pdetr3边界框嵌入差异每层
pdetr3_bbox_embed_diff_each_layer = False
# pdetr3_refHW
pdetr3_refHW = -1
# 随机参考点XY
random_refpoints_xy = False
# 固定参考点HW
fix_refpoints_hw = -1
# dabdetr_yolo_like_anchor_update
dabdetr_yolo_like_anchor_update = False
# dabdetr可变形编码器
dabdetr_deformable_encoder = False
# dabdetr可变形解码器
dabdetr_deformable_decoder = False
# 使用可变形边界框注意力
use_deformable_box_attn = False
# 边界框注意力类型
box_attn_type = 'roi_align'
# 解码器层数
dec_layer_number = None
# 解码器层噪声
decoder_layer_noise = False
# dln_xy噪声
dln_xy_noise = 0.2
# dln_hw噪声
dln_hw_noise = 0.2
# 添加通道注意力
add_channel_attention = False
# 添加位置值
add_pos_value = False
# 两阶段模式嵌入
two_stage_pat_embed = 0
# 两阶段添加查询数量
two_stage_add_query_num = 0
# 两阶段学习WH
two_stage_learn_wh = False
# 两阶段默认HW
two_stage_default_hw = 0.05
# 两阶段保留所有令牌
two_stage_keep_all_tokens = False
# 选择数量
num_select = 300
# 批量归一化类型
batch_norm_type = 'FrozenBatchNorm2d'
# 掩码
masks = False
# 辅助损失
aux_loss = True
# 设置成本类别
set_cost_class = 1.0
# 设置成本边界框
set_cost_bbox = 5.0
# 设置成本giou
set_cost_giou = 2.0
# 类别损失系数
cls_loss_coef = 2.0
# 边界框损失系数
bbox_loss_coef = 5.0
# giou损失系数
giou_loss_coef = 2.0
# 编码器损失系数
enc_loss_coef = 1.0
# 中间损失系数
interm_loss_coef = 1.0
# 无中间边界框损失
no_interm_box_loss = False
# 掩码损失系数
mask_loss_coef = 1.0
# dice损失系数
dice_loss_coef = 1.0
# focal_alpha
focal_alpha = 0.25
# focal_gamma
focal_gamma = 2.0
# 解码器自注意力类型
decoder_sa_type = 'sa'
# 匹配器类型
matcher_type = 'HungarianMatcher'
# 解码器模块序列
decoder_module_seq = ['sa', 'ca', 'ffn']
# nms_iou阈值
nms_iou_threshold = -1
# 解码器预测类别嵌入共享
dec_pred_class_embed_share = True


match_unstable_error = True
use_ema = False
# 使用指数移动平均
ema_decay = 0.9997
# ema衰减
ema_epoch = 0
# ema轮数
use_detached_boxes_dec_out = False
# 使用分离的边界框解码器输出
use_coco_eval = True
# 使用coco评估
dn_scalar = 100# dn标量
