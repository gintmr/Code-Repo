# 设置批处理大小
batch_size = 1
# 模型名称
modelname = "groundingdino"
# 主干网络
backbone = "swin_T_224_1k" # backbone
# 位置编码
position_embedding = "sine"
# 高度位置编码温度
pe_temperatureH = 20
# 宽度位置编码温度
pe_temperatureW = 20
# 返回的中间索引
return_interm_indices = [1, 2, 3]
# 主干网络冻结关键词
backbone_freeze_keywords = None
# 编码层
enc_layers = 6
# 解码层
dec_layers = 6
# 预规范化
pre_norm = False
# 前馈网络维度
dim_feedforward = 2048
# 隐藏维度
hidden_dim = 256
# Dropout率
dropout = 0.0
# 多头注意力数
nheads = 8
# 查询数
num_queries = 900
# 查询维度
query_dim = 4
# 模式数
num_patterns = 0
# 特征级别数
num_feature_levels = 4
# 编码点数
enc_n_points = 4
# 解码点数
dec_n_points = 4
# 两阶段类型
two_stage_type = "standard"
# 两阶段边界框嵌入共享
two_stage_bbox_embed_share = False
# 两阶段类别嵌入共享
two_stage_class_embed_share = False
# 变换器激活函数
transformer_activation = "relu"
# 解码预测边界框嵌入共享
dec_pred_bbox_embed_share = True
# dn边界框噪声比例
dn_box_noise_scale = 1.0
# dn标签噪声比例
dn_label_noise_ratio = 0.5
# dn标签系数
dn_label_coef = 1.0
# dn边界框系数
dn_bbox_coef = 1.0
# 初始化目标嵌入
embed_init_tgt = True
# dn标签簿大小
dn_labelbook_size = 2000
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
# 文本Dropout率
text_dropout = 0.0
# 融合Dropout率
fusion_dropout = 0.0
# 融合DropPath率
fusion_droppath = 0.1
# 子句存在
sub_sentence_present = True
