# GPU数量
GPU_NUM=4
# 配置文件
CFG="/data2/wuxinrui/OpenGDINO_DisC/config/cfg_coco.py"
# 数据集
DATASETS="/data2/wuxinrui/Datasets/COCO/annotations/captions_train2017.json"
# 输出目录
OUTPUT_DIR="/data2/wuxinrui/OpenGDINO_DisC/outputs"
# 节点数量，默认为1
NNODES=${NNODES:-1}
# 节点排名，默认为0
NODE_RANK=${NODE_RANK:-0}
# 端口号，默认为29500
PORT=${PORT:-29500}
# 主机地址，默认为127.0.0.1
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} TRAIN_DISC.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /data2/wuxinrui/OpenGDINO_DisC/weights/groundingdino_swint_ogc.pth \
        --options text_encoder_type="/data2/wuxinrui/OpenGDINO_DisC/bert-base-uncased"


