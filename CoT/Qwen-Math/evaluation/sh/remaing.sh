# 两阶段推理测试


set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2


SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math500"

# 定义 max_tokens_per_call 的取值范围
for tokens in 125 250 500 1000 2000 4000 6000 8000 10000 15000
# for tokens in 2050 4100 14500  
do
    echo "max_tokens_per_call: $tokens \n"
    export BUDGET=$tokens
    echo "export BUDGET=$tokens \n"
    TOKENIZERS_PARALLELISM=false \
    python3 -u /data05/wuxinrui/Qwen2.5-Math/evaluation/remaining_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAME} \
        --output_dir ./$modelname/$tokens \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --num_test_sample 5000\
        --use_safetensors \
        --save_outputs \
        --use_vllm \
        --overwrite \
        --max_tokens_per_call $tokens 
done