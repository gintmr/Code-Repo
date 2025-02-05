# 设置脚本执行时遇到错误立即退出
set -ex

# 获取第一个参数，表示提示类型
PROMPT_TYPE=$1
# 获取第二个参数，表示模型名称或路径
MODEL_NAME_OR_PATH=$2
# 设置输出目录为模型名称或路径下的math_eval文件夹
OUTPUT_DIR=/data05/user/Qwen2.5-Math/eval_result

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"


DATA_NAME="math"

for max_tokens in 128 256 512 768 1024
do
    export BUDGET=$max_tokens
    echo "budget = $BUDGET"
    echo "runing with max_tokens_per_call = $max_tokens"
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --max_tokens_per_call $max_tokens\

done

# # English multiple-choice datasets
# DATA_NAME="aqua,sat_math,mmlu_stem"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --num_shots 5

# # Chinese gaokao collections
# DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot

# # Chinese other datasets
# DATA_NAME="cmath,cn_middle_school"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot


# # English competition datasets
# DATA_NAME="aime24,amc23"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
