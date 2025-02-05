# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="4"
MODEL_NAME_OR_PATH="/data05/user/DATA/QW2_5-3B-instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH