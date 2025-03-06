# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="deepseek3"
# PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="2"



# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
echo MODEL_NAME: $MODEL_NAME
export PE_MODE=default
export position=ori
export tip=remaining 
export stage=1
export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH



# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


# MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
# PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
# MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
# echo MODEL_NAME: $MODEL_NAME
# export PE_MODE=default
# export position=ori
# export tip=remaining 
# export stage=2
# export mode=TIP-$tip-STAGE-$stage
# export model=MODEL-$MODEL_NAME
# export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
# bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH



bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
echo MODEL_NAME: $MODEL_NAME
export PE_MODE=default
export position=ori
export tip=Ahead 
export stage=1
export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH



# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
# bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


# MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
# PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
# MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
# echo MODEL_NAME: $MODEL_NAME
# export PE_MODE=default
# export position=ori
# export tip=Ahead 
# export stage=2
# export mode=TIP-$tip-STAGE-$stage
# export model=MODEL-$MODEL_NAME
# export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
# bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
echo MODEL_NAME: $MODEL_NAME
export PE_MODE=default
export position=ori
export tip=prompt-based 
export stage=1
export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/start_positions.pt
bash delete_file.sh /data05/wuxinrui/Qwen2.5-Math/evaluation/early_positions.pt


MODEL_NAME_OR_PATH='/data05/wuxinrui/LLaMA-Factory/long_short_inserted_RL/models'
PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
echo MODEL_NAME: $MODEL_NAME
export PE_MODE=default
export position=ori
export tip=prompt-based 
export stage=2
export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage
bash /data05/wuxinrui/Qwen2.5-Math/evaluation/sh/remaining.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH