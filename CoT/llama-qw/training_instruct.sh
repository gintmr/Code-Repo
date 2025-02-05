#!/bin/bash

# 设置日志文件路径
LOG_FILE="train_log.txt"


# 函数：记录命令执行
log_command() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Executing: $1" | tee -a "$LOG_FILE"
}

# 函数：记录命令执行结果
log_result() {
    if [ $? -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Success: $1" | tee -a "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed: $1" | tee -a "$LOG_FILE"
        exit 1
    fi
}


# 开始日志记录
echo "Training log started at $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"

# 设置CUDA_VISIBLE_DEVICES
log_command "export CUDA_VISIBLE_DEVICES=1,3,5,7"
export CUDA_VISIBLE_DEVICES=1,3,5,7
log_result "export CUDA_VISIBLE_DEVICES=1,3,5,7"

# 设置PE_MODE
log_command "export PE_MODE=ldpe"
export PE_MODE=ldpe
log_result "export PE_MODE=ldpe"


export PE_MODE=ldpe


log_command "train ldpe"


llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data05/wuxinrui/DATA/QW2_5-3B-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset MetaMathQA-395K \
    --cutoff_len 512 \
    --learning_rate 5e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-3B-Instruct/lora/ldpe \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

log_result "llamafactory-cli train"


log_command "merge ldpe"


llamafactory-cli export /data05/wuxinrui/LLaMA-Factory/merge_ldpe.yaml


log_result "llamafactory-cli export"
    # --include_num_input_tokens_seen True \



log_command "export PE_MODE=lrpe"
export PE_MODE=lrpe
log_result "export PE_MODE=lrpe"



log_command "train lrpe"

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data05/wuxinrui/DATA/QW2_5-3B-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset MetaMathQA-395K \
    --cutoff_len 512 \
    --learning_rate 5e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-3B-Instruct/lora/lrpe \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

log_result "llamafactory-cli train"


log_command "merge lrpe"

llamafactory-cli export /data05/wuxinrui/LLaMA-Factory/merge_lrpe.yaml

log_result "llamafactory-cli export"



# 设置PE_MODE
log_command "export PE_MODE=reverse"
export PE_MODE=reverse
log_result "export PE_MODE=reverse"


log_command "train reverse"

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data05/wuxinrui/DATA/QW2_5-3B-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset MetaMathQA-395K \
    --cutoff_len 512 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-3B-Instruct/lora/reverse \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

log_result "llamafactory-cli train"



log_command "merge reverse"

llamafactory-cli export /data05/wuxinrui/LLaMA-Factory/merge_reverse.yaml

log_result "merged"




export PE_MODE=hybrid


log_command "train hybrid"

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data05/wuxinrui/DATA/QW2_5-3B-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset MetaMathQA-395K \
    --cutoff_len 512 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-3B-Instruct/lora/hybrid \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all


log_result "llamafactory-cli train"


log_command "merge hybrid"

llamafactory-cli export /data05/wuxinrui/LLaMA-Factory/merge_hybrid.yaml

log_result "merged"

