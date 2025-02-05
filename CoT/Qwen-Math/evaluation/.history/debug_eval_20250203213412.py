import os
import subprocess
import sys

# # 获取命令行参数
# PROMPT_TYPE = sys.argv[1]
# MODEL_NAME_OR_PATH = sys.argv[2]

# 获取命令行参数

PROMPT_TYPE = 'qwen25-math-cot'
MODEL_NAME_OR_PATH = "/data05/user/DATA/QW2_5-3B-instruct"

# 设置输出目录
OUTPUT_DIR = "/data05/user/Qwen2.5-Math/eval_result"
SPLIT = "test"
NUM_TEST_SAMPLE = -1
DATA_NAME = "math"

# 模型评估相关配置
max_tokens_list = [128, 256, 512, 768, 1024]

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PE_MODE"] = 'lrpe'

# 调用每个 max_tokens 设置
for max_tokens in max_tokens_list:
    os.environ["BUDGET"] = str(max_tokens)
    print(f"Running with max_tokens_per_call = {max_tokens}")
    subprocess.run([
        "python3", "-u", "math_eval.py",
        "--model_name_or_path", MODEL_NAME_OR_PATH,
        "--data_name", DATA_NAME,
        "--output_dir", OUTPUT_DIR,
        "--split", SPLIT,
        "--prompt_type", PROMPT_TYPE,
        "--num_test_sample", str(NUM_TEST_SAMPLE),
        "--seed", "0",
        "--temperature", "0",
        "--n_sampling", "1",
        "--top_p", "1",
        "--start", "0",
        "--end", "-1",
        "--use_vllm",
        "--save_outputs",
        "--overwrite",
        "--max_tokens_per_call", str(max_tokens)
    ])
