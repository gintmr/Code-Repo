import trace
import sys

# 创建一个 tracer 对象
tracer = trace.Trace(count=False, trace=True)

# 运行你的代码
tracer.run('''
import os
import subprocess
import sys

PROMPT_TYPE = 'qwen25-math-cot'
MODEL_NAME_OR_PATH = "/data05/user/DATA/QW2_5-3B-instruct"
OUTPUT_DIR = "/data05/user/Qwen2.5-Math/eval_result"
SPLIT = "test"
NUM_TEST_SAMPLE = -1
DATA_NAME = "math"
max_tokens_list = [128, 256, 512, 768, 1024]
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
''')
