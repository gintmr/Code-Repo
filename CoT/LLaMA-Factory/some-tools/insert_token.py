## 在数据中穿插remaining token(输入未处理的数据,自动从答案的开头往后连续添加)
## 同时，insert操作向上以50为跨度取整

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 加载模型

tokenizer = AutoTokenizer.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247", trust_remote_code=True)

# data_path = "/data/wuxinrui/LLaMA-Factory/data/OT_long_short_formatted_cleaned.jsonl"
data_path = "/data/wuxinrui/LLaMA-Factory/data/OT_6_modes_cleaned_inserted.jsonl"

bins = [409600, 204800, 102400, 51200, 25600, 12800, 6400, 3200, 1600, 800, 400, 200, 100, 50, 0]

bins_tokens = [
    "\n<remaining>409600</remaining>\n",
    "\n<remaining>204800</remaining>\n",
    "\n<remaining>102400</remaining>\n",
    "\n<remaining>51200</remaining>\n",
    "\n<remaining>25600</remaining>\n",
    "\n<remaining>12800</remaining>\n",
    "\n<remaining>6400</remaining>\n",
    "\n<remaining>3200</remaining>\n",
    "\n<remaining>1600</remaining>\n",
    "\n<remaining>800</remaining>\n",
    "\n<remaining>400</remaining>\n",
    "\n<remaining>200</remaining>\n",
    "\n<remaining>100</remaining>\n",
    "\n<remaining>50</remaining>\n",
    "",
]


def split_array_by_bins(input_array, bins):
    # 计算新输入数组的长度
    array_length = len(input_array)
    
    divide_50 = array_length // 50
    array_length = (divide_50+1) * 50
    
    # 初始化结果列表
    result = []
    
    # 从分档数组的最后一个元素开始向前遍历
    i = 0
    indice = 0
    while array_length < bins[i]:
            i += 1
            indice += 1
    i -= 1
    while i < len(bins) - 1:

        start_index = max(array_length - bins[i], 0)
        end_index = array_length - bins[i + 1]
        
        result.append(input_array[start_index:end_index])
        # print(f"{i}_th: {start_index}:{end_index}")
        i += 1
        
    # print(indice)
    
    return result, indice, array_length


def split_string(input_string):
    # 要匹配的字符串
    match_string = "\n</think>\n"
    
    # 找到匹配字符串的起始位置
    start_index = input_string.find(match_string)
    
    if start_index == -1:
        print("匹配的字符串未找到")
        return None, None
    
    # 获取匹配字符串之前的字符串
    before_string = input_string[:start_index]
    
    # 获取匹配字符串之后的所有字符串
    after_string = input_string[start_index:]
    
    return before_string, after_string

def count_down(sub_cot, indice, up_50_len):
    inserted_cot = f"\n<remaining>{up_50_len}</remaining>\n"
    for i in range(len(sub_cot)):
        inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i]) + bins_tokens[i+indice]
   
    return inserted_cot


def count_down_RL(sub_cot, indice, up_50_len):
    inserted_cot = f"<think>\n"
    for i in range(len(sub_cot)):
        inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i]) + bins_tokens[i+indice]
   
    return inserted_cot


def insert_token(data_path):
    inserted_data_path = data_path.replace(".jsonl", "_inserted.jsonl")
    if os.path.exists(inserted_data_path):
        os.remove(inserted_data_path)
    with open(data_path, "r") as f:
        datas = [json.loads(line) for line in f]
        inserted_datas  ={}
        for data in tqdm(datas, desc="inserting token with origin format"):
            prompt = data["prompt"]
            response = data["response"]
            
            cot, answer = split_string(response)
            tokenized_cot = tokenizer(cot, return_tensors="pt").input_ids[0]
            chunk_size = 100
            length_of_tokenized_cot = len(tokenized_cot)
            sub_cot, indice, up_50_len = split_array_by_bins(tokenized_cot, bins)
            inserted_cot = count_down(sub_cot, indice, up_50_len)  
            response = inserted_cot + answer
            inserted_data = {
                "prompt": prompt,
                "response": response
            }
            # print(inserted_data)
            with open(inserted_data_path, "a") as f:
                f.write(json.dumps(inserted_data) + "\n")

def insert_token_RL(data_path):
    inserted_data_path = data_path.replace(".jsonl", "_inserted_RL.jsonl")
    if os.path.exists(inserted_data_path):
        os.remove(inserted_data_path)
    with open(data_path, "r") as f:
        datas = [json.loads(line) for line in f]
        inserted_datas  ={}
        for data in tqdm(datas, desc="inserting token with RL format"):
            prompt = data["prompt"]
            response = data["response"]
            
            cot, answer = split_string(response)
            tokenized_cot = tokenizer(cot, return_tensors="pt").input_ids[0]
            chunk_size = 100
            length_of_tokenized_cot = len(tokenized_cot)
            sub_cot, indice, up_50_len = split_array_by_bins(tokenized_cot, bins)
            inserted_cot = count_down_RL(sub_cot, indice, up_50_len)  
            response = inserted_cot + answer
            inserted_data = {
                "prompt": prompt + f"\n<remaining>{up_50_len}</remaining>\n",
                "response": response
            }
            # print(inserted_data)
            with open(inserted_data_path, "a") as f:
                f.write(json.dumps(inserted_data) + "\n")
                

insert_token_RL(data_path=data_path)