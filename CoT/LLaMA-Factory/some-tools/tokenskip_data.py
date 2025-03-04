import json 
import transformers
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

# model = AutoModelForCausalLM.from_pretrained('/data03/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/14dd1130311655b43c3ce41dd505f70f6ca89845')
tokenizer = AutoTokenizer.from_pretrained('/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247')


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
    # after_string = input_string[start_index + len(match_string):]
    
    return before_string, after_string

dataset_path = '/data/wuxinrui/LLaMA-Factory/data/OT_long_short.jsonl' ## 训练用的数据，两个key，prompt 和 response



def add_Q_tail(Q, A, tokens_len, data):
    QA = {}
    QA['prompt'] = Q + f"You need to complete the thinking within {tokens_len} tokens"
    QA['response'] = A
    data.append(QA)
    
    
    
def add_A_head(Q, A, tokens_len, data):
    QA = {} 
    if tokens_len <= 1000:
        tokens_len = (tokens_len//10 + 1) * 10
    elif tokens_len >= 1000 and tokens_len <= 10000:
        tokens_len = (tokens_len//50 + 1) * 50
    elif tokens_len >= 10000 and tokens_len <= 100000:
        tokens_len = (tokens_len//100 + 1) * 100
    QA['prompt'] = Q
    QA['response'] = f"\n<len>{tokens_len}<\len>\n" + A
    data.append(QA)


# OT_long_short_token_add_Qtail = []
OT_long_short_token_add_Ahead = []
with open(dataset_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        prompt = data['prompt']
        response = data['response']
        cot, answer = split_string(response)
        # print(f"cot = {cot}")
        # print(f"answer = {answer}")
        tokenized_cot = tokenizer(response, return_tensors="pt")
        tokens_len = len(tokenized_cot.input_ids[0])
        # add_Q_tail(prompt, response, tokens_len, OT_long_short_token_add_Qtail)
        add_A_head(prompt, response, tokens_len, OT_long_short_token_add_Ahead)
        # print(f"length of tokenized_cot = {len(tokenized_cot.input_ids[0])}")
        
# with open('/data05/wuxinrui/LLaMA-Factory/data/OT_long_short_token_add_Qtail.jsonl', 'w') as f:
#     for item in OT_long_short_token_add_Qtail:
#         f.write(json.dumps(item) + '\n')
with open('/data/wuxinrui/LLaMA-Factory/data/OT_long_short_Ahead.jsonl', 'w') as f:
    for item in OT_long_short_token_add_Ahead:
        f.write(json.dumps(item) + '\n')
