# 匹配所有的**Final Answer**，替换成\n

# 找出\n</think>\n在其后面加上**Final Answer**

import json
from tqdm import tqdm

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile):
            # 解析每一行的 JSON 数据
            data = json.loads(line)
            
            # 检查是否存在 'response' 键
            if 'response' in data:
                # 替换所有的 **Final Answer** 为 \n
                if '**Final Answer**' in data['response']:
                    data['response'] = data['response'].replace('**Final Answer**', '\n')

                data['response'] = data['response'].replace('\n</think>\n', '\n</think>\n**Final Answer**')
            
            # 将修改后的数据写入输出文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# 输入文件和输出文件路径
input_file = '/data/wuxinrui/LLaMA-Factory/data/OT_long_short_formatted.jsonl'
output_file = input_file.replace('.jsonl', '_cleaned.jsonl')

# 处理文件
process_jsonl_file(input_file, output_file)