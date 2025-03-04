## 查看数据前10条

import json
import random
import os
def extract_first_ten_lines(input_file_path, output_file_path):
    # 打开输入文件和输出文件
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line_number, line in enumerate(input_file):
            if line_number < 50:  # 只处理前10行
                try:
                    data = json.loads(line)  # 尝试将行内容解析为JSON
                    json_line = json.dumps(data)  # 将JSON对象转换回字符串
                    output_file.write(json_line + '\n')  # 写入输出文件，并添加换行符
                except json.JSONDecodeError as e:
                    print(f"解析错误在第 {line_number + 1} 行: {e}")
            else:
                break  # 已经处理了10行，可以停止读取


def extract_last_fifty_lines(input_file_path, output_file_path):
    # 读取整个文件内容到一个列表中
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    # 确保文件至少有50行
    if len(lines) < 50:
        print("文件中的行数少于50行。")
        return

    # 提取最后50行
    last_fifty_lines = lines[-50:]

    # 写入输出文件
    with open(output_file_path, 'w') as output_file:
        for line in last_fifty_lines:
            try:
                data = json.loads(line)  # 尝试将行内容解析为JSON
                json_line = json.dumps(data)  # 将JSON对象转换回字符串
                output_file.write(json_line + '\n')  # 写入输出文件，并添加换行符
            except json.JSONDecodeError as e:
                print(f"解析错误在第 {len(lines) - len(last_fifty_lines) + 1} 行: {e}")

def random_extract_fifty_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
        if len(lines) < 50:
            print("文件中的行数少于50行。")
            return
        # 随机选择50行
        selected_lines = random.sample(lines, 50)
        # 写入输出文件
    with open (output_file_path, 'w') as output_file:
        for line in selected_lines:
            try:
                data = json.loads(line)  # 尝试将行内容解析为JSON
                json_line = json.dumps(data)  # 将JSON对象转换回字符串
                output_file.write(json_line + '\n')
            except json.JSONDecodeError as e:
                print(f"解析错误在第 {len(lines) - len(selected_lines) + 1} 行: {e}")
                


# 输入文件路径
input_file_path = "/data/wuxinrui/LLaMA-Factory/data/OT_long_short_formatted_cleaned_inserted_RL.jsonl"
# 输出文件路径


output_file_path = input_file_path.replace("LLaMA-Factory/data", "LLaMA-Factory/data_sample_10")
if os.path.exists(output_file_path):
    os.remove(output_file_path)
# 调用函数
# extract_first_ten_lines(input_file_path, output_file_path)

# extract_last_fifty_lines(input_file_path, output_file_path)

random_extract_fifty_lines( input_file_path, output_file_path)



