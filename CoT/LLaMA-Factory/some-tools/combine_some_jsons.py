import os
import json
import argparse

def merge_jsonl_files(input_files, output_file):
    """
    合并多个 JSONL 文件到一个文件中。
    
    :param input_files: 输入的 JSONL 文件列表
    :param output_file: 输出的合并后的 JSONL 文件
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in input_files:
            if not os.path.exists(file):
                print(f"警告：文件 {file} 不存在，跳过。")
                continue
            
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 确保每行是一个有效的 JSON 对象
                    try:
                        json.loads(line.strip())
                        outfile.write(line)
                    except json.JSONDecodeError as e:
                        print(f"警告：文件 {file} 中的某行不是有效的 JSON 格式。跳过该行。")
    print(f"合并完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个 JSONL 文件")
    parser.add_argument("input_files", nargs="+", help="输入的 JSONL 文件列表")
    parser.add_argument("-o", "--output", required=True, help="输出的 JSONL 文件路径")
    
    args = parser.parse_args()
    merge_jsonl_files(args.input_files, args.output)