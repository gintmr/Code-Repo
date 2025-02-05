from safetensors import safe_open
import torch

# 定义要合并的safetensors文件路径列表
file_paths = [
    "/data05/user/DATA/QW2_5-3B-instruct/model-00001-of-00002.safetensors",
    "/data05/user/DATA/QW2_5-3B-instruct/model-00002-of-00002.safetensors"
]

# 创建一个字典来存储所有键值对
merged_tensors = {}

# 遍历每个文件并读取键值对
for path in file_paths:
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 检查键是否已经存在，如果存在则抛出错误
            if key in merged_tensors:
                raise ValueError(f"Duplicate key found: {key}. Please ensure all keys are unique across files.")
            merged_tensors[key] = tensor
            print(f"Loaded key: {key}, Tensor shape: {tensor.shape}")

# 将合并后的键值对保存为一个新的safetensors文件
output_path = "/data05/user/DATA/QW2_5-3B-instruct/model.safetensors"
torch.save(merged_tensors, output_path)

print(f"All tensors have been merged and saved to {output_path}")

with safe_open(output_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"Key: {key}, Tensor shape: {tensor.shape}")