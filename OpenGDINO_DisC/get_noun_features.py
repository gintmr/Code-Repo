import os 

import os
import torch
from torch.nn.functional import cosine_similarity

noun = 'person'
folder_path = f'/data2/wuxinrui/OpenGDINO_DisC/noun_features/{noun}'


# 获取文件夹中所有的.pth文件
files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]

# 加载.pth文件并计算形状和余弦相似性
features = []
for file in files:
    file_path = os.path.join(folder_path, file)
    # 加载.pth文件
    feature = torch.load(file_path)
    shape = feature.shape
    print(f"Loaded {file} with shape: {shape}")
    features.append(feature)

# 计算两两之间的余弦相似性
num_files = len(features)
similarity_matrix = torch.zeros((num_files, num_files))

for i in range(num_files):
    for j in range(num_files):
        if i != j:
            # 计算余弦相似性
            similarity = cosine_similarity(features[i], features[j], dim=1, eps=1e-6)
            similarity_matrix[i, j] = similarity.item()
        else:
            similarity_matrix[i, j] = 1.0  # 自己与自己的相似性为1
            

# 计算每个feature中256项的平方和
squared_sums = []
for feature in features:
    # 计算256项的平方和
    squared_sum = torch.sum(feature ** 2)
    squared_sums.append(squared_sum.item())
    print(f"Squared sum of feature: {squared_sum.item()}")

    

# 打印余弦相似性矩阵
print("Cosine similarity matrix:")
print(similarity_matrix)