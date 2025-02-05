import torch
import torch.nn.functional as F
import torch.nn as nn

def vis2text_crossentropy(text_features, visual_feature): 
    ## input text_dict["encoded_text"]([bs, num_tokens, f_dim])
    '''
    鉴于下文代码中将bs维度消除，此处也取消bs维度
    text_features: [num_tokens, f_dim]
    visual_feature: [f_dim]
    本段代码计算视觉特征与文本特征之间的相似度，并返回一个字典，包含基础相似度和邻居相似度
    '''
    if text_features is None or len(text_features) == 0:
        print("text_features is None or len(text_features) == 0")
        sim_dict = {}
        sim_dict['base_sim'] = 0
        sim_dict['neighbors_sim'] = 0
        return sim_dict
    else:
        base_feature = text_features[0]
        base_sim = get_sim(visual_feature, base_feature)
        neighbors_sim = 0
        for neighbor_feature in text_features[1:]:
            neighbors_sim += get_sim(visual_feature, neighbor_feature)
            
        sim_dict = {"base_sim": base_sim, "neighbors_sim": neighbors_sim}
        return sim_dict
    
def get_sim(visual_feature, text_feature): ## use cross_entropy to measure the similarity

    

    text_feature_shaped = text_feature.squeeze(0)
    visual_feature_shaped = visual_feature.squeeze(0)

    # 首先将 base_feature_reshaped 和 neighbor_feature_reshaped 转换为概率分布
    visual_feature_prob = F.softmax(visual_feature_shaped, dim=0)
    neighbor_feature_prob = F.softmax(text_feature_shaped, dim=0)
    # 计算交叉熵
    neighbor_feature_prob.to('cuda')
    visual_feature_prob.to('cuda')
    cross_entropy = - torch.sum(neighbor_feature_prob.to('cuda') * torch.log(visual_feature_prob))
    return cross_entropy









def distll_relation_map(text_features, visual_feature):
    if text_features is None or len(text_features) == 0:
        kl_divergence = 0
        return kl_divergence
        
    else:
        text_condition_maps, visual_condtion_maps = KD_relation_map(text_features, visual_feature)
        base_word_condition_map = text_condition_maps['base']
        visual_condtion_map = visual_condtion_maps['visual']
        

        base_word_condition_map = F.softmax(base_word_condition_map, dim=None)
        visual_condtion_map = F.softmax(visual_condtion_map, dim=None)

        # 计算KL散度
        kl_divergence = F.kl_div(visual_condtion_map.log(), base_word_condition_map.to('cuda'), reduction='batchmean')

        return kl_divergence



def KD_relation_map(text_features, visual_feature):
    '''
    由于加上bs难以计算，因此，此函数需要消去bs维度
    batched_text_features: [num_tokens, f_dim]
    batched_visual_feature: [f_dim], visual_feature is the feature of the chosen box、
    本段代码计算文本特征和视觉特征的条件概率图
    '''
    text_condition_maps = {}
    base_text_feature = text_features[0]
    base_condition_map = get_condition_map(base_text_feature)
    text_condition_maps['base'] = base_condition_map
    neighbor_text_features = text_features[1:]
    for i, neighbor_text_feature in enumerate(neighbor_text_features):
        neighbor_condition_map = get_condition_map(neighbor_text_feature)
        text_condition_maps[f'neighbor{i}'] = neighbor_condition_map
            
    visual_condtion_maps = {}
    visual_condtion_map = get_condition_map(visual_feature)
    visual_condtion_maps['visual'] = visual_condtion_map
    
    return text_condition_maps, visual_condtion_maps
    
    
def get_condition_map(feature):
    # feature = torch.tensor(feature.unsqueeze(-1))
    T_feature = feature.t()
    condition_map = T_feature @ feature
    return condition_map
    



class Adjust_Module(nn.Module):
    def __init__(self):
        super(Adjust_Module, self).__init__()
        
        # 全连接层
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        # CNN 层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 256)
        
    def forward(self, x):
        ori_x = x
        # 输入: (n, 256)
        x = self.fc1(x)  # (n, 512)
        x = self.fc2(x)  # (n, 1024)
    
        # Reshape 成 (n, 32, 32)
        x = x.view(-1, 1, 32, 32)  # (n, 1, 32, 32)
        
        # CNN 层
        x = self.conv1(x)  # (n, 32, 32, 32)
        x = self.pool(x)   # (n, 32, 16, 16)
        x = self.conv2(x)  # (n, 64, 16, 16)
        x = self.pool(x)   # (n, 64, 8, 8)
        
        # Reshape 回 (n, 256)
        x = x.view(-1, 64 * 8 * 8)  # (n, 4096)
        x = self.fc3(x)  # (n, 1024)
        x = self.fc4(x)
        return x + ori_x