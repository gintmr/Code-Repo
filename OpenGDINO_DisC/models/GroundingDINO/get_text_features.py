import copy
import sys

# sys.path.append("/data2/wuxinrui/OpenGDINO_DisC/groundingdino/util")
# sys.path.append("/data2/wuxinrui/OpenGDINO_DisC/models/GroundingDINO")
# sys.path.append("/data2/wuxinrui/OpenGDINO_DisC")

from typing import List
import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

from .matcher import build_matcher
# from .DisCutils import KD_relation_map, vis2text_crossentropy
from groundingdino.util.vl_utils import create_positive_map_from_span





gdino_weights_path = '/data2/wuxinrui/OpenGDINO_DisC/weights/groundingdino_swint_ogc.pth'

text_encoder_type = "bert-base-uncased"


featmap_weight_path = '/data2/wuxinrui/OpenGDINO_DisC/LLM_SOLVE/featmap_weight.pth'





output_folder = '/data2/wuxinrui/OpenGDINO_DisC/text_features'
os.makedirs(output_folder, exist_ok=True)

tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)

class TextFeatureExtractor(nn.Module):
    def __init__(self, noun:str):
        super(TextFeatureExtractor, self).__init__()
        # self.noun_len = len(noun)
        self.noun = noun
        # self.span_token = [[[0, self.noun_len]]]
        
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        self.hidden_dim = 256
        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        # 初始化特征图偏置为0
        nn.init.constant_(self.feat_map.bias.data, 0)
        # 初始化特征图权重为Xavier均匀分布
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.max_text_len = 256
        
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])


    def get_text(self, noun):
        path_2_json_of_noun = f"/data2/wuxinrui/OpenGDINO_DisC/LLM_SOLVE/gintmr_jsons_with_tree_structure/{noun}.json"
        with open(path_2_json_of_noun, 'r') as f:
            data = json.load(f)
        texts = []
        for key, item in data.items():
            # print(item)
            
            input_text = str(item['name']) + ',' + " " + str(item['discription'])
            texts.append(input_text)
            
        return texts
    
    def init_text(self, text):
        text = text.lower()
        text = text.strip()
        if not text.endswith("."):
            text = text + "."
        return text
        
        
    def load_weights(self, path):
        state_dict = torch.load(path)
         # 加载 feat_map 的权重
        feat_map_state_dict = {k: v for k, v in state_dict.items() if k.startswith('feat_map.')}
        self.feat_map.load_state_dict(feat_map_state_dict, strict=False)

        
        
    def forward(self):
        global featmap_weight_path
        noun = self.noun
        state_dict = torch.load(featmap_weight_path)
         # 加载 feat_map 的权重
        # feat_map_state_dict = {k: v for k, v in state_dict.items() if k.startswith('feat_map.')}
        self.feat_map.weight.data = state_dict
        self.feat_map.to("cpu")
        
        texts = self.get_text(noun)
        for text in texts:
            subnoun = text.split(",")[0]
            splited = text.split(",")
            self.span_token = [[[0, len(subnoun)]]]
            print(f"{subnoun} ===> span_token = {self.span_token}")
            text = self.init_text(text)
            # print(text)
            captions = [text]
            tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to("cpu")
            
            
            one_hot_token = tokenized
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )
            
            
            
            if text_self_attention_masks.shape[1] > 256:
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]


            # 将tokenized字典中除了"attention_mask"以外的键值对存入tokenized_for_encoder字典中
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            # 将text_self_attention_masks赋值给tokenized_for_encoder字典中的"attention_mask"键
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            # 将position_ids赋值给tokenized_for_encoder字典中的"position_ids"键
            tokenized_for_encoder["position_ids"] = position_ids


            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

            encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]

            text_dict = {
                "encoded_text": encoded_text,  # bs, 195, d_model
                "text_token_mask": text_token_mask,  # bs, 195
                "position_ids": position_ids,  # bs, 195
                "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            }
            
            positive_features = get_positive_features(text_dict["encoded_text"], 
                                                    token_span=self.span_token, tokenized=tokenized)
            
            os.makedirs(f"/data2/wuxinrui/OpenGDINO_DisC/noun_features/{self.noun}/", exist_ok=True)
            
            torch.save(positive_features, f"/data2/wuxinrui/OpenGDINO_DisC/noun_features/{self.noun}/{subnoun}.pth")
            print(f"successfully saved {subnoun}.pth with the text of {text}")
        
        
        return text_dict, positive_features
    



def get_positive_features(encoded_text, token_span, tokenized):
    '''
    将被分词表划分开的词汇拼接起来
    
    tokenized： 上述类中 forward 函数中计算得出
    token_span： 词汇在分词表中的起始位置和结束位置
    encoded_text： 上述类中 forward 函数中计算得出 bs, n_tokens, 256
    
    
    '''
    positive_map = create_positive_map_from_span(
        tokenized=tokenized, token_span=token_span
    ) ##输出的positive_map的phrase，256形状中第一个维度是有多少组span，而非最初的token数量。
    ## 1, 256 ＆ 1, n_tokens, 256
    encoded_text = encoded_text.squeeze(0) ## 维度变成 n_tokens, 256
    positive_map = positive_map.squeeze(0) ## 维度变成 256
    ## 选取positive_map中值为1的索引
    # print(positive_map)
    # print(positive_map.shape)
    positive_indices = positive_map.nonzero()
    # print(positive_indices)
    ## 选取encoded_text中positive_indices对应的行
    positive_features = torch.zeros(1, 256) 
    for positive_indice in positive_indices:
        positive_features += encoded_text[positive_indice]
    ## 归一化
    positive_features = positive_features / positive_features.norm(dim=-1, keepdim=True)
    
    return positive_features
     