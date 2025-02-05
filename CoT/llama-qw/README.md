## 文件对应路径
/data03/xxxx/xxxx/miniconda3/envs/llama-qw/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py

## Details
- 对modeling_qwen2文件进行了修改，新增4个模式，并且以环境变量形式传入BUDGET变量与PE_MODE变量
    
    4个模式分别是:
    - reverse
    - hybrid
    - ldpe
    - lrpe
  
    最后两个模式详情参见:https://arxiv.org/pdf/1904.07418
- CoT\llama-qw\training_instruct.sh 为在LLaMaFactory中训练的自动化脚本
- CoT\llama-qw\merge_reverse.yaml 为Lora合并所用yaml配置
## Tip
- 在ldpe与lrpe中，需要的是实现sin与cos的穿插摆放而非简单拼接
- reverse、ldpe与lrpe比较类似，前者是直接点对点相乘，后两者则是直接叠加
  
