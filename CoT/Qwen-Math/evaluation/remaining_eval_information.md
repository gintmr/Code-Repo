`/data05/wuxinrui/Qwen2.5-Math/evaluation/remaining_eval.py`

修改后的推理代码👆

## 主要改动：
### 功能
涉及到两个环境变量：tip和stage

tip包括：

- Ahead：

    将`\n<remaining>[{budget} token]</remaining>\n`加入进问题的末尾

- remaining：

    包含Ahead功能，且同时在推理时递归地插入倒计时budget token

- prompt-based：

    在问题末尾加上基于自然语言的prompt提示，例如`You should finish thinking with in {budget} tokens.\n`

- default：

    对问题不做处理

stage包括：

- 1：一次性输出完成

- 2：在第一轮输出完成后，在文本末尾处添加`</think>\n\n**Final Answer**\n\\boxed`

### template
Ⅰ增加了`deepseek3`的提示模板

- 路径：`/data05/wuxinrui/Qwen2.5-Math/evaluation/utils.py`

Ⅱ同时在终止符列表中加入了`deepseek3`模板对应的终止符

- 路径：`/data05/wuxinrui/Qwen2.5-Math/evaluation/remaining_eval.py`

