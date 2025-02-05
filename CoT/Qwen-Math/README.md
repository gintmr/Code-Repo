## Intro
该repo基于https://github.com/QwenLM/Qwen2.5-Math修改

## Main
主要关注：
- **Core**: CoT\Qwen2.5-Math\vllm\rotary_embedding.py 文件对应路径为/data03/xxxx/xxxx/miniconda3/envs/QMath/lib/python3.9/site-packages/vllm/model_executor/layers/rotary_embedding.py
  
    👆：主要更新四个新功能。同时，找出测验时调用的函数路径：
    
    - `rope_scaling` 为None
    - 进入 `RotaryEmbedding`
    - `RotaryEmbedding` 中调用 `forward_cuda`
    - 在`forward_cuda` 实现四个功能的切换

    同样的，对应CoT\llama-qw\modeling_qwen2.py 文件，两者其中一个用于llama-factory的微调，另一个用于Qwen2.5-Math的测验。
---
- CoT\Qwen2.5-Math\debug_QW.py 为简单调用本地模型推理，常用于debug
- CoT\Qwen2.5-Math\merge_safetensors.py 是一个可能存在问题的safetensors合并脚本
- CoT\Qwen2.5-Math\evaluation\eval.sh 可以批量进行测试的脚本
- CoT\Qwen2.5-Math\evaluation\debug_eval.py 将测试sh脚本转换成py脚本
- CoT\Qwen2.5-Math\evaluation\debug_trace.py 跟踪上述debug_eval文件执行时历经的文件调用/代码情况
- CoT\Qwen2.5-Math\evaluation\.vscode\launch.json 用于调试debug_eval文件时的配置
- D:\@Github_local_storage\Code-Repo\CoT\Qwen2.5-Math\evaluation\math_eval.py 新增了logging模块，记录当前调用的模型路径、环境变量参数内容、测试结果···

- 方便起见，删除了data文件夹中的数据内容，仅保留了文件夹
---
仓库代码更新：
- 将inv_freq实现移动至init函数中，原因：
    ```text
    问题分析
    CUDA流捕获：CUDA流捕获（CUDA Stream Capture）是一种用于优化CUDA操作的技术，通常在捕获期间，某些操作（如内存分配、设备同步等）是不被允许的。你的代码在捕获期间尝试将inv_freq移动到GPU，这导致了错误。

    inv_freq的移动：在_compute_ldpe函数中，inv_freq是在CPU上生成的，然后你尝试将其移动到GPU。这个操作在CUDA流捕获期间是不允许的。

    解决方案
    为了避免在CUDA流捕获期间进行设备间的数据传输，你可以在捕获开始之前预先计算并移动inv_freq到GPU。具体来说，你可以在类的初始化阶段或第一次调用_compute_ldpe时，将inv_freq计算并移动到GPU，然后在后续调用中直接使用已经存在于GPU上的inv_freq。
    ```
- 与本改动牵连的代码有：
  - /data05/xxxx/Qwen2.5-Math/evaluation/math_eval.py
  - /data05/xxxx/Qwen2.5-Math/evaluation/debug_eval.py
  - /data03/xxxx/xxxx/miniconda3/envs/QMath/lib/python3.9/site-packages/vllm/model_executor/models/qwen2.py
  - /data03/xxxx/xxxx/miniconda3/envs/QMath/lib/python3.9/site-packages/vllm/model_executor/layers/rotary_embedding.py
  - /data03/xxxx/xxxx/miniconda3/envs/QMath/lib/python3.9/site-packages/vllm_flash_attn/flash_attn_interface.py
  