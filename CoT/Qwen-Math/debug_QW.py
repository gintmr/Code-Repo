from transformers import AutoTokenizer, Qwen2ForCausalLM

model = Qwen2ForCausalLM.from_pretrained('/data05/user/DATA/QW2_5-3B-instruct')
tokenizer = AutoTokenizer.from_pretrained('/data05/user/DATA/QW2_5-3B-instruct')

# prompt = "Hey, are you conscious? Can you talk to me?"
prompt = "Hey, 1+1=?"


inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=30)



# 打印 pad_token 和 eos_token 的 ID
print("Pad Token ID:", tokenizer.pad_token_id)
print("EOS Token ID:", tokenizer.eos_token_id)

# 如果 pad_token 和 eos_token 相同，可以尝试重新设置
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    # 添加 pad_token 和 eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"eos_token": "[EOS]"})
    
    
output_txt = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_txt)
