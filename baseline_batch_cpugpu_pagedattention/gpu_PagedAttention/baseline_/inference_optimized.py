#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEVICE = "cuda:0"
DTYPE = torch.float16

def load_model(model_path: str):
    print(f"[INFO] 极致吞吐模式加载: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left", # 批量推理必须左填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit 量化：大幅减少显存占用，从而允许挤进更大的 Batch Size
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map=DEVICE,
        trust_remote_code=True,
        # 如果硬件支持 Flash Attention 2 (如 A100/4090)，取消下方注释
        # attn_implementation="flash_attention_2",
    )
    model.eval()
    return tokenizer, model

@torch.inference_mode()
def infer_batch(tokenizer, model, input_tensors, max_new_tokens: int):
    """
    直接接收预处理好的张量进行推理，规避循环内的 Tokenize 耗时
    """
    input_ids = input_tensors["input_ids"]
    attention_mask = input_tensors["attention_mask"]
    
    # 仅针对生成阶段计时，排除数据传输波动
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy Search 速度最快
        use_cache=True,   # 必须开启 KV Cache
        pad_token_id=tokenizer.pad_token_id,
    )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    # 计算本次 batch 生成的总 token 数
    gen_len = output_ids.shape[1] - input_ids.shape[1]
    total_tokens = gen_len * input_ids.shape[0]
    duration_sec = t_end - t_start

    return total_tokens, duration_sec