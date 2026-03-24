#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
import torch
import numpy as np
from inference_optimized import load_model, infer_batch, DEVICE

def load_all_prompts(prompt_file: str):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append(data.get("prompt", ""))
    return prompts


def run_benchmark(tokenizer, model, prompts, batch_size, max_new_tokens):
    print(f"\n[Benchmark] 准备处理 {len(prompts)} 条数据 | Batch Size: {batch_size}")
    
    # 1. 预处理所有数据 (不再计入推理时间)
    print("[INFO] 正在全量预处理 Prompts...")
    encoded_all = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    input_ids_all = encoded_all["input_ids"]
    mask_all = encoded_all["attention_mask"]
    num_samples = input_ids_all.shape[0]

    # 2. 显存预热
    print("[INFO] 正在预热 GPU...")
    warmup_ids = input_ids_all[:1]
    warmup_mask = mask_all[:1]
    model.generate(input_ids=warmup_ids, attention_mask=warmup_mask, max_new_tokens=10)
    
    # 3. 循环推理
    total_gen_tokens = 0
    total_inference_time = 0
    
    print("[INFO] 开始正式计时推理...")
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_tensors = {
            "input_ids": input_ids_all[i:end_idx],
            "attention_mask": mask_all[i:end_idx]
        }
        
        tokens, duration = infer_batch(tokenizer, model, batch_tensors, max_new_tokens)
        total_gen_tokens += tokens
        total_inference_time += duration
        
        print(f"  Batch {i//batch_size + 1}: 生成 {tokens} tokens, 耗时 {duration:.2f}s")

    # 4. 计算指标
    overall_throughput = total_gen_tokens / total_inference_time
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    print("\n" + "=" * 60)
    print(f"  极致吞吐量测试结果")
    print("-" * 60)
    print(f"  Batch Size          : {batch_size}")
    print(f"  总计生成 Tokens      : {total_gen_tokens}")
    print(f"  总推理净耗时 (sec)   : {total_inference_time:.2f}")
    print(f"  >>> 吞吐率 (TPS)     : {overall_throughput:.2f} tokens/sec")
    print(f"  峰值显存占用 (GB)    : {peak_mem:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default="prompts.jsonl")
    parser.add_argument("--batch_size", type=int, default=16, help="在不爆显存的前提下尽量调大")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    prompts = load_all_prompts(args.prompt_file)
    
    run_benchmark(tokenizer, model, prompts, args.batch_size, args.max_new_tokens)