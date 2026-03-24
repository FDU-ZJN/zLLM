#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_inference.py
=================
基于vLLM的推理基准脚本，功能对齐baseline_inference.py

快速运行：
  python vllm_inference.py --model_path /path/to/model --prompt "你的prompt"
"""

import argparse
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

DEVICE = "cuda:1"
DTYPE = "float16"
MAX_NEW_TOKENS = 256


def load_model(model_path: str):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备: {DEVICE} | 数据类型: {DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(model=model_path, dtype=DTYPE, trust_remote_code=True, tensor_parallel_size=1)
    return tokenizer, llm


def infer_single(tokenizer, llm, prompt: str) -> dict:
    inputs = prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    input_len = input_ids.shape[0]
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, top_p=1.0, stop=None)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    outputs = llm.generate([inputs], sampling_params)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_text = outputs[0].outputs[0].text
    output_ids = tokenizer(output_text, return_tensors="pt").input_ids[0]
    output_len = output_ids.shape[0]
    total_ms = (t_end - t_start) * 1000
    ttft_approx = total_ms
    throughput = output_len / total_ms * 1000 if total_ms > 0 else 0

    return {
        "prompt": prompt,
        "output": output_text,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_latency_ms": round(total_ms, 2),
        "ttft_approx_ms": round(ttft_approx, 2),
        "throughput_tps": round(throughput, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM 推理基准 —— 单条 prompt 快速验证"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /data/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="请用三句话解释大语言模型推理中KV Cache的作用。",
        help="测试用 prompt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer, llm = load_model(args.model_path)

    print(f"\n[推理] prompt: {args.prompt}")
    result = infer_single(tokenizer, llm, args.prompt)

    print("\n" + "=" * 64)
    print(" 推理结果")
    print("=" * 64)
    print(f"  输入   : {result['prompt']}")
    print(f"  输出   : {result['output']}")
    print("-" * 64)
    print(f"  输入 tokens   : {result['input_tokens']}")
    print(f"  输出 tokens   : {result['output_tokens']}")
    print(f"  总延迟         : {result['total_latency_ms']} ms")
    print(f"  TTFT (近似)   : {result['ttft_approx_ms']} ms/token")
    print(f"  吞吐率         : {result['throughput_tps']} tokens/sec")
    print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print("=" * 64)
