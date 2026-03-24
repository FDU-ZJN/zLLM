#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_vllm.py
============
vLLM 基准测试 - 最终修正版
修复重点：
1. 在非流式模式下，使用 output.metrics 获取真实的 TTFT。
2. 在流式模式下，使用 vLLM 的异步引擎实现真正的流式输出。
"""

import argparse
import json
import time
import os
import numpy as np
from pathlib import Path
import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer
import torch

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"
DEFAULT_MAX_TOKENS = 256


def load_prompts(prompt_file: str) -> list:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"[INFO] 已加载 {len(prompts)} 条 prompt")
    return prompts


async def run_streaming_benchmark(engine, tokenizer, prompts: list, args) -> dict:
    """流式模式基准测试"""
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理...")
    print("[INFO] 当前模式：真正的流式输出")
    print("-" * 68)
    
    latencies = []
    ttfts = []
    total_out = 0
    
    torch.cuda.reset_peak_memory_stats()
    t_wall_start = time.perf_counter()
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        
        if args.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # 记录开始时间
        t_start = time.perf_counter()
        first_token_time = None
        output_tokens = 0
        
        # 生成请求 ID
        request_id = f"request_{i}"
        
        # 流式生成
        output_text = ""
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in results_generator:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            
            # 更新输出 token 数
            for output in request_output.outputs:
                output_text += output.text
                output_tokens = len(output.token_ids)
        
        # 计算 TTFT
        if first_token_time is not None:
            ttft_ms = (first_token_time - t_start) * 1000
        else:
            ttft_ms = 0
        
        # 记录结束时间
        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000
        
        latencies.append(total_latency_ms)
        ttfts.append(ttft_ms)
        total_out += output_tokens
        
        print(
            f"  [{i+1:3d}/{len(prompts)}]  "
            f"latency={total_latency_ms:8.1f} ms  "
            f"ttft={ttft_ms:8.1f} ms  "
            f"output={output_tokens:4d} tokens"
        )
    
    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start
    
    stats = {
        "total_prompts":          len(prompts),
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     args.max_tokens,
        "is_streaming":           args.streaming,
        "overall_throughput_tps": round(total_out / wall_time, 2),
        "avg_latency_ms":         round(float(np.mean(latencies)), 2),
        "p50_latency_ms":         round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":         round(float(np.percentile(latencies, 99)), 2),
        "avg_ttft_ms":            round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms":            round(float(np.percentile(ttfts, 95)), 2),
        "peak_gpu_mem_gb":        round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    return stats


def run_nonstreaming_benchmark(llm, tokenizer, prompts: list, args) -> dict:
    """非流式模式基准测试"""
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理...")
    print("[INFO] 当前模式：非流式输出 (Batch)")
    print("-" * 68)
    
    latencies = []
    ttfts = []
    total_out = 0
    
    torch.cuda.reset_peak_memory_stats()
    t_wall_start = time.perf_counter()
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        
        if args.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # 记录开始时间
        t_start = time.perf_counter()
        
        # 非流式生成
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
        
        if output.metrics is not None:
            ttft_ms = output.metrics.first_token_time * 1000
        else:
            ttft_ms = 0
        
        # 记录结束时间
        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000
        
        latencies.append(total_latency_ms)
        ttfts.append(ttft_ms)
        total_out += output_tokens
        
        print(
            f"  [{i+1:3d}/{len(prompts)}]  "
            f"latency={total_latency_ms:8.1f} ms  "
            f"ttft={ttft_ms:8.1f} ms  "
            f"output={output_tokens:4d} tokens"
        )
    
    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start
    
    stats = {
        "total_prompts":          len(prompts),
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     args.max_tokens,
        "is_streaming":           args.streaming,
        "overall_throughput_tps": round(total_out / wall_time, 2),
        "avg_latency_ms":         round(float(np.mean(latencies)), 2),
        "p50_latency_ms":         round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":         round(float(np.percentile(latencies, 99)), 2),
        "avg_ttft_ms":            round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms":            round(float(np.percentile(ttfts, 95)), 2),
        "peak_gpu_mem_gb":        round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    return stats


def print_stats(stats: dict):
    labels = {
        "total_prompts":          "测试 prompt 数",
        "total_output_tokens":    "总输出 tokens",
        "wall_time_sec":          "总耗时 (sec)",
        "max_new_tokens_cfg":     "max_new_tokens 配置",
        "overall_throughput_tps": "整体吞吐率 (tokens/sec)  ",
        "avg_latency_ms":         "平均延迟 (ms)            ",
        "p50_latency_ms":         "P50 延迟 (ms)            ",
        "p95_latency_ms":         "P95 延迟 (ms)            ",
        "p99_latency_ms":         "P99 延迟 (ms)            ",
        "avg_ttft_ms":            "平均 TTFT (ms)            ",
        "p95_ttft_ms":            "P95 TTFT (ms)             ",
        "peak_gpu_mem_gb":        "峰值显存 (GB)             ",
    }
    print("\n" + "=" * 68)
    print(" Benchmark 结果汇总（vLLM）")
    if stats.get("is_streaming"):
        print(" (流式模式)")
    print("=" * 68)
    for key, label in labels.items():
        val = stats.get(key, "N/A")
        print(f"  {label:<40s}: {val}")
    print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理基准测试 (vLLM)")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE), help="Prompt 文件")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--streaming", action="store_true", help="启用流式模式")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--quantization", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 模型路径不存在：{args.model_path}")
        exit(1)
    
    print(f"[INFO] 加载模型：{args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.streaming:
        # 流式模式：使用 AsyncLLMEngine
        print("[INFO] 初始化异步引擎...")
        engine_args = AsyncEngineArgs(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            quantization=args.quantization,
            dtype=torch.float16
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        prompts = load_prompts(args.prompt_file)
        stats = asyncio.run(run_streaming_benchmark(engine, tokenizer, prompts, args))
    else:
        # 非流式模式：使用 LLM
        from vllm import LLM
        llm = LLM(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            quantization=args.quantization,
            dtype=torch.float16
        )
        prompts = load_prompts(args.prompt_file)
        stats = run_nonstreaming_benchmark(llm, tokenizer, prompts, args)
    
    print_stats(stats)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至：{args.output}")