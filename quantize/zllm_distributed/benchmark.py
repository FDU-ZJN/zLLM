#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py
============
吞吐量 & 延迟基准测试脚本

使用方式：
  # 运行基准测试
  python benchmark.py --model_path /path/to/model --output results_baseline.json
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from zllm_inference import load_model, infer_single, DEVICE, MAX_NEW_TOKENS

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def _resolve_cuda_metric_devices(tp_world_size: int) -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return []
    world = max(1, int(tp_world_size))
    visible = torch.cuda.device_count()
    return list(range(min(world, visible)))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_percentile(values: list[float], q: int) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def load_prompts(prompt_file: str) -> list:
    """读取 prompts.jsonl，每行格式：{"id": 1, "prompt": "..."}"""
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"[INFO] 已加载 {len(prompts)} 条 prompt（来自 {prompt_file}）")
    return prompts


def run_benchmark(tokenizer, model, prompts: list, tp_world_size: int = 4, warmup_prompts: int = 1) -> dict:
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理...")
    print("-" * 68)

    warmup_count = max(0, int(warmup_prompts))
    if warmup_count > len(prompts):
        warmup_count = len(prompts)

    if warmup_count > 0:
        print(f"[Warmup] 预热 {warmup_count} 条请求（不计入统计）...")
        for item in prompts[:warmup_count]:
            prompt = item["prompt"] if isinstance(item, dict) else item
            infer_single(tokenizer, model, prompt)

    eval_prompts = prompts[warmup_count:]
    print(f"[Benchmark] 计入统计 {len(eval_prompts)} 条请求")

    latencies = []
    ttfts = []
    prefill_latencies = []
    decode_latencies = []
    total_out = 0
    total_in = 0

    metric_devices = _resolve_cuda_metric_devices(tp_world_size)
    for dev in metric_devices:
        torch.cuda.reset_peak_memory_stats(dev)
    t_wall_start = time.perf_counter()

    for i, item in enumerate(eval_prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        res    = infer_single(tokenizer, model, prompt)

        latencies.append(res["total_latency_ms"])
        ttfts.append(res["ttft_ms"])
        prefill_latencies.append(float(res.get("prefill_ms", 0.0)))
        decode_latencies.append(float(res.get("decode_ms", 0.0)))
        total_out += res["output_tokens"]
        total_in += res.get("input_tokens", 0)

        print(
            f"  [{i+1:3d}/{len(eval_prompts)}]  "
            f"latency={res['total_latency_ms']:8.1f} ms  "
            f"ttft={res['ttft_ms']:8.1f} ms  "
            f"throughput={res['throughput_tps']:6.1f} token/s "
            f"output={res['output_tokens']:4d} tokens"
        )

    t_wall_end = time.perf_counter()
    wall_time  = t_wall_end - t_wall_start

    measured_prompts = len(eval_prompts)
    total_tokens = total_in + total_out
    overall_tps = (total_out / wall_time) if wall_time > 0 else 0.0
    qps = (measured_prompts / wall_time) if wall_time > 0 else 0.0
    per_gpu_tps = (overall_tps / tp_world_size) if tp_world_size > 0 else 0.0
    gpu_seconds = wall_time * tp_world_size
    tokens_per_gpu_second = (total_out / gpu_seconds) if gpu_seconds > 0 else 0.0
    avg_ms_per_output_token = (sum(latencies) / total_out) if total_out > 0 else 0.0

    peak_gpu_mem_each_gb = []
    for dev in metric_devices:
        peak_gpu_mem_each_gb.append(round(torch.cuda.max_memory_allocated(dev) / 1e9, 3))

    peak_gpu_mem_gb = max(peak_gpu_mem_each_gb) if peak_gpu_mem_each_gb else None
    avg_gpu_mem_gb = round(float(np.mean(peak_gpu_mem_each_gb)), 3) if peak_gpu_mem_each_gb else None
    prefill_ratio = (total_in / total_tokens) if total_tokens > 0 else 0.0
    decode_ratio = (total_out / total_tokens) if total_tokens > 0 else 0.0

    stats = {
        "tp_world_size_assumed":  tp_world_size,
        "gpu_count_visible":      torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_count_measured":     len(metric_devices),
        "warmup_prompts":         warmup_count,
        "total_prompts":          len(prompts),
        "measured_prompts":       measured_prompts,
        "total_input_tokens":     total_in,
        "total_output_tokens":    total_out,
        "total_tokens":           total_tokens,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     MAX_NEW_TOKENS,

        "overall_throughput_tps": round(overall_tps, 2),
        "overall_qps":            round(qps, 2),
        "per_gpu_throughput_tps": round(per_gpu_tps, 2),
        "tokens_per_gpu_second":  round(tokens_per_gpu_second, 2),
        "avg_ms_per_output_token": round(avg_ms_per_output_token, 3),
        "prefill_token_ratio":    round(prefill_ratio, 4),
        "decode_token_ratio":     round(decode_ratio, 4),

        "avg_latency_ms":         round(_safe_mean(latencies), 2),
        "p50_latency_ms":         round(_safe_percentile(latencies, 50), 2),
        "p95_latency_ms":         round(_safe_percentile(latencies, 95), 2),
        "p99_latency_ms":         round(_safe_percentile(latencies, 99), 2),

        "avg_ttft_ms":            round(_safe_mean(ttfts), 2),
        "p95_ttft_ms":            round(_safe_percentile(ttfts, 95), 2),

        "avg_prefill_ms":         round(_safe_mean(prefill_latencies), 2),
        "avg_decode_ms":          round(_safe_mean(decode_latencies), 2),

        "peak_gpu_mem_gb":        peak_gpu_mem_gb,
        "avg_gpu_mem_gb":         avg_gpu_mem_gb,
        "peak_gpu_mem_each_gb":   peak_gpu_mem_each_gb,
    }
    return stats


def print_stats(stats: dict):
    """格式化打印 benchmark 结果"""
    labels = {
        "tp_world_size_assumed":  "TP 并行卡数（假设）       ",
        "gpu_count_visible":      "可见 GPU 数              ",
        "gpu_count_measured":     "参与显存统计 GPU 数       ",
        "warmup_prompts":         "预热请求数（不计入）      ",
        "total_prompts":          "测试 prompt 数",
        "measured_prompts":       "计入统计请求数            ",
        "total_input_tokens":     "总输入 tokens             ",
        "total_output_tokens":    "总输出 tokens",
        "total_tokens":           "总 tokens（入+出）        ",
        "wall_time_sec":          "总耗时 (sec)",
        "max_new_tokens_cfg":     "max_new_tokens 配置",
        "overall_throughput_tps": "集群吞吐率 (tokens/sec)   ",
        "overall_qps":            "请求吞吐 (req/sec)        ",
        "per_gpu_throughput_tps": "单卡吞吐 (tokens/sec)     ",
        "tokens_per_gpu_second":  "效率 (tokens/GPU-second)  ",
        "avg_ms_per_output_token": "平均每输出token耗时 (ms)   ",
        "prefill_token_ratio":    "输入token占比             ",
        "decode_token_ratio":     "输出token占比             ",
        "avg_latency_ms":         "平均延迟 (ms)            ",
        "p50_latency_ms":         "P50 延迟 (ms)            ",
        "p95_latency_ms":         "P95 延迟 (ms)            ",
        "p99_latency_ms":         "P99 延迟 (ms)            ",
        "avg_ttft_ms":            "平均 TTFT (ms)            ",
        "p95_ttft_ms":            "P95 TTFT (ms)             ",
        "avg_prefill_ms":         "平均 Prefill 时延 (ms)      ",
        "avg_decode_ms":          "平均 Decode 时延 (ms)       ",
        "peak_gpu_mem_gb":        "最大单卡峰值显存 (GB)      ",
        "avg_gpu_mem_gb":         "平均峰值显存 (GB)          ",
        "peak_gpu_mem_each_gb":   "逐卡峰值显存 (GB)          ",
    }
    print("\n" + "=" * 68)
    print(" Benchmark 结果汇总（baseline）")
    print("=" * 68)
    for key, label in labels.items():
        val = stats.get(key, "N/A")
        print(f"  {label:<40s}: {val}")
    print("=" * 68)
    print("  注：精度指标请运行 evaluate_accuracy.py")
    print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理吞吐 & 延迟基准测试")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /data/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE),
        help="prompt 文件路径（默认：prompts.jsonl）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 results_baseline.json"
    )
    parser.add_argument(
        "--tp_world_size", type=int, default=4,
        help="用于统计口径的 TP 卡数（不改变实际推理拓扑）"
    )
    parser.add_argument(
        "--warmup_prompts", type=int, default=1,
        help="预热请求数（不计入统计），默认 1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载模型
    tokenizer, model = load_model(args.model_path)

    # 加载 prompt
    prompts = load_prompts(args.prompt_file)

    # 运行测试
    stats = run_benchmark(
        tokenizer,
        model,
        prompts,
        tp_world_size=args.tp_world_size,
        warmup_prompts=args.warmup_prompts,
    )

    # 打印结果
    print_stats(stats)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")
