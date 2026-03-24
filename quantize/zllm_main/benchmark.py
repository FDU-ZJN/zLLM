

import argparse
import asyncio
import json
import time
import torch
import numpy as np
from pathlib import Path
from vllm_inference import load_model, infer_stream, DEVICE, MAX_NEW_TOKENS

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def load_prompts(prompt_file: str) -> list:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"[INFO] 已加载 {len(prompts)} 条 prompt（来自 {prompt_file}）")
    return prompts


async def run_benchmark(tokenizer, model, prompts: list, batch_size: int = 100) -> dict:
    batch_size = max(1, int(batch_size))
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，按 batch_size={batch_size} 分批并发提交...")
    print("-" * 68)

    if len(prompts) == 0:
        return {
            "total_prompts": 0,
            "total_output_tokens": 0,
            "wall_time_sec": 0.0,
            "max_new_tokens_cfg": MAX_NEW_TOKENS,
            "overall_throughput_tps": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "avg_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "peak_gpu_mem_gb": 0.0,
        }

    latencies = []
    ttfts = []
    total_out = 0
    per_prompt_results = []

    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_wall_start = time.perf_counter()

    async def run_single_prompt(index: int, item):
        prompt = item["prompt"] if isinstance(item, dict) else item
        res = {}
        async for frame in infer_stream(tokenizer, model, prompt):
            if frame["finished"]:
                res = frame
                break
        return index, res

    all_results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        tasks = [
            asyncio.create_task(run_single_prompt(start + i, item))
            for i, item in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

    for i, res in sorted(all_results, key=lambda x: x[0]):
        if not res:
            continue
        latencies.append(res["total_latency_ms"])
        ttfts.append(res["ttft_ms"])
        total_out += res["output_tokens"]
        per_prompt_results.append((i, res))

    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start
    overall_throughput = (total_out / wall_time) if wall_time > 0 else 0.0

    print("[Benchmark] 全部请求完成，统一输出明细：")
    for i, res in per_prompt_results:
        print(
            f"  [{i+1:3d}/{len(prompts)}]  "
            f"latency={res['total_latency_ms']:8.1f} ms  "
            f"ttft={res['ttft_ms']:8.1f} ms  "
            f"throughput={res['throughput_tps']:6.1f} token/s "
            f"output={res['output_tokens']:4d} tokens"
        )

    stats = {
        "total_prompts":          len(prompts),
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     MAX_NEW_TOKENS,

        "overall_throughput_tps": round(overall_throughput, 2),

        "avg_latency_ms":         round(float(np.mean(latencies)), 2),
        "p50_latency_ms":         round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":         round(float(np.percentile(latencies, 99)), 2),

        "avg_ttft_ms":            round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms":            round(float(np.percentile(ttfts, 95)), 2),

        "peak_gpu_mem_gb":        round(
            torch.cuda.max_memory_allocated(DEVICE) / 1e9, 3
        ),
    }
    return stats


def print_stats(stats: dict):
    """格式化打印 benchmark 结果"""
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
        "--batch_size", type=int, default=100,
        help="调度批大小（每批并发 prompt 数，默认 100）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载模型
    tokenizer, model = load_model(args.model_path)

    # 加载 prompt
    prompts = load_prompts(args.prompt_file)

    # 运行测试（异步）
    stats = asyncio.run(run_benchmark(tokenizer, model, prompts, batch_size=args.batch_size))

    # 打印结果
    print_stats(stats)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")