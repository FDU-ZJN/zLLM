import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

DEFAULT_MODEL_PATH = "/inspire/ssd/project/mianxiangdayuyanmoxing/public/Qwen2.5-14B-Instruct/"
DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


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


def run_benchmark(
    tokenizer,
    model,
    prompts: list,
    meta: Optional[dict] = None,
    infer_mode: str = "single",
    dynamic_policy: str = "sorted_tokens",
    max_batch_size: int = 16,
    max_wait_ms: int = 20,
    aging_rate: float = 0.1,
    max_starve_ms: float = 500.0,
) -> dict:
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理（infer_mode={infer_mode}）...")
    print("-" * 68)

    latencies   = []
    ttfts       = []
    total_out   = 0

    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_wall_start = time.perf_counter()

    if infer_mode == "dynamic":
        prompt_strs = [item["prompt"] if isinstance(item, dict) else item for item in prompts]
        results = infer_dynamic_batch(
            tokenizer,
            model,
            prompt_strs,
            policy=dynamic_policy,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            aging_rate=aging_rate,
            max_starve_ms=max_starve_ms,
        )
        for i, res in enumerate(results):
            latencies.append(res["total_latency_ms"])
            ttfts.append(res["ttft_ms"])
            total_out += res["output_tokens"]
            print(
                f"  [{i+1:3d}/{len(prompts)}]  "
                f"latency={res['total_latency_ms']:8.1f} ms  "
                f"ttft={res['ttft_ms']:8.1f} ms  "
                f"throughput={res['throughput_tps']:6.1f} token/s "
                f"output={res['output_tokens']:4d} tokens"
                f"  mode={res.get('mode', '')} batch={res.get('batch_size', '')}"
            )
    else:
        for i, item in enumerate(prompts):
            prompt = item["prompt"] if isinstance(item, dict) else item
            res    = infer_single(tokenizer, model, prompt)

            latencies.append(res["total_latency_ms"])
            ttfts.append(res["ttft_ms"])
            total_out += res["output_tokens"]

            print(
                f"  [{i+1:3d}/{len(prompts)}]  "
                f"latency={res['total_latency_ms']:8.1f} ms  "
                f"ttft={res['ttft_ms']:8.1f} ms  "
                f"throughput={res['throughput_tps']:6.1f} token/s "
                f"output={res['output_tokens']:4d} tokens"
            )

    t_wall_end = time.perf_counter()
    wall_time  = t_wall_end - t_wall_start

    stats = {
        "total_prompts":          len(prompts),
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     MAX_NEW_TOKENS,

        "overall_throughput_tps": round(total_out / wall_time, 2),

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
    if meta:
        stats = {**meta, **stats}
    return stats


def print_stats(stats: dict):
    """格式化打印 benchmark 结果"""
    labels = {
        "infer_mode":             "推理模式                 ",
        "dynamic_policy":          "动态 batch 策略         ",
        "backend":                "推理后端                 ",
        "use_paged_kv":           "启用 Paged KV            ",
        "weight_dtype":           "权重 dtype               ",
        "kv_pool_gb":             "KV 池预算 (GiB)         ",
        "model_path":             "模型路径                 ",
        "inference_module":       "推理模块                 ",
        "torch_version":          "PyTorch                  ",
        "cuda_version":           "CUDA                     ",
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
    print(" Benchmark 结果汇总（默认 HF 原生 KV；--paged_kv 时为 Paged 池）")
    print("=" * 68)
    for key, label in labels.items():
        val = stats.get(key, "N/A")
        if val is None or val == "":
            val = "N/A"
        print(f"  {label:<40s}: {val}")
    print("=" * 68)
    print("  注：精度指标请运行 evaluate_accuracy.py")
    print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理吞吐 & 延迟基准测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"模型本地路径（默认: {DEFAULT_MODEL_PATH}）",
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
        "--backend",
        type=str,
        default="fp16",
        choices=["fp16", "awq"],
        help="fp16=zllm_inference；默认 HF 原生 KV（低延迟），可加 --paged_kv 测 Paged 池",
    )
    parser.add_argument(
        "--paged_kv",
        action="store_true",
        help="启用 Paged 物理 KV + attention patch（延迟更高，仅对比/continuous 场景）",
    )
    parser.add_argument(
        "--kv_pool_gb",
        type=float,
        default=None,
        help="Paged KV 池显存上限 (GiB)，写入环境变量 ZLLM_KV_POOL_GB；默认沿用 PagedKVCacheManager 或已有环境变量",
    )
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="single",
        choices=["single", "dynamic"],
        help="single=逐条 infer_single（默认）；dynamic=批次级调度 infer_dynamic_batch（poll_batch 组批后顺序执行）",
    )
    parser.add_argument(
        "--dynamic_policy",
        type=str,
        default="sorted_tokens",
        choices=["sorted_tokens", "fifo", "aging"],
        help="仅 infer_mode=dynamic 时有效：sorted_tokens / fifo / aging（见 dynamic_batch_scheduler）",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="dynamic 模式下每批最大请求数（默认 16）",
    )
    parser.add_argument(
        "--max_wait_ms",
        type=int,
        default=20,
        help="dynamic 模式下调度器等待窗口 (ms)",
    )
    parser.add_argument(
        "--aging_rate",
        type=float,
        default=0.1,
        help="dynamic_policy=aging 时的 aging 系数",
    )
    parser.add_argument(
        "--max_starve_ms",
        type=float,
        default=500.0,
        help="dynamic_policy=aging 时的最大饥饿时间 (ms)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.kv_pool_gb is not None:
        os.environ["ZLLM_KV_POOL_GB"] = str(args.kv_pool_gb)

    if args.backend == "awq":
        import zllm_inference_awq as _inf
    else:
        import zllm_inference as _inf

    load_model = _inf.load_model
    infer_single = _inf.infer_single
    infer_dynamic_batch = _inf.infer_dynamic_batch
    DEVICE = _inf.DEVICE
    MAX_NEW_TOKENS = _inf.MAX_NEW_TOKENS

    kv_pool_env = os.environ.get("ZLLM_KV_POOL_GB", "")
    print(
        f"[INFO] benchmark backend: {args.backend} ({_inf.__name__}) | "
        f"paged_kv={args.paged_kv} | infer_mode={args.infer_mode} | "
        f"dynamic_policy={args.dynamic_policy if args.infer_mode == 'dynamic' else 'N/A'} | "
        f"KV池 ZLLM_KV_POOL_GB={kv_pool_env or '(默认)'}"
    )

    # 加载模型（默认关闭 Paged，单条推理走 HF 原生 past，延迟低）
    tokenizer, model = load_model(args.model_path, use_paged_kv=args.paged_kv)

    weight_dtype = str(next(model.parameters()).dtype)

    bench_meta = {
        "infer_mode": args.infer_mode,
        "dynamic_policy": args.dynamic_policy if args.infer_mode == "dynamic" else None,
        "backend": args.backend,
        "use_paged_kv": args.paged_kv,
        "inference_module": _inf.__name__,
        "weight_dtype": weight_dtype,
        "kv_pool_gb": kv_pool_env or None,
        "model_path": args.model_path,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        bench_meta["cuda_version"] = torch.version.cuda

    # 加载 prompt
    prompts = load_prompts(args.prompt_file)

    # 运行测试
    stats = run_benchmark(
        tokenizer,
        model,
        prompts,
        meta=bench_meta,
        infer_mode=args.infer_mode,
        dynamic_policy=args.dynamic_policy,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        aging_rate=args.aging_rate,
        max_starve_ms=args.max_starve_ms,
    )

    # 打印结果
    print_stats(stats)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")
