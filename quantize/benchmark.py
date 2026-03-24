#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py
============
吞吐量 & 延迟基准测试脚本（含困惑度测试）

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
from baseline_inference import load_model, infer_single, DEVICE, MAX_NEW_TOKENS

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"
DEFAULT_WIKITEXT_FILE = Path(__file__).parent / "test.txt"

# 困惑度计算固定参数
PPL_MAX_LENGTH = 1024  # 固定最大序列长度

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


def load_wikitext_from_file(file_path: str) -> list:
    """
    从本地文件读取 wikitext 数据
    文件格式：每段文本以引号包围，类似 " text "
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 按引号分割，提取文本内容
    import re
    # 匹配引号包围的文本
    pattern = r'"\s*\n(.*?)\n\s*"'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        # 清理文本：去除首尾空白，合并多余换行
        text = match.strip()
        if text:  # 只保留非空文本
            texts.append(text)
    
    print(f"[INFO] 从 {file_path} 加载了 {len(texts)} 个文本片段")
    return texts


def compute_perplexity(tokenizer, model, texts: list) -> dict:
    """
    计算困惑度（Perplexity）
    直接截断到固定长度
    
    Args:
        tokenizer: 分词器
        model: 模型
        texts: 文本列表
    
    Returns:
        困惑度统计信息
    """
    max_length = PPL_MAX_LENGTH
    
    print(f"\n[Perplexity] 开始计算困惑度...")
    print(f"[Perplexity] 文本数量={len(texts)}, 最大长度={max_length}")
    print("-" * 68)
    
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_text_ppl = []
    
    with torch.no_grad():
        for idx, text in enumerate(texts):
            # 编码文本并截断
            encodings = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=False
            )
            seq_len = encodings.input_ids.size(1)
            
            if seq_len < 2:
                print(f"  [{idx+1:4d}/{len(texts)}] 跳过: 文本长度不足")
                continue
            
            input_ids = encodings.input_ids.to(DEVICE)
            attention_mask = encodings.attention_mask.to(DEVICE)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失（排除第一个token）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            nll = loss.item()
            token_count = shift_labels.numel()
            
            if token_count > 0:
                text_ppl = np.exp(nll / token_count)
                per_text_ppl.append(text_ppl)
                total_nll += nll
                total_tokens += token_count
                
                # 打印进度
                if (idx + 1) % 5 == 0 or idx == len(texts) - 1:
                    avg_ppl_so_far = np.exp(total_nll / total_tokens) if total_tokens > 0 else 0
                    print(f"  [{idx+1:4d}/{len(texts)}] "
                          f"tokens={token_count:5d}, "
                          f"ppl={text_ppl:8.2f}, "
                          f"avg_ppl={avg_ppl_so_far:8.2f}")
    
    # 计算平均困惑度
    if total_tokens > 0:
        avg_nll = total_nll / total_tokens
        avg_ppl = np.exp(avg_nll)
    else:
        avg_ppl = float('inf')
        avg_nll = float('inf')
    
    # 立即打印平均困惑度结果
    print("\n" + "=" * 68)
    print(" 【困惑度测试结果】")
    print("=" * 68)
    print(f"  平均困惑度 (PPL)         : {avg_ppl:.3f}")
    print(f"  平均负对数似然 (NLL)     : {avg_nll:.3f}")
    print(f"  处理的总 token 数        : {total_tokens}")
    print(f"  测试文本数               : {len(per_text_ppl)}")
    
    # 提供参考值
    print(f"\n  参考值：")
    print(f"    - GPT-2 (124M): ~45.0")
    print(f"    - GPT-2 (1.5B): ~29.0")
    print(f"    - LLaMA-7B: ~20-25")
    print(f"    - Qwen2.5-7B: ~15-20")
    
    if avg_ppl < 30:
        print(f"\n  ✓ 困惑度表现优秀 (<30)，模型在英文文本上的预测能力很强")
    elif avg_ppl < 50:
        print(f"\n  ✓ 困惑度表现良好 (30-50)，模型在英文文本上的预测能力较好")
    elif avg_ppl < 100:
        print(f"\n  ○ 困惑度表现一般 (50-100)，模型在英文文本上的预测能力中等")
    else:
        print(f"\n  △ 困惑度偏高 (>100)，可能需要检查量化精度或模型配置")
    print("=" * 68)
    
    # 统计信息
    if per_text_ppl:
        stats = {
            "dataset": "wikitext-local",
            "total_texts": len(per_text_ppl),
            "total_tokens_processed": total_tokens,
            "avg_nll": round(avg_nll, 3),
            "avg_perplexity": round(avg_ppl, 3),
            "perplexity_std": round(np.std(per_text_ppl), 3),
            "perplexity_min": round(np.min(per_text_ppl), 3),
            "perplexity_max": round(np.max(per_text_ppl), 3),
            "perplexity_median": round(np.median(per_text_ppl), 3),
            "max_length": max_length,
        }
    else:
        stats = {
            "dataset": "wikitext-local",
            "total_texts": 0,
            "total_tokens_processed": 0,
            "avg_nll": 0,
            "avg_perplexity": float('inf'),
            "perplexity_std": 0,
            "perplexity_min": 0,
            "perplexity_max": 0,
            "perplexity_median": 0,
            "max_length": max_length,
        }
    
    return stats


def run_benchmark(tokenizer, model, prompts: list) -> dict:
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理...")
    print("-" * 68)

    latencies   = []
    ttfts       = []
    total_out   = 0

    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_wall_start = time.perf_counter()

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
    parser = argparse.ArgumentParser(description="LLM 推理吞吐 & 延迟基准测试（含困惑度）")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /data/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE),
        help="prompt 文件路径（默认：prompts.jsonl）"
    )
    parser.add_argument(
        "--wikitext_file", type=str, default=str(DEFAULT_WIKITEXT_FILE),
        help="WikiText 数据文件路径（默认：test.txt）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 results_baseline.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载模型
    tokenizer, model = load_model(args.model_path)
    
    # 设置模型为评估模式
    model.eval()
    
    results = {}
    
    # 1. 加载本地 WikiText 数据并计算困惑度（立即显示结果）
    wikitext_file = Path(args.wikitext_file)
    if wikitext_file.exists():
        wikitext_texts = load_wikitext_from_file(str(wikitext_file))
        if wikitext_texts:
            perplexity_stats = compute_perplexity(
                tokenizer, model, wikitext_texts
            )
            results["perplexity"] = perplexity_stats
        else:
            print(f"[WARNING] 从 {args.wikitext_file} 没有加载到任何文本")
    else:
        print(f"[WARNING] WikiText 文件不存在: {args.wikitext_file}")
        print(f"[INFO] 跳过困惑度测试")

    # 2. 加载 prompt
    prompts = load_prompts(args.prompt_file)

    # 3. 运行吞吐量和延迟测试
    stats = run_benchmark(tokenizer, model, prompts)
    results["benchmark"] = stats

    # 4. 打印吞吐量结果
    print_stats(stats)
    
    # 5. 添加配置信息
    results["config"] = {
        "model_path": args.model_path,
        "prompt_file": args.prompt_file,
        "wikitext_file": args.wikitext_file,
        "perplexity_max_length": PPL_MAX_LENGTH,
    }

    # 6. 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")
