#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_accuracy_vllm.py
====================
使用 vLLM 进行精度评测脚本（流式输出版本）

使用方式：
  # 评测 vLLM 模型
  python evaluate_accuracy_vllm.py --model_path /path/to/model --eval_file ceval_subset.jsonl
  # 流式输出
  python evaluate_accuracy_vllm.py --model_path /path/to/model --stream
"""

import argparse
import json
import time
import os
from pathlib import Path
import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer

DEFAULT_EVAL_FILE   = Path(__file__).parent / "ceval_subset.jsonl"
ACCURACY_DROP_LIMIT = 0.05   # 精度损失上限（绝对值）


def load_eval_data(eval_file: str) -> list:
    """加载 C-Eval 格式数据集"""
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[INFO] 已加载 {len(data)} 道评测题（来自 {eval_file}）")
    return data[:100]


def build_prompt(item: dict) -> str:
    """构建评测 prompt（与 evaluate_accuracy.py 保持一致）"""
    return (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )


def build_chat_prompt(tokenizer, item: dict) -> str:
    """使用 chat_template 构建 prompt（参考 eva_vllm.py）"""
    content = (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )
    messages = [{"role": "user", "content": content}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def extract_answer(output: str) -> str:
    """提取答案（与 evaluate_accuracy.py 保持一致）"""
    for ch in output.strip():
        if ch.upper() in ("A", "B", "C", "D"):
            return ch.upper()
    return "X"   # 未能解析时标记为错误


async def generate_stream_async(engine, prompt, sampling_params, request_id):
    """
    异步流式生成（使用 AsyncLLMEngine）
    """
    full_text = ""
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    async for request_output in results_generator:
        for output in request_output.outputs:
            new_text = output.text
            print(new_text, end="", flush=True)
            full_text += new_text
    
    return full_text


def run_nonstreaming_eval(llm, tokenizer, eval_data: list, args) -> dict:
    """
    非流式模式：使用 vLLM LLM 逐题推理并统计准确率。
    """
    print(f"\n[Accuracy] 开始精度评测（非流式模式），共 {len(eval_data)} 道题...")
    print("-" * 60)

    correct      = 0
    wrong_cases  = []
    t_start      = time.perf_counter()

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    for i, item in enumerate(eval_data):
        # 准备 prompt
        if args.use_chat_template:
            prompt = build_chat_prompt(tokenizer, item)
        else:
            prompt = build_prompt(item)

        # 非流式生成
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # 提取答案并比对
        pred        = extract_answer(generated_text)
        gold        = item["answer"].upper()
        is_correct  = (pred == gold)

        if is_correct:
            correct += 1
        else:
            wrong_cases.append({
                "id":       item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred":     pred,
                "gold":     gold,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率：{acc_so_far*100:.1f}%  "
                f"正确：{correct}  错误：{i+1-correct}"
            )
            print("-" * 60)

    t_end    = time.perf_counter()
    accuracy = correct / len(eval_data)

    result = {
        "total":        len(eval_data),
        "correct":      correct,
        "wrong":        len(eval_data) - correct,
        "accuracy":     round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases":    wrong_cases[:10],   # 最多展示前 10 个错误案例
    }
    return result


async def run_streaming_eval(engine, tokenizer, eval_data: list, args) -> dict:
    """
    流式模式：使用 AsyncLLMEngine 进行异步流式评测
    """
    print(f"\n[Accuracy] 开始精度评测（流式模式），共 {len(eval_data)} 道题...")
    print("-" * 60)

    correct      = 0
    wrong_cases  = []
    t_start      = time.perf_counter()

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    for i, item in enumerate(eval_data):
    
        if args.use_chat_template:
            prompt = build_chat_prompt(tokenizer, item)
        else:
            prompt = build_prompt(item)

 
        # print(f"\n[题目 {i+1}] ", end="", flush=True)
        request_id = f"request_{i}"
        generated_text = await generate_stream_async(engine, prompt, sampling_params, request_id)

        pred        = extract_answer(generated_text)
        gold        = item["answer"].upper()
        is_correct  = (pred == gold)
      
        if is_correct:
            correct += 1
            print(f"  ✓ 正确 (预测={pred}, 答案={gold})")
        else:
            wrong_cases.append({
                "id":       item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred":     pred,
                "gold":     gold,
            })
            print(f"  ✗ 错误 (预测={pred}, 答案={gold})")

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"\n  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率：{acc_so_far*100:.1f}%  "
                f"正确：{correct}  错误：{i+1-correct}"
            )
            print("-" * 60)

    t_end    = time.perf_counter()
    accuracy = correct / len(eval_data)

    result = {
        "total":        len(eval_data),
        "correct":      correct,
        "wrong":        len(eval_data) - correct,
        "accuracy":     round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases":    wrong_cases[:10],   # 最多展示前 10 个错误案例
    }
    return result


def print_accuracy_result(result: dict, baseline_acc: float = None):
    """格式化打印精度评测结果（与 evaluate_accuracy.py 保持一致）"""
    print("\n" + "=" * 60)
    print(" 精度评测结果")
    print("=" * 60)
    print(f"  总题数       : {result['total']}")
    print(f"  答对题数     : {result['correct']}")
    print(f"  答错题数     : {result['wrong']}")
    print(f"  准确率       : {result['accuracy_pct']:.2f}%")
    print(f"  评测耗时     : {result['eval_time_sec']} sec")

    if baseline_acc is not None:
        drop = baseline_acc - result["accuracy"]
        status = "达标" if drop <= ACCURACY_DROP_LIMIT else "超标（扣分）"
        print("-" * 60)
        print(f"  基线准确率   : {baseline_acc*100:.2f}%")
        print(f"  精度下降     : {drop*100:.2f}% （上限 {ACCURACY_DROP_LIMIT*100:.0f}%）")
        print(f"  精度约束状态 : {status}")
        if drop > ACCURACY_DROP_LIMIT:
            print("  [警告] 精度损失超过阈值，「优化效果」评分将扣 50%！")

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="C-Eval 精度评测 (vLLM)")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径"
    )
    parser.add_argument(
        "--eval_file", type=str, default=str(DEFAULT_EVAL_FILE),
        help="评测数据集路径（默认：ceval_subset.jsonl）"
    )
    parser.add_argument(
        "--baseline_acc", type=float, default=None,
        help="基线准确率（0~1），用于判断精度是否达标"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON）"
    )
    # vLLM 相关参数
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--use_chat_template", action="store_true", help="使用 chat_template 构建 prompt")
    parser.add_argument("--quantization", type=str, default=None, help="量化方式，如 AWQ")
    # 流式输出参数
    parser.add_argument("--stream", action="store_true", help="启用流式输出（真正的流式输出）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 模型路径不存在：{args.model_path}")
        exit(1)
    
    print(f"[INFO] 加载模型：{args.model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 加载评测数据
    eval_data = load_eval_data(args.eval_file)
    
    # 根据是否流式选择不同的执行方式
    if args.stream:
        # 流式模式：使用 AsyncLLMEngine
        print("[INFO] 初始化异步引擎（真正的流式模式）...")
        engine_args = AsyncEngineArgs(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            quantization=args.quantization,
            dtype="float16"
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 运行异步流式评测
        result = asyncio.run(run_streaming_eval(engine, tokenizer, eval_data, args))
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
            dtype="float16"
        )
        result = run_nonstreaming_eval(llm, tokenizer, eval_data, args)
    
    # 打印结果
    print_accuracy_result(result, args.baseline_acc)

    # 保存结果
    if args.output:
        out = {k: v for k, v in result.items() if k != "wrong_cases"}
        out["wrong_cases_count"] = result["wrong"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至：{args.output}")