import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from distributed import TensorParallelConfig, TensorParallelLauncher


DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256


@dataclass
class _TPModelHandle:
	model_path: str
	max_new_tokens: int = MAX_NEW_TOKENS
	launcher: TensorParallelLauncher | None = None


def load_model(model_path: str):
	print(f"[INFO] Tensor Parallel 模型句柄初始化: {model_path}")
	handle = _TPModelHandle(model_path=model_path)
	return None, handle


def infer_single(tokenizer, model, prompt: str) -> dict:
	"""与 benchmark.py 对齐的单条推理接口。

	参数 tokenizer 保留仅为兼容；实际推理基于 TensorParallelLauncher。
	"""
	if not isinstance(model, _TPModelHandle):
		raise TypeError("model 需要由 load_model() 返回")
	if model.launcher is None:
		model.launcher = TensorParallelLauncher(
			TensorParallelConfig(
				model_path=model.model_path,
				prompts=[],
				max_new_tokens=int(model.max_new_tokens),
				dtype="float16" if DTYPE == torch.float16 else str(DTYPE),
			)
		)

	t_start = time.perf_counter()
	results = model.launcher.run_sync(prompts=[prompt], max_new_tokens=int(model.max_new_tokens))
	t_end = time.perf_counter()

	if not results:
		total_ms = (t_end - t_start) * 1000
		return {
			"prompt": prompt,
			"output": "",
			"input_tokens": 0,
			"output_tokens": 0,
			"total_latency_ms": round(total_ms, 2),
			"ttft_ms": round(total_ms, 2),
			"throughput_tps": 0.0,
		}
	return results[0]


def _load_prompts(prompt_file: str) -> list[str]:
	prompts: list[str] = []
	with open(prompt_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			prompts.append(obj["prompt"] if isinstance(obj, dict) else str(obj))
	return prompts

def parse_args():
	parser = argparse.ArgumentParser(description="仅4卡张量并行推理脚本")
	parser.add_argument("--model_path", type=str, required=True, help="模型本地路径")
	parser.add_argument(
		"--mode",
		type=str,
		default="tensor_parallel",
		choices=["single", "tensor_parallel"],
		help="运行模式：single=单条输入，tensor_parallel=按prompt_file批量输入",
	)
	parser.add_argument("--prompt", type=str, default="请解释张量并行在大模型推理中的作用。", help="单条 prompt")
	parser.add_argument(
		"--prompt_file",
		type=str,
		default=str(Path(__file__).resolve().parent / "prompts.jsonl"),
		help="prompt 文件路径（jsonl）",
	)
	parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="最大生成 token 数")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	if args.mode == "single":
		prompts = [args.prompt]
	elif args.prompt_file and Path(args.prompt_file).exists():
		prompts = _load_prompts(args.prompt_file)
	else:
		prompts = [args.prompt]

	print(
		f"\n[TensorParallel] 启动张量并行推理 | "
		f"prompts={len(prompts)}"
	)

	launcher = TensorParallelLauncher(
		TensorParallelConfig(
			model_path=args.model_path,
			prompts=prompts,
			max_new_tokens=args.max_new_tokens,
			dtype="float16" if DTYPE == torch.float16 else str(DTYPE),
		)
	)
	results = launcher.run()

	print("\n" + "=" * 64)
	print(" Tensor Parallel 结果汇总")
	print("=" * 64)
	print(f"  请求总数       : {len(results)}")
	if results:
		avg_latency = sum(r["total_latency_ms"] for r in results) / len(results)
		avg_tps = sum(r["throughput_tps"] for r in results) / len(results)
		total_out = sum(r["output_tokens"] for r in results)
		if args.mode == "single":
			first = results[0]
			print(f"  输入           : {first.get('prompt', '')}")
			print(f"  输出           : {first.get('output', '')}")
		print(f"  总输出 tokens  : {total_out}")
		print(f"  平均延迟       : {avg_latency:.2f} ms")
		print(f"  平均吞吐率     : {avg_tps:.2f} tokens/sec")
	if torch.cuda.is_available():
		print(f"  可见GPU数量    : {torch.cuda.device_count()}")
	print("=" * 64)
