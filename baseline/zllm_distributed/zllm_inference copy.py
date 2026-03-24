import argparse
import json
import time
import math
from optimizations.kv_cache import PagedKVCacheManager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimizations.memory_optimizer import MemoryOptimizer
from optimizations.dynamic_batch_scheduler import DynamicBatchScheduler, ContinuousBatchScheduler


DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


KV_CACHE = None
MEMORY_OPTIMIZER = MemoryOptimizer(clear_interval=8, force_gc=True, cuda_empty_cache=True)
MEMORY_OPTIMIZER.try_set_per_process_memory_fraction(0.9)


def _build_kv_cache(page_size_tokens: int = 64):
	return PagedKVCacheManager(
		max_entries=256,
		max_cache_tokens=786432,
		ttl_seconds=7200,
		page_size_tokens=page_size_tokens,
		enable_prefix_sharing=False,
	)


def _ensure_kv_cache(page_size_tokens: int = 64):
	global KV_CACHE
	if KV_CACHE is None:
		KV_CACHE = _build_kv_cache(page_size_tokens=page_size_tokens)
	return KV_CACHE

def _to_device(inputs: dict, device: str) -> dict:
	return {k: v.to(device) for k, v in inputs.items()}


def _safe_cuda_sync():
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def _safe_cuda_empty_cache():
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def load_model(model_path: str):
	print(f"[INFO] 加载模型: {model_path}")
	print(f"[INFO] 设备: {DEVICE} | 数据类型: {DTYPE}")

	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		trust_remote_code=True,
		padding_side="left",
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		torch_dtype=DTYPE,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
		device_map=DEVICE,
	)
	model.eval()

	n_params = sum(p.numel() for p in model.parameters()) / 1e9
	mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
	print(f"[INFO] 加载完成 | 参数量: {n_params:.2f}B | 显存占用: {mem_gb:.2f} GB")
	return tokenizer, model


def infer_single(tokenizer, model, prompt: str, use_kv_cache: bool = True) -> dict:
	cache = _ensure_kv_cache()
	cache_hit = False
	payload = None
	cached_output_text = None
	cached_output_tokens = None
	page_count = None
	prefix_reuse_pages = None
	if use_kv_cache:
		payload = cache.get_sequence(prompt)
		if payload is not None:
			cached_output_text = payload.get("cached_output_text") if isinstance(payload, dict) else None
			cached_output_tokens = payload.get("cached_output_tokens") if isinstance(payload, dict) else None
			page_count = payload.get("cached_page_count") if isinstance(payload, dict) else None
	MEMORY_OPTIMIZER.before_infer()

	did_compute = False
	t_start = time.perf_counter()

	with torch.inference_mode():
		if payload is not None and cached_output_text is not None and cached_output_tokens is not None:
			cache_hit = True
			input_len = int(payload["input_len"])
			output_len = int(cached_output_tokens)
			output_text = cached_output_text
		elif payload is not None:
			cache_hit = True
			did_compute = True
			_safe_cuda_sync()
			output_ids = model.generate(
				input_ids=payload["input_ids"],
				attention_mask=payload["attention_mask"],
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=False,
				use_cache=True,
				pad_token_id=tokenizer.pad_token_id,
			)
			input_len = payload["input_len"]
			generated = output_ids[0][input_len:]
			output_len = int(generated.shape[0])
			output_text = tokenizer.decode(generated, skip_special_tokens=True)
			cached_page_count = int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None
			if use_kv_cache:
				cache.put_sequence(
					prompt,
					payload={
						"input_ids": payload["input_ids"],
						"attention_mask": payload["attention_mask"],
						"input_len": input_len,
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": cached_page_count,
					},
					token_count=input_len,
				)
		else:
			did_compute = True
			inputs = tokenizer(prompt, return_tensors="pt")
			inputs = _to_device(inputs, DEVICE)
			input_len = inputs["input_ids"].shape[1]

			_safe_cuda_sync()
			output_ids = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=False,
				use_cache=True,
				pad_token_id=tokenizer.pad_token_id,
			)
			generated = output_ids[0][input_len:]
			output_len = int(generated.shape[0])
			output_text = tokenizer.decode(generated, skip_special_tokens=True)
			cached_page_count = int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None

			if use_kv_cache:
				cache.put_sequence(
					prompt,
					payload={
						"input_ids": inputs["input_ids"],
						"attention_mask": inputs["attention_mask"],
						"input_len": input_len,
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": cached_page_count,
					},
					token_count=input_len,
				)
				page_count = cached_page_count

	if did_compute:
		_safe_cuda_sync()
	t_end = time.perf_counter()

	if use_kv_cache:
		prefix_reuse_pages = cache.stats().get("prefix_reuse_pages")
	total_ms = (t_end - t_start) * 1000
	ttft_approx = total_ms
	throughput = (output_len / total_ms * 1000) if total_ms > 0 else 0.0

	MEMORY_OPTIMIZER.after_infer()

	return {
		"prompt": prompt,
		"output": output_text,
		"input_tokens": input_len,
		"output_tokens": output_len,
		"total_latency_ms": round(total_ms, 2),
		"ttft_ms": round(ttft_approx, 2),
		"throughput_tps": round(throughput, 2),
		"kv_cache_hit": cache_hit,
		"kv_cache_pages": page_count,
		"kv_prefix_reuse_pages": prefix_reuse_pages,
		"kv_cache_stats": cache.stats() if use_kv_cache else {},
	}



def infer_dynamic_batch(tokenizer, model, prompts: list[str], max_batch_size: int = 4, max_wait_ms: int = 20, max_prompt_tokens: int = 4096) -> list[dict]:
	cache = _ensure_kv_cache()
	scheduler = DynamicBatchScheduler(max_batch_size=max_batch_size, max_wait_ms=max_wait_ms, max_prompt_tokens=max_prompt_tokens)
	results = []
	result_slots = [None] * len(prompts)

	def _cached_result(prompt: str, payload: dict):
		input_len = int(payload["input_len"])
		output_text = payload.get("cached_output_text")
		output_len = payload.get("cached_output_tokens")
		if output_text is None or output_len is None:
			return None
		return {
			"prompt": prompt,
			"output": output_text,
			"input_tokens": input_len,
			"output_tokens": int(output_len),
			"total_latency_ms": 0.0,
			"ttft_ms": 0.0,
			"throughput_tps": 0.0,
			"batch_size": 0,
		}
	# 预处理：统计每个prompt的token数
	for idx, prompt in enumerate(prompts):
		payload = cache.get_sequence(prompt)
		cached = _cached_result(prompt, payload) if payload is not None else None
		if cached is not None:
			result_slots[idx] = cached
			continue
		scheduler.add_request(idx, prompt, max(1, len(prompt) // 2))

	# 主循环：不断poll batch直到全部发车
	while True:
		batch = scheduler.poll_batch()
		if not batch:
			break
		batch_prompts = [item["prompt"] for item in batch]
		MEMORY_OPTIMIZER.before_infer()
		inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
		inputs = _to_device(inputs, DEVICE)

		_safe_cuda_sync()
		t_start = time.perf_counter()

		with torch.inference_mode():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=False,
				use_cache=True,
				pad_token_id=tokenizer.pad_token_id,
			)

		_safe_cuda_sync()
		t_end = time.perf_counter()

		total_batch_ms = (t_end - t_start) * 1000
		input_lens = inputs["attention_mask"].sum(dim=1).tolist()

		for idx, item in enumerate(batch):
			prompt = item["prompt"]
			request_id = int(item["request_id"])
			input_len = int(input_lens[idx])
			generated = output_ids[idx][input_len:]
			output_text = tokenizer.decode(generated, skip_special_tokens=True)
			output_len = int(generated.shape[0])
			throughput = (output_len / total_batch_ms * 1000) if total_batch_ms > 0 else 0.0

			result_slots[request_id] = {
				"prompt": prompt,
				"output": output_text,
				"input_tokens": input_len,
				"output_tokens": output_len,
				"total_latency_ms": round(total_batch_ms, 2),
				"ttft_ms": round(total_batch_ms, 2),
				"throughput_tps": round(throughput, 2),
				"batch_size": len(batch_prompts),
			}

			cache.put_sequence(
				prompt,
				payload={
					"input_ids": inputs["input_ids"][idx:idx + 1],
					"attention_mask": inputs["attention_mask"][idx:idx + 1],
					"input_len": input_len,
					"cached_output_text": output_text,
					"cached_output_tokens": output_len,
					"cached_page_count": int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None,
				},
				token_count=input_len,
			)

		MEMORY_OPTIMIZER.after_infer()

	# 最后flush所有剩余请求
	for batch in scheduler.flush_all():
		if not batch:
			continue
		batch_prompts = [item["prompt"] for item in batch]
		MEMORY_OPTIMIZER.before_infer()
		inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
		inputs = _to_device(inputs, DEVICE)

		_safe_cuda_sync()
		t_start = time.perf_counter()

		with torch.inference_mode():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=False,
				use_cache=True,
				pad_token_id=tokenizer.pad_token_id,
			)

		_safe_cuda_sync()
		t_end = time.perf_counter()

		total_batch_ms = (t_end - t_start) * 1000
		input_lens = inputs["attention_mask"].sum(dim=1).tolist()

		for idx, item in enumerate(batch):
			prompt = item["prompt"]
			request_id = int(item["request_id"])
			input_len = int(input_lens[idx])
			generated = output_ids[idx][input_len:]
			output_text = tokenizer.decode(generated, skip_special_tokens=True)
			output_len = int(generated.shape[0])
			throughput = (output_len / total_batch_ms * 1000) if total_batch_ms > 0 else 0.0

			result_slots[request_id] = {
				"prompt": prompt,
				"output": output_text,
				"input_tokens": input_len,
				"output_tokens": output_len,
				"total_latency_ms": round(total_batch_ms, 2),
				"ttft_ms": round(total_batch_ms, 2),
				"throughput_tps": round(throughput, 2),
				"batch_size": len(batch_prompts),
			}

			cache.put_sequence(
				prompt,
				payload={
					"input_ids": inputs["input_ids"][idx:idx + 1],
					"attention_mask": inputs["attention_mask"][idx:idx + 1],
					"input_len": input_len,
					"cached_output_text": output_text,
					"cached_output_tokens": output_len,
					"cached_page_count": int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None,
				},
				token_count=input_len,
			)

		MEMORY_OPTIMIZER.after_infer()

	for item in result_slots:
		if item is not None:
			results.append(item)

	return results


def infer_continuous_batch(
	tokenizer, model, prompts: list[str],
	max_batch_size: int = 8, max_new_tokens: int = None,
) -> list[dict]:
	"""Continuous Batching 推理 —— iteration-level 调度。

	与 infer_dynamic_batch 的关键区别：
	  - 不使用 model.generate()，而是手动执行逐 token 的 forward pass
	  - 每个 decode step 后，完成的请求立即驱逐，空闲 slot 立即填入新请求
	  - 消除 batch bubble，提升 GPU 利用率
	"""
	if max_new_tokens is None:
		max_new_tokens = MAX_NEW_TOKENS

	cache = _ensure_kv_cache()
	scheduler = ContinuousBatchScheduler(
		max_batch_size=max_batch_size,
		max_new_tokens=max_new_tokens,
	)

	result_slots = [None] * len(prompts)
	request_states = {}

	for idx, prompt in enumerate(prompts):
		payload = cache.get_sequence(prompt)
		if payload is not None:
			output_text = payload.get("cached_output_text")
			output_tokens = payload.get("cached_output_tokens")
			if output_text is not None and output_tokens is not None:
				result_slots[idx] = {
					"prompt": prompt,
					"output": output_text,
					"input_tokens": int(payload["input_len"]),
					"output_tokens": int(output_tokens),
					"total_latency_ms": 0.0,
					"ttft_ms": 0.0,
					"throughput_tps": 0.0,
					"batch_size": 0,
					"mode": "continuous_batch",
				}
				continue

		tokens = tokenizer(prompt, return_tensors="pt")
		prompt_len = tokens["input_ids"].shape[1]
		scheduler.add_request(idx, prompt, prompt_len)

	step_count = 0
	while scheduler.has_unfinished():
		new_reqs, active_reqs = scheduler.schedule()
		new_rid_set = {r["request_id"] for r in new_reqs}

		MEMORY_OPTIMIZER.before_infer()

		for req in new_reqs:
			rid = req["request_id"]
			prompt = req["prompt"]

			inputs = tokenizer(prompt, return_tensors="pt")
			inputs = _to_device(inputs, DEVICE)
			input_len = inputs["input_ids"].shape[1]

			_safe_cuda_sync()
			t_start = time.perf_counter()

			with torch.inference_mode():
				outputs = model(
					input_ids=inputs["input_ids"],
					attention_mask=inputs["attention_mask"],
					use_cache=True,
				)

			_safe_cuda_sync()
			t_prefill = time.perf_counter()

			next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

			request_states[rid] = {
				"past_key_values": outputs.past_key_values,
				"generated_ids": [next_token_id],
				"input_len": input_len,
				"t_start": t_start,
				"ttft_ms": (t_prefill - t_start) * 1000,
			}
			scheduler.mark_decoding(rid)

		for req in active_reqs:
			rid = req["request_id"]
			if rid in new_rid_set:
				continue

			state = request_states.get(rid)
			if state is None:
				continue

			last_token = state["generated_ids"][-1]
			next_input = torch.tensor([[last_token]], device=DEVICE, dtype=torch.long)

			with torch.inference_mode():
				outputs = model(
					input_ids=next_input,
					past_key_values=state["past_key_values"],
					use_cache=True,
				)

			next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
			state["past_key_values"] = outputs.past_key_values
			state["generated_ids"].append(next_token_id)

		finished_ids = []
		for req in active_reqs:
			rid = req["request_id"]
			state = request_states.get(rid)
			if state is None:
				continue
			gen_ids = state["generated_ids"]
			if gen_ids[-1] == tokenizer.eos_token_id or len(gen_ids) >= max_new_tokens:
				finished_ids.append(rid)

				_safe_cuda_sync()
				t_end = time.perf_counter()
				total_ms = (t_end - state["t_start"]) * 1000
				output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
				output_len = len(gen_ids)
				throughput = (output_len / total_ms * 1000) if total_ms > 0 else 0.0

				result_slots[rid] = {
					"prompt": req["prompt"],
					"output": output_text,
					"input_tokens": state["input_len"],
					"output_tokens": output_len,
					"total_latency_ms": round(total_ms, 2),
					"ttft_ms": round(state["ttft_ms"], 2),
					"throughput_tps": round(throughput, 2),
					"batch_size": len(active_reqs),
					"mode": "continuous_batch",
				}

				cache.put_sequence(
					req["prompt"],
					payload={
						"input_len": state["input_len"],
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": int(math.ceil(float(state["input_len"]) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None,
					},
					token_count=state["input_len"],
				)

				del state["past_key_values"]
				del request_states[rid]

		if finished_ids:
			scheduler.finish_requests(finished_ids)
			_safe_cuda_empty_cache()

		MEMORY_OPTIMIZER.after_infer()
		step_count += 1

		if step_count % 50 == 0:
			s = scheduler.stats()
			print(
				f"[ContinuousBatch] step={step_count} "
				f"running={s['running']} waiting={s['waiting']} "
				f"finished={s['total_finished']}/{len(prompts)}"
			)

	print(f"[ContinuousBatch] 完成，共 {step_count} 个 decode step，{scheduler.stats()['total_finished']} 条请求")

	results = [item for item in result_slots if item is not None]
	return results


def _load_prompts(prompt_file: str) -> list[str]:
	prompts = []
	with open(prompt_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			prompts.append(obj["prompt"] if isinstance(obj, dict) else str(obj))
	return prompts


def parse_args():
	parser = argparse.ArgumentParser(description="zLLM 推理脚本（等效 baseline + 简单优化）")
	parser.add_argument("--model_path", type=str, required=True, help="模型本地路径")
	parser.add_argument(
		"--prompt",
		type=str,
		default="请用三句话解释大语言模型推理中KV Cache的作用。",
		help="单条测试 prompt",
	)
	parser.add_argument("--mode", type=str, default="single", choices=["single", "batch", "continuous"], help="运行模式: single=单条, batch=动态Batch, continuous=Continuous Batch")
	parser.add_argument("--prompt_file", type=str, default=str(Path(__file__).resolve().parent.parent / "prompts.jsonl"), help="batch模式下的prompt文件")
	parser.add_argument("--max_batch_size", type=int, default=4, help="batch模式下的动态批大小")
	parser.add_argument("--kv_page_size", type=int, default=16, help="paged后端的每页token数")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	KV_CACHE = _build_kv_cache(page_size_tokens=args.kv_page_size)
	tokenizer, model = load_model(args.model_path)
	MEMORY_OPTIMIZER.reset_peak()

	if args.mode == "single":
		print(f"\n[推理] prompt: {args.prompt}")
		result = infer_single(tokenizer, model, args.prompt)

		print("\n" + "=" * 64)
		print(" 推理结果")
		print("=" * 64)
		print(f"  输入   : {result['prompt']}")
		print(f"  输出   : {result['output']}")
		print("-" * 64)
		print(f"  输入 tokens   : {result['input_tokens']}")
		print(f"  输出 tokens   : {result['output_tokens']}")
		print(f"  总延迟         : {result['total_latency_ms']} ms")
		print(f"  TTFT (近似)   : {result['ttft_ms']} ms")
		print(f"  吞吐率         : {result['throughput_tps']} tokens/sec")
		print(f"  KV Cache命中   : {result['kv_cache_hit']}")
		if result.get("kv_cache_pages") is not None:
			print(f"  KV Page数      : {result['kv_cache_pages']}")
		if result.get("kv_prefix_reuse_pages") is not None:
			print(f"  KV前缀复用页   : {result['kv_prefix_reuse_pages']}")
		print("  KV后端         : paged")
		if torch.cuda.is_available():
			print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
		print("=" * 64)
	else:
		prompts = _load_prompts(args.prompt_file)
		results = infer_dynamic_batch(tokenizer, model, prompts, max_batch_size=args.max_batch_size)
		print(f"[INFO] 动态Batch完成，共处理 {len(results)} 条请求")
		if results:
			avg_latency = sum(x["total_latency_ms"] for x in results) / len(results)
			avg_tps = sum(x["throughput_tps"] for x in results) / len(results)
			print(f"[INFO] 平均延迟: {avg_latency:.2f} ms")
			print(f"[INFO] 平均吞吐: {avg_tps:.2f} tokens/sec")

	mem = MEMORY_OPTIMIZER.memory_snapshot()
	if mem.get("cuda", False):
		print(
			f"[INFO] 显存快照 | allocated={mem['allocated_gb']:.3f} GB "
			f"reserved={mem['reserved_gb']:.3f} GB "
			f"max_allocated={mem['max_allocated_gb']:.3f} GB"
		)
