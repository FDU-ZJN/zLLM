import argparse
import json
import time
import math
from pathlib import Path
from optimizations.kv_cache import PagedKVCacheManager

import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from optimizations.memory_optimizer import MemoryOptimizer
from optimizations.dynamic_batch_scheduler import ContinuousBatchScheduler


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
		enable_prefix_sharing=True,
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


def _infer_seq_dim(tensor: torch.Tensor, expected_seq_len: int | None = None) -> int:
	if tensor.dim() <= 1:
		return 0
	if expected_seq_len is not None and int(expected_seq_len) > 0:
		for dim in range(tensor.dim() - 1, 0, -1):
			if int(tensor.shape[dim]) == int(expected_seq_len):
				return dim
	if tensor.dim() >= 3:
		return tensor.dim() - 2
	return 1


def _slice_tensor_seq(tensor: torch.Tensor, start: int, end: int, seq_dim: int) -> torch.Tensor:
	slices = [slice(None)] * tensor.dim()
	slices[int(seq_dim)] = slice(int(start), int(end))
	return tensor[tuple(slices)].contiguous()


def _slice_past_key_values_by_range(past_key_values, start: int, end: int, prompt_len: int):
	if past_key_values is None:
		return None
	page_layers = []
	seq_dims = []
	for layer in past_key_values:
		if not isinstance(layer, (tuple, list)) or len(layer) < 2:
			return None
		key, value = layer[0], layer[1]
		key_seq_dim = _infer_seq_dim(key, expected_seq_len=prompt_len)
		value_seq_dim = _infer_seq_dim(value, expected_seq_len=prompt_len)
		page_layers.append(
			(
				_slice_tensor_seq(key, start, end, key_seq_dim),
				_slice_tensor_seq(value, start, end, value_seq_dim),
			)
		)
		seq_dims.append((int(key_seq_dim), int(value_seq_dim)))
	return {"past_key_values_page": tuple(page_layers), "seq_dims": tuple(seq_dims)}


def _build_kv_page_payload_builder(past_key_values, prompt_len: int):
	def _builder(_token_chunk, _page_idx, start, end):
		return _slice_past_key_values_by_range(
			past_key_values=past_key_values,
			start=int(start),
			end=int(end),
			prompt_len=int(prompt_len),
		)
	return _builder


def _restore_past_key_values_from_pages(sequence_pages):
	if not sequence_pages:
		return None
	page_payloads = []
	for page in sequence_pages:
		payload = page.get("payload") if isinstance(page, dict) else None
		if not isinstance(payload, dict):
			continue
		if payload.get("past_key_values_page") is None:
			continue
		if payload.get("seq_dims") is None:
			continue
		page_payloads.append(payload)
	if not page_payloads:
		return None

	first_layers = page_payloads[0]["past_key_values_page"]
	if not isinstance(first_layers, (tuple, list)) or not first_layers:
		return None

	n_layers = len(first_layers)
	seq_dims = page_payloads[0]["seq_dims"]
	merged_layers = []
	for layer_idx in range(n_layers):
		k_parts = []
		v_parts = []
		for payload in page_payloads:
			layer = payload["past_key_values_page"][layer_idx]
			k_parts.append(layer[0])
			v_parts.append(layer[1])
		k_dim, v_dim = seq_dims[layer_idx]
		merged_layers.append((torch.cat(k_parts, dim=int(k_dim)), torch.cat(v_parts, dim=int(v_dim))))
	return tuple(merged_layers)


def _greedy_decode_from_prefill(
	model,
	tokenizer,
	past_key_values,
	first_token_id,
	prompt_len,
	max_new_tokens,
	device,
):
	if max_new_tokens <= 0:
		empty = torch.empty((1, 0), dtype=torch.long, device=device)
		return empty

	next_token = torch.tensor([[int(first_token_id)]], dtype=torch.long, device=device)
	attention_mask = torch.ones((1, int(prompt_len)), dtype=torch.long, device=device)
	generated_tokens = []
	eos_id = tokenizer.eos_token_id
	past = past_key_values

	for _ in range(int(max_new_tokens)):
		generated_tokens.append(next_token)
		if eos_id is not None and int(next_token.item()) == int(eos_id):
			break

		attention_mask = torch.cat(
			[attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
			dim=1,
		)
		outputs = model(
			input_ids=next_token,
			attention_mask=attention_mask,
			past_key_values=past,
			use_cache=True,
			return_dict=True,
		)
		past = outputs.past_key_values
		next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

	if not generated_tokens:
		return torch.empty((1, 0), dtype=torch.long, device=device)
	return torch.cat(generated_tokens, dim=1)


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

	model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=False,   # 先关闭，确保 KV Cache 兼容
        trust_remote_code=True,
        device_map="auto",
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
	cached_prompt_past_key_values = None
	cached_first_token_id = None
	page_count = None
	prefix_reuse_pages = None
	if use_kv_cache:
		payload = cache.get_sequence(prompt)
		if payload is not None:
			cached_output_text = payload.get("cached_output_text") if isinstance(payload, dict) else None
			cached_output_tokens = payload.get("cached_output_tokens") if isinstance(payload, dict) else None
			cached_prompt_past_key_values = payload.get("cached_prompt_past_key_values") if isinstance(payload, dict) else None
			cached_first_token_id = payload.get("cached_first_token_id") if isinstance(payload, dict) else None
			page_count = payload.get("cached_page_count") if isinstance(payload, dict) else None
			if cached_prompt_past_key_values is None and cached_first_token_id is not None:
				sequence_pages = cache.get_sequence_pages(prompt, touch_stats=False)
				cached_prompt_past_key_values = _restore_past_key_values_from_pages(sequence_pages)
	MEMORY_OPTIMIZER.before_infer()

	did_compute = False
	t_start = time.perf_counter()

	with torch.inference_mode():
		if payload is not None and cached_output_text is not None and cached_output_tokens is not None:
			cache_hit = True
			input_len = int(payload["input_len"])
			output_len = int(cached_output_tokens)
			output_text = cached_output_text
		elif payload is not None and cached_prompt_past_key_values is not None and cached_first_token_id is not None:
			cache_hit = True
			did_compute = True
			input_len = int(payload["input_len"])
			_safe_cuda_sync()
			generated = _greedy_decode_from_prefill(
				model=model,
				tokenizer=tokenizer,
				past_key_values=cached_prompt_past_key_values,
				first_token_id=int(cached_first_token_id),
				prompt_len=input_len,
				max_new_tokens=MAX_NEW_TOKENS,
				device=payload["input_ids"].device,
			)
			output_len = int(generated.shape[1])
			output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
			cached_page_count = int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None
			if use_kv_cache:
				payload_builder = _build_kv_page_payload_builder(
					past_key_values=cached_prompt_past_key_values,
					prompt_len=input_len,
				)
				cache.put_sequence(
					prompt,
					payload={
						"input_ids": payload["input_ids"],
						"attention_mask": payload["attention_mask"],
						"input_len": input_len,
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": cached_page_count,
						"cached_first_token_id": int(cached_first_token_id),
					},
					payload_builder=payload_builder,
					token_count=input_len,
				)
		elif payload is not None:
			cache_hit = True
			did_compute = True
			_safe_cuda_sync()
			prefill_outputs = model(
				input_ids=payload["input_ids"],
				attention_mask=payload["attention_mask"],
				use_cache=True,
				return_dict=True,
			)
			input_len = payload["input_len"]
			prefill_first_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1).item()
			generated = _greedy_decode_from_prefill(
				model=model,
				tokenizer=tokenizer,
				past_key_values=prefill_outputs.past_key_values,
				first_token_id=int(prefill_first_token),
				prompt_len=input_len,
				max_new_tokens=MAX_NEW_TOKENS,
				device=payload["input_ids"].device,
			)
			output_len = int(generated.shape[1])
			output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
			cached_page_count = int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None
			if use_kv_cache:
				payload_builder = _build_kv_page_payload_builder(
					past_key_values=prefill_outputs.past_key_values,
					prompt_len=input_len,
				)
				cache.put_sequence(
					prompt,
					payload={
						"input_ids": payload["input_ids"],
						"attention_mask": payload["attention_mask"],
						"input_len": input_len,
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": cached_page_count,
						"cached_first_token_id": int(prefill_first_token),
					},
					payload_builder=payload_builder,
					token_count=input_len,
				)
		else:
			did_compute = True
			inputs = tokenizer(prompt, return_tensors="pt")
			inputs = _to_device(inputs, DEVICE)
			input_len = inputs["input_ids"].shape[1]

			_safe_cuda_sync()
			prefill_outputs = model(
				input_ids=inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				use_cache=True,
				return_dict=True,
			)
			prefill_first_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1).item()
			generated = _greedy_decode_from_prefill(
				model=model,
				tokenizer=tokenizer,
				past_key_values=prefill_outputs.past_key_values,
				first_token_id=int(prefill_first_token),
				prompt_len=input_len,
				max_new_tokens=MAX_NEW_TOKENS,
				device=inputs["input_ids"].device,
			)
			output_len = int(generated.shape[1])
			output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
			cached_page_count = int(math.ceil(float(input_len) / float(cache.page_size_tokens))) if getattr(cache, "page_size_tokens", 0) > 0 else None

			if use_kv_cache:
				payload_builder = _build_kv_page_payload_builder(
					past_key_values=prefill_outputs.past_key_values,
					prompt_len=input_len,
				)
				cache.put_sequence(
					prompt,
					payload={
						"input_ids": inputs["input_ids"],
						"attention_mask": inputs["attention_mask"],
						"input_len": input_len,
						"cached_output_text": output_text,
						"cached_output_tokens": output_len,
						"cached_page_count": cached_page_count,
						"cached_first_token_id": int(prefill_first_token),
					},
					payload_builder=payload_builder,
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


def infer_continuous_batch(
	tokenizer, model, prompts: list[str],
	max_batch_size: int = 8, max_new_tokens: int = None,
) -> list[dict]:
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
					return_dict=True,
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
					return_dict=True,
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
						"cached_page_count": int(math.ceil(
							float(state["input_len"]) / float(cache.page_size_tokens)
						)) if getattr(cache, "page_size_tokens", 0) > 0 else None,
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

	cb_stats = scheduler.stats()
	print(
		f"[ContinuousBatch] 完成 | decode steps={step_count} "
		f"finished={cb_stats['total_finished']} "
		f"avg_active_per_step={cb_stats['avg_active_per_step']} "
		f"avg_queue_wait={cb_stats['avg_queue_wait_ms']:.1f} ms"
	)

	results = [item for item in result_slots if item is not None]
	return results


def parse_args():
	parser = argparse.ArgumentParser(description="zLLM 推理脚本（等效 baseline + 简单优化）")
	parser.add_argument("--model_path", type=str, default="/inspire/ssd/project/mianxiangdayuyanmoxing/public/Qwen2.5-14B-Instruct/", help="模型本地路径")
	parser.add_argument(
		"--prompt",
		type=str,
		default="请用三句话解释大语言模型推理中KV Cache的作用。",
		help="单条测试 prompt",
	)
	parser.add_argument(
		"--mode", type=str, default="single",
		choices=["single", "continuous"],
		help="运行模式: single=单条推理, continuous=Continuous Batch",
	)
	parser.add_argument(
		"--prompt_file", type=str,
		default=str(Path(__file__).resolve().parent.parent / "prompts.jsonl"),
		help="continuous 模式下的 prompt 文件路径",
	)
	parser.add_argument("--max_batch_size", type=int, default=8, help="continuous 模式下的最大并发 batch 大小")
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

	elif args.mode == "continuous":
		prompts = _load_prompts(args.prompt_file)
		print(f"\n[ContinuousBatch] 加载 {len(prompts)} 条 prompt，max_batch_size={args.max_batch_size}")
		results = infer_continuous_batch(
			tokenizer, model, prompts,
			max_batch_size=args.max_batch_size,
		)

		print("\n" + "=" * 64)
		print(" Continuous Batch 结果汇总")
		print("=" * 64)
		print(f"  请求总数       : {len(results)}")
		if results:
			avg_latency = sum(r["total_latency_ms"] for r in results) / len(results)
			avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
			avg_tps = sum(r["throughput_tps"] for r in results) / len(results)
			total_out = sum(r["output_tokens"] for r in results)
			print(f"  总输出 tokens  : {total_out}")
			print(f"  平均延迟       : {avg_latency:.2f} ms")
			print(f"  平均 TTFT      : {avg_ttft:.2f} ms")
			print(f"  平均吞吐率     : {avg_tps:.2f} tokens/sec")
		if torch.cuda.is_available():
			print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
		print("=" * 64)

	mem = MEMORY_OPTIMIZER.memory_snapshot()
	if mem.get("cuda", False):
		print(
			f"[INFO] 显存快照 | allocated={mem['allocated_gb']:.3f} GB "
			f"reserved={mem['reserved_gb']:.3f} GB "
			f"max_allocated={mem['max_allocated_gb']:.3f} GB"
		)
