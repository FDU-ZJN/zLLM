import argparse
import json
import time
import math
from pathlib import Path
from optimizations.kv_cache import PagedKVCacheManager, PagedAttentionKVCacheManager
from optimizations.kv_cache.kv_offload_manager import KVCacheOffloadManager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimizations.memory_optimizer import MemoryOptimizer
from optimizations.dynamic_batch_scheduler import ContinuousBatchScheduler
from optimizations.device_map import (
    build_device_map,
    get_first_device,
    get_output_device,
    summarize_device_map,
    print_device_map_debug,
)
from optimizations.heterogeneous_cpu_gpu import attach_kv_offload_to_memory_optimizer
from optimizations.heterogeneous_pipeline import (
	get_async_boundary_stats,
	get_front_cpu_layer_count,
	patch_qwen2_model_async_boundary,
)


DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256
SEED = 42

# 异构推理调试开关：打印 CPU/GPU 数据搬运等信息
DEBUG_HETEROGENEOUS = False
# 编排式异构：greedy decode 时最后一维 logits 在 CPU 上 argmax（轻量算不占用 GPU）
DECODE_LOGITS_ON_CPU = False

torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)

ACTIVE_DEVICE_MAP = None
INPUT_DEVICE = DEVICE
OUTPUT_DEVICE = DEVICE
KV_OFFLOAD_MANAGER = None

KV_CACHE = None
MEMORY_OPTIMIZER = MemoryOptimizer(clear_interval=8, force_gc=True, cuda_empty_cache=True)
MEMORY_OPTIMIZER.try_set_per_process_memory_fraction(0.9)


def _build_kv_cache(
	page_size_tokens: int = 64,
	use_paged_attention: bool = False,
	block_pool_config: dict | None = None,
):
	if use_paged_attention:
		return PagedAttentionKVCacheManager(
			max_entries=256,
			max_cache_tokens=786432,
			ttl_seconds=7200,
			page_size_tokens=page_size_tokens,
			enable_prefix_sharing=False,
			block_pool_config=block_pool_config or {},
		)
	return PagedKVCacheManager(
		max_entries=256,
		max_cache_tokens=786432,
		ttl_seconds=7200,
		page_size_tokens=page_size_tokens,
		enable_prefix_sharing=False,
	)


def _ensure_kv_cache(
	page_size_tokens: int = 64,
	use_paged_attention: bool = False,
	block_pool_config: dict | None = None,
):
	global KV_CACHE
	if KV_CACHE is None:
		KV_CACHE = _build_kv_cache(
			page_size_tokens=page_size_tokens,
			use_paged_attention=use_paged_attention,
			block_pool_config=block_pool_config,
		)
	return KV_CACHE

def _argmax_last_token_id(logits: torch.Tensor, logits_on_cpu: bool) -> int:
	"""最后一维 vocab 上 greedy argmax；``logits_on_cpu`` 时在 CPU 上算。"""
	last = logits[:, -1, :]
	if logits_on_cpu:
		return int(torch.argmax(last.float().cpu(), dim=-1).item())
	return int(torch.argmax(last, dim=-1).item())


def _greedy_next_token_tensor(logits: torch.Tensor, logits_on_cpu: bool, input_device) -> torch.Tensor:
	"""与 ``_argmax_last_token_id`` 相同语义，返回 ``(B,1)`` long tensor 于 ``input_device``。"""
	if logits_on_cpu:
		idx = torch.argmax(logits[:, -1, :].float().cpu(), dim=-1, keepdim=True)
		return idx.to(input_device)
	return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _debug_transfer(label: str, src: str, dst: str, shape=None, dtype=None, bytes_approx=None):
	"""打印异构数据搬运调试信息。"""
	if not DEBUG_HETEROGENEOUS:
		return
	info = []
	if shape is not None:
		info.append(f"shape={shape}")
	if dtype is not None:
		info.append(f"dtype={dtype}")
	if bytes_approx is not None:
		info.append(f"~{bytes_approx/1024:.2f} KB")
	extra = f" ({', '.join(info)})" if info else ""
	print(f"[异构调试] {label}: {src} -> {dst}{extra}")


def _to_device(inputs: dict, device: str = None) -> dict:
	if device is None:
		device = INPUT_DEVICE
	result = {}
	for k, v in inputs.items():
		src = str(v.device) if hasattr(v, "device") else "?"
		if hasattr(v, "shape") and str(src) != str(device):
			bytes_approx = v.numel() * v.element_size() if hasattr(v, "numel") else 0
			_debug_transfer("输入搬运", src, device, shape=tuple(v.shape), dtype=str(v.dtype), bytes_approx=bytes_approx)
		result[k] = v.to(device)
	return result


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
	logits_on_cpu: bool = False,
):
	input_dev = INPUT_DEVICE if ACTIVE_DEVICE_MAP else device
	output_dev = OUTPUT_DEVICE if ACTIVE_DEVICE_MAP else device

	if max_new_tokens <= 0:
		empty = torch.empty((1, 0), dtype=torch.long, device=output_dev)
		return empty

	next_token = torch.tensor([[int(first_token_id)]], dtype=torch.long, device=input_dev)
	attention_mask = torch.ones((1, int(prompt_len)), dtype=torch.long, device=input_dev)
	generated_tokens = []
	eos_id = tokenizer.eos_token_id
	past = past_key_values

	for _ in range(int(max_new_tokens)):
		tok_dev = str(next_token.device)
		if tok_dev != str(output_dev):
			_debug_transfer("生成 token -> 输出设备", tok_dev, str(output_dev), shape=next_token.shape, bytes_approx=next_token.numel() * next_token.element_size())
		generated_tokens.append(next_token.to(output_dev))
		if eos_id is not None and int(next_token.item()) == int(eos_id):
			break

		attention_mask = torch.cat(
			[attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=input_dev)],
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
		logits_dev = str(outputs.logits.device)
		next_token = _greedy_next_token_tensor(outputs.logits, logits_on_cpu, input_dev)
		if not logits_on_cpu and str(next_token.device) != str(input_dev):
			_debug_transfer("logits argmax -> 输入设备 (下一轮)", logits_dev, str(input_dev), shape=next_token.shape, bytes_approx=next_token.numel() * next_token.element_size())
			next_token = next_token.to(input_dev)

	if not generated_tokens:
		return torch.empty((1, 0), dtype=torch.long, device=output_dev)
	return torch.cat(generated_tokens, dim=1)


def load_model(
	model_path: str,
	cpu_ratio: float = 0.0,
	gpu_mem_limit: float = None,
	device_strategy: str = "front_cpu",
	offload_folder: str = "offload_cache",
	num_cpu_threads: int = None,
	async_cpu_gpu_boundary: bool = False,
):
	global ACTIVE_DEVICE_MAP, INPUT_DEVICE, OUTPUT_DEVICE, KV_OFFLOAD_MANAGER

	print(f"[INFO] 加载模型: {model_path}")
	print(f"[INFO] 设备: {DEVICE} | 数据类型: {DTYPE}")

	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		trust_remote_code=True,
		padding_side="left",
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	use_heterogeneous = cpu_ratio > 0.0 or gpu_mem_limit is not None

	if use_heterogeneous:
		config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
		num_layers = getattr(config, "num_hidden_layers", 48)
		total_params_b = None
		if hasattr(config, "num_parameters"):
			total_params_b = config.num_parameters / 1e9

		device_map = build_device_map(
			num_hidden_layers=num_layers,
			cpu_offload_ratio=cpu_ratio,
			strategy=device_strategy,
			gpu_device=DEVICE,
			gpu_mem_limit_gb=gpu_mem_limit,
			total_params_b=total_params_b,
		)
		ACTIVE_DEVICE_MAP = device_map
		INPUT_DEVICE = get_first_device(device_map)
		OUTPUT_DEVICE = get_output_device(device_map)

		summary = summarize_device_map(device_map)
		print(
			f"[INFO] 层切分异构 (device_map): {summary['cpu_layers']} 层在 CPU, "
			f"{summary['gpu_layers']} 层在 GPU"
		)
		print(f"[INFO] 输入设备: {INPUT_DEVICE} | 输出设备: {OUTPUT_DEVICE}")
		print(f"[INFO] 策略: {device_strategy}")
		if DEBUG_HETEROGENEOUS:
			print_device_map_debug(device_map)

		if num_cpu_threads is not None:
			torch.set_num_threads(num_cpu_threads)
			print(f"[INFO] CPU 推理线程数: {num_cpu_threads}")

		model = AutoModelForCausalLM.from_pretrained(
			model_path,
			torch_dtype=DTYPE,
			low_cpu_mem_usage=True,
			trust_remote_code=True,
			device_map=device_map,
			offload_folder=offload_folder,
		)

		KV_OFFLOAD_MANAGER = KVCacheOffloadManager(
			num_layers=num_layers,
			max_gpu_pages=256,
			gpu_device=DEVICE,
		)

		if (
			async_cpu_gpu_boundary
			and device_strategy in ("front_cpu", "auto")
			and summary["cpu_layers"] > 0
			and summary["gpu_layers"] > 0
		):
			front_cpu_n, detected_gpu = get_front_cpu_layer_count(model)
			if front_cpu_n != int(summary["cpu_layers"]):
				print(
					f"[WARN] device_map 摘要 CPU 层数 ({summary['cpu_layers']}) 与模型参数实际前缀 "
					f"({front_cpu_n}) 不一致，异步边界以模型为准"
				)
			gpu_for_boundary = (
				detected_gpu if detected_gpu and str(detected_gpu).startswith("cuda") else DEVICE
			)
			xfer = patch_qwen2_model_async_boundary(
				model,
				cpu_layer_count=int(front_cpu_n),
				gpu_device=gpu_for_boundary,
			)
			if xfer is not None:
				print(
					f"[INFO] 已启用 CPU/GPU 异构边界异步 H2D：前 {front_cpu_n} 层 CPU，"
					f"GPU 设备 {gpu_for_boundary}（Qwen2Model + pinned/stream）"
				)
			else:
				print("[WARN] 未启用边界异步：当前模型非 Qwen2Model 或层配置不适用（需首尾均有 CPU/GPU 段）")
	else:
		ACTIVE_DEVICE_MAP = None
		INPUT_DEVICE = DEVICE
		OUTPUT_DEVICE = DEVICE
		KV_OFFLOAD_MANAGER = None
		model = AutoModelForCausalLM.from_pretrained(
			model_path,
			torch_dtype=DTYPE,
			low_cpu_mem_usage=True,
			trust_remote_code=True,
			device_map=DEVICE,
		)

	model.eval()

	if KV_OFFLOAD_MANAGER is None and torch.cuda.is_available():
		n_layers_cfg = getattr(model.config, "num_hidden_layers", 48)
		KV_OFFLOAD_MANAGER = KVCacheOffloadManager(
			num_layers=n_layers_cfg,
			max_gpu_pages=256,
			gpu_device=DEVICE,
		)
	attach_kv_offload_to_memory_optimizer(MEMORY_OPTIMIZER, KV_OFFLOAD_MANAGER)

	n_params = sum(p.numel() for p in model.parameters()) / 1e9
	mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
	print(f"[INFO] 加载完成 | 参数量: {n_params:.2f}B | 显存占用: {mem_gb:.2f} GB")
	if ACTIVE_DEVICE_MAP is None:
		print(
			"[INFO] 编排式 CPU/GPU 异构: 整模在 GPU；调度/KV 策略在主机侧；"
			"显存压力时 MemoryOptimizer 可触发 KV → CPU（见 kv_offload_stats）"
		)
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
				if sequence_pages is not None and hasattr(cache, "get_past_key_values_from_pages"):
					cached_prompt_past_key_values = cache.get_past_key_values_from_pages(sequence_pages)
				else:
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
				logits_on_cpu=DECODE_LOGITS_ON_CPU,
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
			prefill_first_token = _argmax_last_token_id(prefill_outputs.logits, DECODE_LOGITS_ON_CPU)
			generated = _greedy_decode_from_prefill(
				model=model,
				tokenizer=tokenizer,
				past_key_values=prefill_outputs.past_key_values,
				first_token_id=int(prefill_first_token),
				prompt_len=input_len,
				max_new_tokens=MAX_NEW_TOKENS,
				device=payload["input_ids"].device,
				logits_on_cpu=DECODE_LOGITS_ON_CPU,
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
			inputs = _to_device(inputs, INPUT_DEVICE)
			input_len = inputs["input_ids"].shape[1]

			_safe_cuda_sync()
			prefill_outputs = model(
				input_ids=inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				use_cache=True,
				return_dict=True,
			)
			prefill_first_token = _argmax_last_token_id(prefill_outputs.logits, DECODE_LOGITS_ON_CPU)
			generated = _greedy_decode_from_prefill(
				model=model,
				tokenizer=tokenizer,
				past_key_values=prefill_outputs.past_key_values,
				first_token_id=int(prefill_first_token),
				prompt_len=input_len,
				max_new_tokens=MAX_NEW_TOKENS,
				device=inputs["input_ids"].device,
				logits_on_cpu=DECODE_LOGITS_ON_CPU,
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

	result = {
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
	result["cpu_gpu_orchestration"] = {
		"decode_logits_on_cpu": DECODE_LOGITS_ON_CPU,
		"kv_offload_manager": KV_OFFLOAD_MANAGER is not None,
	}
	if KV_OFFLOAD_MANAGER is not None:
		result["kv_offload_stats"] = KV_OFFLOAD_MANAGER.stats()
	if ACTIVE_DEVICE_MAP is not None:
		result["heterogeneous"] = True
		result["layer_split_device_map"] = True
		result["input_device"] = INPUT_DEVICE
		result["output_device"] = OUTPUT_DEVICE
		ab = get_async_boundary_stats(model)
		if ab is not None:
			result["async_boundary_stats"] = ab
	else:
		result["layer_split_device_map"] = False
	return result


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
	"""Continuous Batching 推理 —— iteration-level 调度。

	与 Static / Dynamic Batch 的关键区别：
	  - 不使用 model.generate()，而是手动执行逐 token 的 forward pass
	  - 每个 decode step 后，已完成的请求立即驱逐，空闲 slot 立即填入新请求
	  - 消除 batch bubble（短请求等待长请求完成的空转），最大化 GPU 利用率
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
			enqueue_ms = req["enqueue_ms"]

			inputs = tokenizer(prompt, return_tensors="pt")
			inputs = _to_device(inputs, INPUT_DEVICE)
			input_len = inputs["input_ids"].shape[1]

			_safe_cuda_sync()
			t_prefill_start = time.perf_counter()

			with torch.inference_mode():
				outputs = model(
					input_ids=inputs["input_ids"],
					attention_mask=inputs["attention_mask"],
					use_cache=True,
					return_dict=True,
				)

			_safe_cuda_sync()
			t_prefill_end = time.perf_counter()

			next_token_id = _argmax_last_token_id(outputs.logits, DECODE_LOGITS_ON_CPU)

			now_ms = time.time() * 1000.0
			queue_wait_ms = now_ms - enqueue_ms
			prefill_ms = (t_prefill_end - t_prefill_start) * 1000

			request_states[rid] = {
				"past_key_values": outputs.past_key_values,
				"generated_ids": [next_token_id],
				"input_len": input_len,
				"enqueue_ms": enqueue_ms,
				"prefill_ms": prefill_ms,
				"ttft_ms": queue_wait_ms + prefill_ms,
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
			next_input = torch.tensor([[last_token]], device=INPUT_DEVICE, dtype=torch.long)

			with torch.inference_mode():
				outputs = model(
					input_ids=next_input,
					past_key_values=state["past_key_values"],
					use_cache=True,
					return_dict=True,
				)

			next_token_id = _argmax_last_token_id(outputs.logits, DECODE_LOGITS_ON_CPU)
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
				finish_ms = time.time() * 1000.0
				total_ms = finish_ms - state["enqueue_ms"]
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
					"prefill_ms": round(state["prefill_ms"], 2),
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
	parser = argparse.ArgumentParser(description="zLLM 推理脚本（等效 baseline + 简单优化 + CPU/GPU 异构推理）")
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
	parser.add_argument(
		"--use_paged_attention",
		action="store_true",
		help="启用 PagedAttention 物理块池，减少显存碎片",
	)

	parser.add_argument(
		"--cpu_ratio", type=float, default=0.0,
		help="层切分异构：按 device_map 放到 CPU 的层比例 [0.0,1.0]；0 表示整模 GPU（编排式异构仍可用）",
	)
	parser.add_argument(
		"--gpu_mem_limit", type=float, default=None,
		help="GPU 可用显存上限 (GB)，设置后自动计算需要卸载到 CPU 的层数",
	)
	parser.add_argument(
		"--device_strategy", type=str, default="front_cpu",
		choices=["front_cpu", "back_cpu", "interleave", "auto"],
		help="异构推理层分配策略: front_cpu=前N层放CPU, back_cpu=后N层, interleave=交替, auto=自动",
	)
	parser.add_argument(
		"--offload_folder", type=str, default="offload_cache",
		help="磁盘卸载缓存目录（accelerate offload 使用）",
	)
	parser.add_argument(
		"--num_cpu_threads", type=int, default=None,
		help="CPU 推理时使用的线程数（异构模式下推荐设置）",
	)
	parser.add_argument(
		"--async_cpu_gpu_boundary",
		action="store_true",
		help="front_cpu/auto 异构时，在最后一层 CPU 与第一层 GPU 之间用 pinned + 独立 CUDA stream 做异步 H2D（仅 Qwen2Model）",
	)
	parser.add_argument(
		"--debug_heterogeneous", action="store_true",
		help="打印层切分异构时的数据搬运与设备边界调试信息",
	)
	parser.add_argument(
		"--decode_logits_on_cpu",
		action="store_true",
		help="编排式异构：greedy 解码时最后一维 logits 在 CPU 上 argmax，减轻 GPU 上小算子占用",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	globals()["DEBUG_HETEROGENEOUS"] = args.debug_heterogeneous
	globals()["DECODE_LOGITS_ON_CPU"] = args.decode_logits_on_cpu
	KV_CACHE = _build_kv_cache(page_size_tokens=args.kv_page_size)
	cache = _build_kv_cache(
		page_size_tokens=args.kv_page_size,
		use_paged_attention=getattr(args, "use_paged_attention", False),
		block_pool_config=getattr(args, "block_pool_config", None),
	)
	KV_CACHE = cache
	tokenizer, model = load_model(
		args.model_path,
		cpu_ratio=args.cpu_ratio,
		gpu_mem_limit=args.gpu_mem_limit,
		device_strategy=args.device_strategy,
		offload_folder=args.offload_folder,
		num_cpu_threads=args.num_cpu_threads,
		async_cpu_gpu_boundary=getattr(args, "async_cpu_gpu_boundary", False),
	)
	if hasattr(KV_CACHE, "init_block_pool_from_config") and model.config is not None:
		KV_CACHE.init_block_pool_from_config(model.config)
		print("[INFO] PagedAttention 物理块池已初始化")
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
		if result.get("heterogeneous"):
			print(f"  异构推理       : 是")
			print(f"  输入设备       : {result.get('input_device')}")
			print(f"  输出设备       : {result.get('output_device')}")
			if result.get("async_boundary_stats"):
				abs_ = result["async_boundary_stats"]
				print(
					f"  边界异步H2D    : "
					f"{abs_.get('h2d_calls', 0)} 次, "
					f"{abs_.get('h2d_bytes', 0) / 1024:.1f} KB"
				)
			if result.get("kv_offload_stats"):
				ofs = result["kv_offload_stats"]
				print(f"  KV GPU页       : {ofs.get('gpu_pages', 0)}")
				print(f"  KV CPU页       : {ofs.get('cpu_pages', 0)}")
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
		if ACTIVE_DEVICE_MAP is not None:
			dm_summary = summarize_device_map(ACTIVE_DEVICE_MAP)
			print(f"  异构推理       : 是 ({dm_summary['cpu_layers']} CPU层 / {dm_summary['gpu_layers']} GPU层)")
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
	if mem.get("cpu_available_gb") is not None:
		print(
			f"[INFO] CPU 内存 | used={mem.get('cpu_used_gb', 0):.3f} GB "
			f"available={mem.get('cpu_available_gb', 0):.3f} GB"
		)
