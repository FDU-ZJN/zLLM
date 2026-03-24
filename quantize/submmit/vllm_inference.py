import argparse
import inspect
import os
import time
import multiprocessing as mp

import torch


DEVICE = "cuda"
MAX_NEW_TOKENS = 256
SEED = 42
DEFAULT_KV_BLOCK_SIZE = 16
DEFAULT_SYSTEM_PROMPT = "你是一个稳定、严谨且简洁的智能助手。"
DEFAULT_LEADING_TEMPLATE = "请先基于统一上下文理解问题，再给出直接可执行的回答。"

# 避免 vLLM worker 使用 fork 导致 CUDA 在子进程二次初始化
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

torch.manual_seed(SEED)


def _ensure_spawn_start_method():
	try:
		if mp.get_start_method(allow_none=True) != "spawn":
			mp.set_start_method("spawn", force=True)
	except RuntimeError:
		pass


def _lazy_import_vllm():
	try:
		from vllm import LLM, SamplingParams
		return LLM, SamplingParams
	except Exception as exc:
		raise ImportError(
			"未检测到可用的 vLLM，请先安装并确认 CUDA 环境正常。"
			"示例：pip install vllm"
		) from exc


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
	"""兼容不同 vLLM 版本：仅传入当前版本支持的参数。"""
	try:
		sig = inspect.signature(callable_obj)
		supported = set(sig.parameters.keys())
		return {k: v for k, v in kwargs.items() if k in supported and v is not None}
	except Exception:
		return {k: v for k, v in kwargs.items() if v is not None}


class VLLMOptimizedInference:
	"""
	面向 H100 的 vLLM 推理封装。

	接口设计保持和 zllm_inference 对齐：
	- load_model(model_path) -> (tokenizer, model)
	- infer_single(tokenizer, model, prompt, use_kv_cache=True) -> dict
	"""

	def __init__(
		self,
		max_new_tokens: int = MAX_NEW_TOKENS,
		gpu_memory_utilization: float = 0.93,
		max_num_batched_tokens: int = 32768,
		max_num_seqs: int = 256,
		tensor_parallel_size: int | None = None,
		trust_remote_code: bool = True,
		system_prompt: str = DEFAULT_SYSTEM_PROMPT,
		leading_template: str = DEFAULT_LEADING_TEMPLATE,
		kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
		enable_prompt_standardization: bool = True,
	):
		self.max_new_tokens = int(max_new_tokens)
		self.gpu_memory_utilization = float(gpu_memory_utilization)
		self.max_num_batched_tokens = int(max_num_batched_tokens)
		self.max_num_seqs = int(max_num_seqs)
		self.tensor_parallel_size = tensor_parallel_size
		self.trust_remote_code = bool(trust_remote_code)
		self.system_prompt = (system_prompt or "").strip()
		self.leading_template = (leading_template or "").strip()
		self.kv_block_size = max(1, int(kv_block_size))
		self.enable_prompt_standardization = bool(enable_prompt_standardization)

		self.tokenizer = None
		self.model = None
		self.sampling_params = None
		self._seen_prompts = set()

	@staticmethod
	def _normalize_text(text: str) -> str:
		if text is None:
			return ""
		lines = [line.strip() for line in str(text).splitlines()]
		non_empty = [line for line in lines if line]
		return "\n".join(non_empty)

	def _compose_canonical_prefix(self) -> str:
		parts = [self._normalize_text(self.system_prompt), self._normalize_text(self.leading_template)]
		parts = [part for part in parts if part]
		if not parts:
			return ""
		return "\n\n".join(parts) + "\n\n"

	def _encode_no_special_tokens(self, text: str) -> list[int]:
		if self.tokenizer is None:
			return []
		try:
			return self.tokenizer.encode(text, add_special_tokens=False)
		except TypeError:
			return self.tokenizer.encode(text)

	def _align_prefix_to_block(self, prefix: str) -> tuple[str, int, int]:
		if not prefix or self.tokenizer is None:
			return prefix, 0, 0

		base_ids = self._encode_no_special_tokens(prefix)
		if not base_ids:
			return prefix, 0, 0

		remainder = len(base_ids) % self.kv_block_size
		if remainder == 0:
			return prefix, len(base_ids), 0

		target_pad = self.kv_block_size - remainder
		candidate = prefix
		for _ in range(self.kv_block_size * 8):
			candidate += "\n"
			candidate_ids = self._encode_no_special_tokens(candidate)
			if candidate_ids and len(candidate_ids) % self.kv_block_size == 0:
				pad_tokens = len(candidate_ids) - len(base_ids)
				return candidate, len(candidate_ids), max(0, pad_tokens)

		return prefix, len(base_ids), target_pad

	def _build_standardized_prompt(self, prompt: str) -> tuple[str, dict]:
		normalized_user_prompt = self._normalize_text(prompt)
		if not self.enable_prompt_standardization:
			return normalized_user_prompt, {
				"enabled": False,
				"prefix_tokens": 0,
				"pad_tokens": 0,
			}

		prefix = self._compose_canonical_prefix()
		aligned_prefix, prefix_tokens, pad_tokens = self._align_prefix_to_block(prefix)
		if aligned_prefix:
			final_prompt = f"{aligned_prefix}{normalized_user_prompt}"
		else:
			final_prompt = normalized_user_prompt

		return final_prompt, {
			"enabled": True,
			"prefix_tokens": int(prefix_tokens),
			"pad_tokens": int(pad_tokens),
		}

	def _resolve_dtype(self) -> str:
		return "bfloat16"

	def _resolve_tp_size(self) -> int:
		if self.tensor_parallel_size is not None and self.tensor_parallel_size > 0:
			return int(self.tensor_parallel_size)
		cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
		if not cuda_visible:
			return 1
		visible = [x.strip() for x in cuda_visible.split(",") if x.strip() != ""]
		return max(1, len(visible))

	def _build_llm_kwargs(self, model_path: str) -> dict:
		dtype = self._resolve_dtype()
		tp_size = self._resolve_tp_size()

		# 固定 H100 路径：前缀缓存 + 高 token 预算 + FP8 KV
		kwargs = {
			"model": model_path,
			"trust_remote_code": self.trust_remote_code,
			"tensor_parallel_size": tp_size,
			"dtype": dtype,
   			"quantization":"fp8",
			"gpu_memory_utilization": self.gpu_memory_utilization,
			"max_num_batched_tokens": self.max_num_batched_tokens,
			"max_num_seqs": self.max_num_seqs,
			"enable_prefix_caching": True,
			"disable_log_stats": False,
			"enforce_eager": False,
			"max_model_len": 32768,
			"kv_cache_dtype": "fp8_e5m2",
		}

		return kwargs

	def load_model(self, model_path: str):
		_ensure_spawn_start_method()
		LLM, SamplingParams = _lazy_import_vllm()

		print(f"[INFO] 加载 vLLM 模型: {model_path}")
		print(f"[INFO] 设备: {DEVICE} | 优化目标: H100")

		raw_kwargs = self._build_llm_kwargs(model_path)
		llm_kwargs = _filter_supported_kwargs(LLM.__init__, raw_kwargs)
		self.model = LLM(**llm_kwargs)
		self.tokenizer = self.model.get_tokenizer()
		self.sampling_params = SamplingParams(
			temperature=0.0,
			top_p=1.0,
			max_tokens=self.max_new_tokens,
		)

		print(
			"[INFO] vLLM 加载完成 | "
			f"dtype={raw_kwargs.get('dtype')} | "
			f"tp={raw_kwargs.get('tensor_parallel_size')} | "
			f"prefix_cache={raw_kwargs.get('enable_prefix_caching')}"
		)
		return self.tokenizer, self.model

	def infer_single(self, tokenizer, model, prompt: str, use_kv_cache: bool = True) -> dict:
		if model is None:
			raise ValueError("model 为空，请先调用 load_model")

		standardized_prompt, normalization_stats = self._build_standardized_prompt(prompt)
		cache_hit = bool(use_kv_cache and standardized_prompt in self._seen_prompts)
		t_start = time.perf_counter()

		outputs = model.generate([standardized_prompt], self.sampling_params, use_tqdm=False)
		result = outputs[0]
		output = result.outputs[0]

		text = output.text
		output_tokens = len(output.token_ids)
		input_tokens = len(result.prompt_token_ids) if getattr(result, "prompt_token_ids", None) else 0

		t_end = time.perf_counter()
		total_ms = (t_end - t_start) * 1000
		throughput = (output_tokens / total_ms * 1000) if total_ms > 0 else 0.0

		if use_kv_cache:
			self._seen_prompts.add(standardized_prompt)

		block_size = self.kv_block_size
		page_count = (input_tokens + block_size - 1) // block_size if input_tokens > 0 else None
		prefix_reuse_pages = page_count if cache_hit else 0

		return {
			"prompt": prompt,
			"standardized_prompt": standardized_prompt,
			"output": text,
			"input_tokens": int(input_tokens),
			"output_tokens": int(output_tokens),
			"total_latency_ms": round(total_ms, 2),
			"ttft_ms": round(total_ms, 2),
			"throughput_tps": round(throughput, 2),
			"kv_cache_hit": cache_hit,
			"kv_cache_pages": page_count,
			"kv_prefix_reuse_pages": prefix_reuse_pages,
			"kv_cache_stats": {
				"prefix_cache_enabled": True,
				"tracked_prompt_count": len(self._seen_prompts),
				"scheduler_tokens": self.max_num_batched_tokens,
				"target_arch": "H100",
				"kv_dtype": "fp8_e5m2",
				"prompt_standardization": normalization_stats,
				"kv_block_size": self.kv_block_size,
			},
		}


_ENGINE = VLLMOptimizedInference()


def load_model(model_path: str):
	return _ENGINE.load_model(model_path)


def infer_single(tokenizer, model, prompt: str, use_kv_cache: bool = True) -> dict:
	return _ENGINE.infer_single(tokenizer, model, prompt, use_kv_cache=use_kv_cache)


def parse_args():
	parser = argparse.ArgumentParser(description="vLLM 优化推理脚本（H100 导向）")
	parser.add_argument("--model_path", type=str, required=True, help="模型本地路径")
	parser.add_argument(
		"--prompt",
		type=str,
		default="请用三句话解释大语言模型推理中KV Cache的作用。",
		help="单条测试 prompt",
	)
	parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="最大生成 token 数")
	parser.add_argument("--tp", type=int, default=None, help="tensor parallel size")
	parser.add_argument(
		"--system_prompt",
		type=str,
		default=DEFAULT_SYSTEM_PROMPT,
		help="固定 system prompt（用于前缀标准化）",
	)
	parser.add_argument(
		"--leading_template",
		type=str,
		default=DEFAULT_LEADING_TEMPLATE,
		help="统一前导模板（用于前缀标准化）",
	)
	parser.add_argument(
		"--kv_block_size",
		type=int,
		default=DEFAULT_KV_BLOCK_SIZE,
		help="PagedAttention KV block 大小（用于前缀对齐）",
	)
	parser.add_argument(
		"--disable_prompt_standardization",
		action="store_true",
		help="关闭 prompt 标准化（默认开启）",
	)
	return parser.parse_args()


if __name__ == "__main__":
	_ensure_spawn_start_method()
	args = parse_args()
	_ENGINE.max_new_tokens = int(args.max_new_tokens)
	_ENGINE.tensor_parallel_size = args.tp
	_ENGINE.system_prompt = (args.system_prompt or "").strip()
	_ENGINE.leading_template = (args.leading_template or "").strip()
	_ENGINE.kv_block_size = max(1, int(args.kv_block_size))
	_ENGINE.enable_prompt_standardization = not args.disable_prompt_standardization

	tokenizer, model = load_model(args.model_path)
	print(f"\n[推理] prompt: {args.prompt}")
	result = infer_single(tokenizer, model, args.prompt)

	print("\n" + "=" * 64)
	print(" vLLM 推理结果")
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
	print(f"  标准化启用     : {result['kv_cache_stats']['prompt_standardization']['enabled']}")
	if result.get("kv_cache_pages") is not None:
		print(f"  KV Page数      : {result['kv_cache_pages']}")
	if result.get("kv_prefix_reuse_pages") is not None:
		print(f"  KV前缀复用页   : {result['kv_prefix_reuse_pages']}")
	print(
		f"  前缀tokens/补齐: "
		f"{result['kv_cache_stats']['prompt_standardization']['prefix_tokens']}"
		f"/{result['kv_cache_stats']['prompt_standardization']['pad_tokens']}"
	)
	print("  KV后端         : vLLM prefix cache")
	if torch.cuda.is_available():
		print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
	print("=" * 64)
