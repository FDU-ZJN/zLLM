import argparse
import asyncio
import inspect
import os
import time
import multiprocessing as mp
import uuid

import torch


DEVICE = "cuda"
MAX_NEW_TOKENS = 256
SEED = 42
DEFAULT_KV_BLOCK_SIZE = 16
DEFAULT_SYSTEM_PROMPT = "你是一个稳定、严谨且简洁的智能助手。"
DEFAULT_LEADING_TEMPLATE = "请先基于统一上下文理解问题，再给出直接可执行的回答。"

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
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        return AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    except Exception as exc:
        raise ImportError(
            "未检测到可用的 vLLM，请先安装并确认 CUDA 环境正常。"
            "示例：pip install vllm"
        ) from exc


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj)
        supported = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in supported and v is not None}
    except Exception:
        return {k: v for k, v in kwargs.items() if v is not None}


class VLLMOptimizedInference:
    """
    面向 H100 的 vLLM 推理封装。

    接口：
    - load_model(model_path) -> (tokenizer, model)
    - infer_single(tokenizer, model, prompt, use_kv_cache=True) -> dict
    - infer_stream(tokenizer, model, prompt, use_kv_cache=True) -> AsyncGenerator[dict, None]
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

    # ------------------------------------------------------------------ #
    #  文本处理                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_text(text: str) -> str:
        if text is None:
            return ""
        lines = [line.strip() for line in str(text).splitlines()]
        non_empty = [line for line in lines if line]
        return "\n".join(non_empty)

    def _compose_canonical_prefix(self) -> str:
        parts = [
            self._normalize_text(self.system_prompt),
            self._normalize_text(self.leading_template),
        ]
        parts = [p for p in parts if p]
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
        final_prompt = f"{aligned_prefix}{normalized_user_prompt}" if aligned_prefix else normalized_user_prompt
        return final_prompt, {
            "enabled": True,
            "prefix_tokens": int(prefix_tokens),
            "pad_tokens": int(pad_tokens),
        }

    # ------------------------------------------------------------------ #
    #  引擎构建                                                            #
    # ------------------------------------------------------------------ #

    def _resolve_tp_size(self) -> int:
        if self.tensor_parallel_size is not None and self.tensor_parallel_size > 0:
            return int(self.tensor_parallel_size)
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cuda_visible:
            return 1
        visible = [x.strip() for x in cuda_visible.split(",") if x.strip() != ""]
        return max(1, len(visible))

    def _build_engine_args(self, model_path: str) -> dict:
        return {
            "model": model_path,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self._resolve_tp_size(),
            "dtype": "bfloat16",
            "quantization": "fp8",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": True,
            "disable_log_stats": False,
            "enforce_eager": False,
            "max_model_len": 32768,
            "kv_cache_dtype": "auto",
        }

    # ------------------------------------------------------------------ #
    #  load_model                                                          #
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str):
        _ensure_spawn_start_method()
        AsyncLLMEngine, AsyncEngineArgs, SamplingParams = _lazy_import_vllm()

        print(f"[INFO] 加载 AsyncLLMEngine 模型: {model_path}")
        print(f"[INFO] 设备: {DEVICE} | 优化目标: H100")

        raw_kwargs = self._build_engine_args(model_path)
        engine_args_kwargs = _filter_supported_kwargs(AsyncEngineArgs.__init__, raw_kwargs)
        engine_args = AsyncEngineArgs(**engine_args_kwargs)
        self.model = AsyncLLMEngine.from_engine_args(engine_args)

        # get_tokenizer() 在 vLLM v1 是同步方法；v0 是协程，兼容两者
        tok = self.model.get_tokenizer()
        if inspect.isawaitable(tok):
            self.tokenizer = asyncio.get_event_loop().run_until_complete(tok)
        else:
            self.tokenizer = tok

        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )

        print(
            "[INFO] AsyncLLMEngine 加载完成 | "
            f"dtype={raw_kwargs.get('dtype')} | "
            f"tp={raw_kwargs.get('tensor_parallel_size')} | "
            f"prefix_cache={raw_kwargs.get('enable_prefix_caching')}"
        )
        return self.tokenizer, self.model

    # ------------------------------------------------------------------ #
    #  核心：异步流式生成                                                  #
    # ------------------------------------------------------------------ #

    async def _astream(self, prompt: str, use_kv_cache: bool = True):
        if self.model is None:
            raise ValueError("model 为空，请先调用 load_model")

        standardized_prompt, normalization_stats = self._build_standardized_prompt(prompt)
        cache_hit = bool(use_kv_cache and standardized_prompt in self._seen_prompts)
        request_id = str(uuid.uuid4())

        t_start = time.perf_counter()
        t_first_token: float | None = None

        prev_text = ""
        input_tokens = 0
        output_tokens = 0

        async for request_output in self.model.generate(
            standardized_prompt,
            self.sampling_params,
            request_id=request_id,
        ):
            output = request_output.outputs[0]
            current_text = output.text
            delta = current_text[len(prev_text):]
            prev_text = current_text
            output_tokens = len(output.token_ids)

            # 第一个有内容的 delta 到达时记录 TTFT
            if t_first_token is None and delta:
                t_first_token = time.perf_counter()

            if input_tokens == 0 and getattr(request_output, "prompt_token_ids", None):
                input_tokens = len(request_output.prompt_token_ids)

            finished = output.finish_reason is not None

            frame: dict = {
                "delta": delta,
                "text_so_far": current_text,
                "output_tokens": output_tokens,
                "finished": finished,
            }

            if finished:
                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000
                ttft_ms = ((t_first_token - t_start) * 1000) if t_first_token else total_ms
                throughput = (output_tokens / total_ms * 1000) if total_ms > 0 else 0.0

                if use_kv_cache:
                    self._seen_prompts.add(standardized_prompt)

                block_size = self.kv_block_size
                page_count = (input_tokens + block_size - 1) // block_size if input_tokens > 0 else None
                prefix_reuse_pages = page_count if cache_hit else 0

                frame.update({
                    "prompt": prompt,
                    "standardized_prompt": standardized_prompt,
                    "output": current_text,
                    "input_tokens": int(input_tokens),
                    "total_latency_ms": round(total_ms, 2),
                    "ttft_ms": round(ttft_ms, 2),
                    "throughput_tps": round(throughput, 2),
                    "kv_cache_hit": cache_hit,
                    "kv_cache_pages": page_count,
                    "kv_prefix_reuse_pages": prefix_reuse_pages,
                    "kv_cache_stats": {
                        "prefix_cache_enabled": True,
                        "tracked_prompt_count": len(self._seen_prompts),
                        "scheduler_tokens": self.max_num_batched_tokens,
                        "target_arch": "H100",
                        "kv_dtype": "auto",
                        "prompt_standardization": normalization_stats,
                        "kv_block_size": self.kv_block_size,
                    },
                })

            yield frame

    # ------------------------------------------------------------------ #
    #  公开接口                                                            #
    # ------------------------------------------------------------------ #

    async def infer_stream(
        self,
        tokenizer,
        model,
        prompt: str,
        use_kv_cache: bool = True,
    ):
        async for frame in self._astream(prompt, use_kv_cache=use_kv_cache):
            yield frame

    def infer_single(
        self,
        tokenizer,
        model,
        prompt: str,
        use_kv_cache: bool = True,
    ) -> dict:
        async def _run():
            result = {}
            async for frame in self._astream(prompt, use_kv_cache=use_kv_cache):
                result = frame
            return result

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run())
_ENGINE = VLLMOptimizedInference()
def load_model(model_path: str):
    return _ENGINE.load_model(model_path)


def infer_single(tokenizer, model, prompt: str, use_kv_cache: bool = True) -> dict:
    return _ENGINE.infer_single(tokenizer, model, prompt, use_kv_cache=use_kv_cache)


async def infer_stream(tokenizer, model, prompt: str, use_kv_cache: bool = True):
    """模块级流式接口，与 infer_single 对称。"""
    async for frame in _ENGINE.infer_stream(tokenizer, model, prompt, use_kv_cache=use_kv_cache):
        yield frame

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM 优化推理脚本（H100 导向）")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="请用三句话解释大语言模型推理中KV Cache的作用。",
    )
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--leading_template", type=str, default=DEFAULT_LEADING_TEMPLATE)
    parser.add_argument("--kv_block_size", type=int, default=DEFAULT_KV_BLOCK_SIZE)
    parser.add_argument("--disable_prompt_standardization", action="store_true")
    parser.add_argument("--no_stream", action="store_true", help="使用同步阻塞模式（兼容旧行为）")
    return parser.parse_args()


async def _main_async(args):
    _ENGINE.max_new_tokens = int(args.max_new_tokens)
    _ENGINE.tensor_parallel_size = args.tp
    _ENGINE.system_prompt = (args.system_prompt or "").strip()
    _ENGINE.leading_template = (args.leading_template or "").strip()
    _ENGINE.kv_block_size = max(1, int(args.kv_block_size))
    _ENGINE.enable_prompt_standardization = not args.disable_prompt_standardization

    tokenizer, model = load_model(args.model_path)
    print(f"\n[推理] prompt: {args.prompt}\n")

    if args.no_stream:
        result = infer_single(tokenizer, model, args.prompt)
        _print_result(result)
    else:
        print("=" * 64)
        print(" vLLM 流式推理输出")
        print("=" * 64)
        result = None
        async for chunk in _ENGINE.infer_stream(tokenizer, model, args.prompt):
            print(chunk["delta"], end="", flush=True)
            if chunk["finished"]:
                result = chunk
        print()
        _print_result(result)


def _print_result(result: dict):
    if result is None:
        return
    print("\n" + "=" * 64)
    print(" 统计信息")
    print("=" * 64)
    print(f"  输入 tokens      : {result['input_tokens']}")
    print(f"  输出 tokens      : {result['output_tokens']}")
    print(f"  TTFT             : {result['ttft_ms']} ms")
    print(f"  总延迟            : {result['total_latency_ms']} ms")
    print(f"  吞吐率            : {result['throughput_tps']} tokens/sec")
    print(f"  KV Cache 命中    : {result['kv_cache_hit']}")
    print(f"  标准化启用        : {result['kv_cache_stats']['prompt_standardization']['enabled']}")
    if result.get("kv_cache_pages") is not None:
        print(f"  KV Page 数       : {result['kv_cache_pages']}")
    if result.get("kv_prefix_reuse_pages") is not None:
        print(f"  KV 前缀复用页    : {result['kv_prefix_reuse_pages']}")
    print(
        f"  前缀tokens/补齐  : "
        f"{result['kv_cache_stats']['prompt_standardization']['prefix_tokens']}"
        f"/{result['kv_cache_stats']['prompt_standardization']['pad_tokens']}"
    )
    if torch.cuda.is_available():
        print(f"  峰值显存         : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print("=" * 64)


if __name__ == "__main__":
    _ensure_spawn_start_method()
    args = parse_args()
    asyncio.run(_main_async(args))