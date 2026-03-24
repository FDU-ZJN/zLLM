import argparse
import math
import time
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimizations.kv_cache import PagedKVCacheManager
from optimizations.memory_optimizer import MemoryOptimizer


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


def _stream_decode_from_prefill(
    model,
    tokenizer,
    past_key_values,
    first_token_id,
    prompt_len,
    max_new_tokens,
    device,
    on_token: Callable[[str], None] | None = None,
):
    if max_new_tokens <= 0:
        return torch.empty((1, 0), dtype=torch.long, device=device)

    next_token = torch.tensor([[int(first_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, int(prompt_len)), dtype=torch.long, device=device)
    generated_tokens = []
    eos_id = tokenizer.eos_token_id
    past = past_key_values

    for _ in range(int(max_new_tokens)):
        generated_tokens.append(next_token)
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        if on_token is not None and token_text:
            on_token(token_text)

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


def infer_single_stream(
    tokenizer,
    model,
    prompt: str,
    use_kv_cache: bool = True,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    cache = _ensure_kv_cache()
    cache_hit = False

    payload = None
    cached_output_text = None
    cached_output_tokens = None
    cached_prompt_past_key_values = None
    cached_first_token_id = None
    page_count = None

    if use_kv_cache:
        payload = cache.get_sequence(prompt)
        if isinstance(payload, dict):
            cached_output_text = payload.get("cached_output_text")
            cached_output_tokens = payload.get("cached_output_tokens")
            cached_prompt_past_key_values = payload.get("cached_prompt_past_key_values")
            cached_first_token_id = payload.get("cached_first_token_id")
            page_count = payload.get("cached_page_count")
            if cached_prompt_past_key_values is None and cached_first_token_id is not None:
                sequence_pages = cache.get_sequence_pages(prompt, touch_stats=False)
                cached_prompt_past_key_values = _restore_past_key_values_from_pages(sequence_pages)

    MEMORY_OPTIMIZER.before_infer()
    did_compute = False
    first_token_time = None

    def _token_callback(token_text: str):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if on_token is not None:
            on_token(token_text)

    _safe_cuda_sync()
    t_start = time.perf_counter()

    with torch.inference_mode():
        if payload is not None and cached_output_text is not None and cached_output_tokens is not None:
            cache_hit = True
            input_len = int(payload["input_len"])
            output_text = str(cached_output_text)
            output_len = int(cached_output_tokens)
            if output_text:
                _token_callback(output_text)

        else:
            did_compute = True
            if payload is not None and cached_prompt_past_key_values is not None and cached_first_token_id is not None:
                cache_hit = True
                input_ids = payload["input_ids"]
                attention_mask = payload["attention_mask"]
                input_len = int(payload["input_len"])
                prefill_past = cached_prompt_past_key_values
                first_token_id = int(cached_first_token_id)
            elif payload is not None:
                cache_hit = True
                input_ids = payload["input_ids"]
                attention_mask = payload["attention_mask"]
                input_len = int(payload["input_len"])
                prefill_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                prefill_past = prefill_outputs.past_key_values
                first_token_id = int(torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1).item())
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = _to_device(inputs, DEVICE)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                input_len = int(input_ids.shape[1])
                prefill_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                prefill_past = prefill_outputs.past_key_values
                first_token_id = int(torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1).item())

            generated = _stream_decode_from_prefill(
                model=model,
                tokenizer=tokenizer,
                past_key_values=prefill_past,
                first_token_id=first_token_id,
                prompt_len=input_len,
                max_new_tokens=MAX_NEW_TOKENS,
                device=input_ids.device,
                on_token=_token_callback,
            )
            output_len = int(generated.shape[1])
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            cached_page_count = (
                int(math.ceil(float(input_len) / float(cache.page_size_tokens)))
                if getattr(cache, "page_size_tokens", 0) > 0
                else None
            )
            if use_kv_cache:
                payload_builder = _build_kv_page_payload_builder(
                    past_key_values=prefill_past,
                    prompt_len=input_len,
                )
                cache.put_sequence(
                    prompt,
                    payload={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "input_len": input_len,
                        "cached_output_text": output_text,
                        "cached_output_tokens": output_len,
                        "cached_page_count": cached_page_count,
                        "cached_first_token_id": int(first_token_id),
                    },
                    payload_builder=payload_builder,
                    token_count=input_len,
                )
                page_count = cached_page_count

    if did_compute:
        _safe_cuda_sync()
    t_end = time.perf_counter()

    MEMORY_OPTIMIZER.after_infer()

    total_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time is not None else total_ms
    throughput = (output_len / total_ms * 1000) if total_ms > 0 else 0.0

    return {
        "prompt": prompt,
        "output": output_text,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_latency_ms": round(total_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "throughput_tps": round(throughput, 2),
        "kv_cache_hit": cache_hit,
        "kv_cache_pages": page_count,
        "kv_cache_stats": cache.stats() if use_kv_cache else {},
    }


def infer_single(tokenizer, model, prompt: str, use_kv_cache: bool = True) -> dict:
    return infer_single_stream(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        use_kv_cache=use_kv_cache,
        on_token=None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="zLLM 流式推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型本地路径")
    parser.add_argument(
        "--prompt",
        type=str,
        default="请用三句话解释大语言模型推理中KV Cache的作用。",
        help="单条测试 prompt",
    )
    parser.add_argument("--kv_page_size", type=int, default=16, help="paged后端的每页token数")
    parser.add_argument("--no_kv_cache", action="store_true", help="禁用 KV Cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    KV_CACHE = _build_kv_cache(page_size_tokens=args.kv_page_size)
    tokenizer, model = load_model(args.model_path)
    MEMORY_OPTIMIZER.reset_peak()

    print(f"\n[流式推理] prompt: {args.prompt}")
    print("[输出] ", end="", flush=True)

    result = infer_single_stream(
        tokenizer,
        model,
        args.prompt,
        use_kv_cache=not args.no_kv_cache,
        on_token=lambda t: print(t, end="", flush=True),
    )
    print("\n")

    print("=" * 64)
    print(" 流式推理结果")
    print("=" * 64)
    print(f"  输入 tokens   : {result['input_tokens']}")
    print(f"  输出 tokens   : {result['output_tokens']}")
    print(f"  总延迟         : {result['total_latency_ms']} ms")
    print(f"  TTFT (近似)   : {result['ttft_ms']} ms")
    print(f"  吞吐率         : {result['throughput_tps']} tokens/sec")
    print(f"  KV Cache命中   : {result['kv_cache_hit']}")
    if result.get("kv_cache_pages") is not None:
        print(f"  KV Page数      : {result['kv_cache_pages']}")
    if torch.cuda.is_available():
        print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print("=" * 64)
