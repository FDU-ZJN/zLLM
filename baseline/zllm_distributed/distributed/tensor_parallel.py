import asyncio
import threading
import time
from dataclasses import dataclass, field

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


@dataclass
class TensorParallelConfig:
    model_path: str
    prompts: list[str]
    max_new_tokens: int = 256
    dtype: str = "bfloat16"
    pipeline_parallel_size: int = 4      # 流水线并行：4 卡各持 1/4 层
    gpu_memory_utilization: float = 0.8
    enable_prefix_caching: bool = True
    max_model_len: int = 2048
    max_num_batched_tokens: int = 8192


class TensorParallelLauncher:
    """
    基于 vLLM AsyncLLMEngine 的流水线并行推理启动器。

    修复要点
    --------
    原实现每次调用 run_sync() 都会执行 asyncio.run()，这会创建并销毁一个
    新的 event loop。AsyncLLMEngine 内部的协程、ZMQ 套接字、后台任务全部
    绑定在第一个 loop 上，第二次调用时旧 loop 已关闭，engine 随之"死亡"，
    抛出 EngineDeadError。

    解决方案：在 __init__ 阶段创建一个专用的 **持久后台 event loop**，用
    独立守护线程驱动它持续运行（run_forever）。Engine 初始化和所有推理请求
    都通过 asyncio.run_coroutine_threadsafe() 提交到同一个 loop，从根本上
    消除跨 loop 问题。
    """

    def __init__(self, config: TensorParallelConfig):
        self.config = config
        self._engine: AsyncLLMEngine | None = None

        # ------------------------------------------------------------------ #
        # 核心修复：创建持久后台 event loop，伴随进程生命周期存在              #
        # ------------------------------------------------------------------ #
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="vllm-engine-loop",
            daemon=True,          # 主进程退出时自动终止
        )
        self._loop_thread.start()

        # 用于保证 engine 只初始化一次的锁（线程安全）
        self._init_lock = threading.Lock()
        self._engine_ready = False

    # ---------------------------------------------------------------------- #
    # 内部工具                                                                #
    # ---------------------------------------------------------------------- #

    def _submit(self, coro):
        """
        将协程提交到持久后台 loop 并阻塞等待结果。
        适合从任意同步上下文（主线程、Jupyter、pytest）调用。
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _async_init_engine(self) -> None:
        """在持久 loop 内初始化 AsyncLLMEngine（只执行一次）。"""
        engine_args = AsyncEngineArgs(
            model=self.config.model_path,
            dtype=self.config.dtype,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=self.config.enable_prefix_caching,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            kv_cache_dtype="auto",
            disable_log_stats=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _ensure_loaded(self) -> None:
        """
        线程安全地确保 engine 已在持久 loop 中完成初始化。
        多线程并发调用时只有第一个线程真正执行初始化，其余等待完成。
        """
        if self._engine_ready:
            return
        with self._init_lock:
            if self._engine_ready:   # double-checked locking
                return
            # 阻塞，直到 engine 初始化完毕
            self._submit(self._async_init_engine())
            self._engine_ready = True

    @staticmethod
    def _parse_metrics(metrics, wall_ms: float) -> dict:
        """
        安全解析 vLLM RequestMetrics。
        AsyncLLMEngine 下 metrics 字段可能为 None。
        """
        if metrics is None:
            return {"prefill_ms": 0.0, "decode_ms": 0.0, "total_ms": wall_ms}

        def _diff_ms(t_end, t_start) -> float:
            if t_end is None or t_start is None:
                return 0.0
            return max((t_end - t_start) * 1000.0, 0.0)

        prefill_ms = _diff_ms(
            getattr(metrics, "first_token_time", None),
            getattr(metrics, "first_scheduled_time", None),
        )
        total_ms = _diff_ms(
            getattr(metrics, "finished_time", None),
            getattr(metrics, "first_scheduled_time", None),
        )
        decode_ms = max(total_ms - prefill_ms, 0.0)

        return {
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms":  round(decode_ms, 2),
            "total_ms":   round(total_ms, 2) if total_ms > 0 else wall_ms,
        }

    async def _generate_one(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ):
        """对单条 prompt 调用 AsyncLLMEngine，持续消费直到 finished=True。"""
        final_output = None
        async for output in self._engine.generate(
            prompt, sampling_params, request_id=request_id
        ):
            final_output = output
        return final_output

    # ---------------------------------------------------------------------- #
    # 公开接口                                                                #
    # ---------------------------------------------------------------------- #

    async def run(
        self,
        prompts: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """
        并发提交所有 prompt，等待全部完成后返回结果列表。
        AsyncLLMEngine 内部自动做连续批处理和流水线调度。

        注意：此方法必须在持久后台 loop（self._loop）中执行，
              请勿在外部 event loop 中直接 await，应通过 run_sync() 调用。
        """
        batch_prompts = self.config.prompts if prompts is None else prompts
        if not batch_prompts:
            return []

        n_tokens = int(
            self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        sampling_params = SamplingParams(
            max_tokens=n_tokens,
            temperature=0.0,    # greedy，对应原来的 do_sample=False
        )

        t_wall_start = time.perf_counter()

        # 并发提交：asyncio.gather 让所有请求同时进入引擎队列
        tasks = [
            self._generate_one(
                request_id=str(idx),
                prompt=p,
                sampling_params=sampling_params,
            )
            for idx, p in enumerate(batch_prompts)
        ]
        outputs = await asyncio.gather(*tasks)

        t_wall_end = time.perf_counter()
        wall_ms = (t_wall_end - t_wall_start) * 1000.0

        results = []
        for idx, output in enumerate(outputs):
            if output is None:
                continue

            gen     = output.outputs[0]
            out_len = len(gen.token_ids)

            m          = self._parse_metrics(output.metrics, wall_ms)
            total_ms   = m["total_ms"]
            throughput = (out_len / total_ms * 1000.0) if total_ms > 0 else 0.0

            results.append({
                "request_id":       idx,
                "prompt":           output.prompt,
                "output":           gen.text,
                "input_tokens":     len(output.prompt_token_ids),
                "output_tokens":    out_len,
                "total_latency_ms": round(total_ms, 2),
                "prefill_ms":       m["prefill_ms"],
                "decode_ms":        m["decode_ms"],
                "ttft_ms":          m["prefill_ms"],   # Time-To-First-Token ≈ prefill
                "throughput_tps":   round(throughput, 2),
                "kv_cache_hit":     False,
                "mode":             "vllm_pipeline_parallel",
            })

        return results

    def run_sync(
        self,
        prompts: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """
        同步入口，供脚本或非 async 上下文直接调用。

        修复说明
        --------
        原实现在每次调用时执行 asyncio.run()，新建并销毁 event loop，
        导致 AsyncLLMEngine 第二次使用时因 loop 已关闭而抛出 EngineDeadError。

        现在所有调用都通过 asyncio.run_coroutine_threadsafe() 提交到
        __init__ 中创建的同一个持久后台 loop，彻底规避跨 loop 问题。
        Engine 与 loop 的生命周期完全一致。
        """
        self._ensure_loaded()
        return self._submit(self.run(prompts, max_new_tokens))

    async def kv_cache_stats(self) -> dict:
        if self._engine is None:
            return {"enabled": False}
        try:
            cache_config = self._engine.engine.cache_config
            return {
                "enabled":                True,
                "num_gpu_blocks":         getattr(cache_config, "num_gpu_blocks", None),
                "num_cpu_blocks":         getattr(cache_config, "num_cpu_blocks", None),
                "prefix_caching":         self.config.enable_prefix_caching,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
            }
        except Exception:
            return {
                "enabled":                True,
                "prefix_caching":         self.config.enable_prefix_caching,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
            }

    def get_kv_cache_stats_sync(self) -> dict:
        """kv_cache_stats 的同步版本，与 run_sync 使用同一持久 loop。"""
        self._ensure_loaded()
        return self._submit(self.kv_cache_stats())

    def shutdown(self) -> None:
        """
        主动关闭持久 loop 和后台线程。
        正常情况下可不调用（daemon 线程随主进程退出），
        测试或需要显式释放资源时使用。
        """
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=10)


# --------------------------------------------------------------------------- #
# 使用示例                                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = TensorParallelConfig(
        model_path="path/to/Qwen3.5-122B",
        prompts=[
            "Explain the difference between pipeline and tensor parallelism.",
            "What is PagedAttention?",
        ],
        max_new_tokens=256,
        pipeline_parallel_size=4,
        max_num_batched_tokens=4096,
    )

    launcher = TensorParallelLauncher(cfg)

    try:
        # 第一次调用：初始化 engine 并推理
        results = launcher.run_sync()
        for r in results:
            print(
                f"[{r['request_id']}] {r['output_tokens']} tokens | "
                f"latency {r['total_latency_ms']:.1f} ms | "
                f"TTFT {r['ttft_ms']:.1f} ms | "
                f"{r['throughput_tps']:.1f} tok/s"
            )
            print(r["output"][:200], "\n")

        # 第二次调用：复用已有 engine，不再崩溃
        results2 = launcher.run_sync(
            prompts=["What are the benefits of KV cache?"],
            max_new_tokens=128,
        )
        for r in results2:
            print(r["output"][:200])

        # 查询 KV cache 状态
        stats = launcher.get_kv_cache_stats_sync()
        print("KV cache stats:", stats)

    finally:
        launcher.shutdown()