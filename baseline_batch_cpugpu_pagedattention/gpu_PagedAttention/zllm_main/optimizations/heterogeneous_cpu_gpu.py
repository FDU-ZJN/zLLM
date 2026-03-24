"""
heterogeneous_cpu_gpu.py
==========================
**编排式 CPU/GPU 异构**（本仓库主推语义）：不把模型按层切开，而是

- **CPU**：请求调度、动态 batch、KV 元数据与 offload 策略、decode 阶段的轻量算
  （如 greedy 的 argmax / 将来 top-k、p）、host 侧缓存与拼接；
- **GPU**：Transformer 矩阵乘与 Attention 等核心算子。

这与 ``heterogeneous_pipeline`` 中的 **层切分 + CPU/GPU 边界异步 H2D** 是两条线：
后者用 ``device_map`` 把前 N 层放 CPU，属于「显存换算力」的另一种手段。

典型动机（与实现对应关系）：

1. 调度与 batching：见 ``dynamic_batch_scheduler``、``infer_continuous_batch``。
2. KV：Paged KV、``KVCacheOffloadManager``、``MemoryOptimizer`` 在显存压力下触发 LRU 卸到 CPU。
3. decode 轻量算：可选 ``decode_logits_on_cpu``，最后一维 logits 在 CPU 上 argmax，减轻 GPU 上碎片化小算子。

本模块仅集中文档与薄封装，避免与 ``device_map`` / Qwen2 patch 混淆。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_optimizer import MemoryOptimizer
    from .kv_cache.kv_offload_manager import KVCacheOffloadManager


def attach_kv_offload_to_memory_optimizer(
    memory_optimizer: "MemoryOptimizer",
    kv_offload_manager: "KVCacheOffloadManager | None",
) -> None:
    """让 ``MemoryOptimizer`` 在显存压力下能调用 ``KVCacheOffloadManager.evict_lru_to_cpu``。"""
    memory_optimizer.kv_offload_manager = kv_offload_manager
