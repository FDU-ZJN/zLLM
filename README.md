# LLM 推理优化实训项目 - 完整目录详解

---

## 📋 项目概述

本项目是一个面向大语言模型（LLM）高效推理框架的系统级优化实训平台。项目提供了从基线实现到多种优化策略的完整代码框架，涵盖**KV Cache 管理**、**动态批处理**、**内存优化**、**量化技术**（AWQ/GPTQ/HQQ）、**分布式张量并行**等关键技术方向。

### 核心优化技术

1. **PagedAttention** - 分页式 KV Cache 管理，提升显存利用率 30-50%
2. **Dynamic Batching** - 动态请求合并，提升吞吐量 2-3 倍
3. **Tensor Parallel** - 多卡张量并行，支持超大模型推理
4. **Quantization** - 4bit/8bit 量化，减少显存占用 50-75%

### 学习目标

通过本项目，学习者可以：
- 理解 LLM 推理的基本原理和性能瓶颈
- 掌握主流推理优化技术（量化、PagedAttention、张量并行等）
- 实践系统级优化的完整流程（基准测试 → 优化实现 → 效果评估）
- 深入理解每个优化模块的核心代码实现

---

## 🗂️ 完整目录结构详解

```
ours_project/
│
├── README.md                              # 本文件：项目完整目录详解
├── 实验指标记录.xlsx                       # 实验数据记录表（Excel 格式）
│
├── baseline/                              # 【基线目录】基础推理实现和基准测试
│   ├── README.md                         # baseline 子目录说明
│   ├── baseline_inference.py             # 单条 prompt 朴素推理实现
│   ├── benchmark.py                      # 批量吞吐量 & 延迟基准测试
│   ├── evaluate_accuracy.py              # C-Eval 精度评测脚本
│   ├── vllm_inference.py                 # vLLM 推理实现参考
│   ├── prompts.jsonl                     # 统一测试 prompt 集合（11KB）
│   ├── ceval_part.jsonl                  # C-Eval 部分测试集（45KB）
│   ├── ceval_subset.jsonl                # C-Eval 完整测试集（402KB）
│   ├── requirements.txt                  # Python 依赖列表
│   ├── results_baseline.json             # 基线性能测试结果
│   ├── core                              # 核心文件（可能是模型或数据，1.5MB）
│   │
│   ├── .sii/                             # SII 配置目录（版本控制相关）
│   ├── .vscode/                          # VSCode 配置目录
│   ├── __pycache__/                      # Python 字节码缓存
│   │
│   ├── submmit/                          # 【提交版本】优化的最终提交代码
│   │   ├── benchmark.py                  # 性能基准测试脚本
│   │   ├── evaluate_accuracy.py          # 精度评估脚本
│   │   ├── vllm_inference.py             # vLLM 推理集成（11.5KB）
│   │   ├── zllm_inference_awq.py         # AWQ 量化推理实现（21KB）
│   │   ├── accuracy_baseline.json        # 基线精度结果
│   │   ├── test.sh                       # 测试运行脚本
│   │   ├── zllm.sh                       # ZLLM 主运行脚本
│   │   ├── zllm_accuracy.sh              # 精度测试脚本
│   │   ├── zllm_benchmark.sh             # 性能测试脚本
│   │   │
│   │   ├── optimizations/                # 【核心优化模块】
│   │   │   ├── dynamic_batch_scheduler.py    # 动态批处理调度器（16KB）
│   │   │   ├── memory_optimizer.py           # 内存优化器（1.9KB）
│   │   │   ├── __init__.py                   # 包初始化
│   │   │   │
│   │   │   └── kv_cache/                   # 【KV Cache 优化专题】
│   │   │       ├── paged_kv_cache_manager.py     # PagedAttention 实现（15KB）
│   │   │       ├── simple_kv_cache_manager.py    # 简单 KV Cache 管理（3.6KB）
│   │   │       ├── paged_kv_cache_manage_backup.py  # PagedAttention 备份
│   │   │       └── __init__.py
│   │   │
│   │   ├── .sii/
│   │   └── __pycache__/
│   │
│   ├── zllm_distributed/                 # 【分布式推理】张量并行实现
│   │   ├── zllm_inference.py             # 分布式推理主程序（4.1KB）
│   │   ├── zllm_drain_inference.py       # Drain 模式推理（14.5KB）
│   │   ├── zllm_inference copy.py        # 推理脚本副本（18.8KB）
│   │   ├── benchmark.py                  # 分布式性能测试（9.7KB）
│   │   ├── evaluate_accuracy.py          # 精度评估
│   │   ├── accuracy_baseline.json        # 精度基线（0.8KB）
│   │   ├── test.sh                       # 测试脚本
│   │   ├── zllm.sh                       # 运行脚本
│   │   ├── zllm_accuracy.sh              # 精度测试脚本
│   │   ├── zllm_benchmark.sh             # 性能测试脚本
│   │   │
│   │   ├── distributed/                  # 【分布式核心模块】
│   │   │   ├── tensor_parallel.py        # 张量并行启动器（12KB）
│   │   │   └── __init__.py
│   │   │
│   │   ├── optimizations/                # 优化模块（同 submmit）
│   │   │   ├── kv_cache/
│   │   │   ├── dynamic_batch_scheduler.py
│   │   │   └── memory_optimizer.py
│   │   │
│   │   ├── .sii/
│   │   └── __pycache__/
│   │
│   ├── zllm_main/                        # 【主分支】主要优化实现
│   │   ├── vllm_inference.py             # vLLM 推理（16.4KB）
│   │   ├── benchmark.py                  # 性能测试（6.3KB）
│   │   ├── evaluate_accuracy.py          # 精度评估
│   │   ├── accuracy_baseline.json        # 精度基线
│   │   ├── test.tar.gz                   # 测试数据包（64KB）
│   │   ├── zllm.sh                       # 运行脚本
│   │   ├── zllm_accuracy.sh              # 精度测试脚本
│   │   ├── zllm_benchmark.sh             # 性能测试脚本
│   │   │
│   │   ├── optimizations/                # 优化模块
│   │   │   ├── kv_cache/
│   │   │   ├── dynamic_batch_scheduler.py
│   │   │   └── memory_optimizer.py
│   │   │
│   │   ├── .sii/
│   │   └── __pycache__/
│   │
│   └── zllm_quant/                       # 【量化专用】量化优化实现
│       ├── zllm_inference_awq.py         # AWQ 量化推理（21KB）
│       ├── benchmark.py                  # 性能测试
│       ├── evaluate_accuracy.py          # 精度评估
│       ├── accuracy_baseline.json        # 精度基线
│       ├── zllm.tar.gz                   # 量化模型包（59KB）
│       ├── test.sh                       # 测试脚本
│       ├── zllm_accuracy.sh              # 精度测试脚本
│       ├── zllm_benchmark.sh             # 性能测试脚本
│       │
│       ├── optimizations/                # 优化模块
│       │   ├── kv_cache/
│       │   ├── dynamic_batch_scheduler.py
│       │   └── memory_optimizer.py
│       │
│       ├── .sii/
│       └── __pycache__/
│
├── quantize/                              # 【量化主目录】量化相关优化
│   ├── README.md                         # quantize 子目录说明
│   │
│   ├── HQQ.py                            # HQQ 量化实现（2.3KB）
│   ├── awq.py                            # AWQ 量化实现（1KB）
│   ├── gptq.py                           # GPTQ 量化实现（5.7KB）
│   ├── baseline_inference.py             # 基线推理（4.1KB）
│   ├── benchmark.py                      # 基准测试（11.8KB）
│   ├── benchmark_hqq.py                  # HQQ 量化基准测试（11.9KB）
│   ├── benchmark_l.py                    # 基准测试变体（10.8KB）
│   ├── evaluate_accuracy.py              # 精度评估（5.7KB）
│   ├── evaluate_accuracy_hqq.py          # HQQ 精度评估（6.3KB）
│   ├── evaluate_accuracy_l.py            # 精度评估变体（11.9KB）
│   ├── eva_vllm.py                       # vLLM 评估工具（1.7KB）
│   ├── pre_data.py                       # 数据预处理（1.3KB）
│   ├── train.py                          # 模型训练脚本（13.6KB）
│   │
│   ├── vllm_inference.py                 # vLLM 推理（3.2KB）
│   ├── ceval_converted.json              # C-Eval 转换格式数据（625KB）
│   ├── ceval_part.jsonl                  # C-Eval 部分测试集（45KB）
│   ├── ceval_subset.jsonl                # C-Eval 完整测试集（402KB）
│   ├── prompts.jsonl                     # 测试 prompt 集合（11KB）
│   ├── requirements.txt                  # 依赖列表
│   │
│   ├── results_baseline.json             # 基线性能结果
│   ├── results_vllm.json                 # vLLM 性能结果
│   ├── accuracy_baseline.json            # 基线精度结果
│   ├── core                              # 核心文件（1.5MB）
│   ├── test.txt                          # 测试文本文件（1.1MB）
│   │
│   ├── submmit/                          # 【提交版本】同 baseline/submmit
│   │   ├── optimizations/
│   │   │   ├── kv_cache/
│   │   │   ├── dynamic_batch_scheduler.py
│   │   │   └── memory_optimizer.py
│   │   ├── benchmark.py
│   │   ├── evaluate_accuracy.py
│   │   ├── vllm_inference.py             #最好结果vllm推理版本
│   │   └── *.sh
│   │
│   ├── zllm_distributed/                 # 【分布式】同 baseline/zllm_distributed
│   │   ├── distributed/
│   │   │   └── tensor_parallel.py
│   │   ├── optimizations/
│   │   ├── zllm_inference.py
│   │   ├── zllm_drain_inference.py
│   │   └── *.sh
│   │
│   ├── zllm_main/                        # 【主分支】同 baseline/zllm_main
│   │   ├── optimizations/
│   │   ├── vllm_inference.py
│   │   ├── benchmark.py
│   │   └── *.sh
│   │
│   ├── zllm_quant/                       # 【量化专用】同 baseline/zllm_quant
│   │   ├── optimizations/
│   │   ├── zllm_inference_awq.py
│   │   ├── benchmark.py
│   │   └── *.sh
│   │
│   ├── .sii/
│   ├── .vscode/
│   └── __pycache__/
│
└── baseline_batch_cpugpu_pagedattention/  # 【CPU+GPU 批处理 +PagedAttention】
    └── gpu_PagedAttention/               # GPU 版 PagedAttention 实现
        ├── README.md                     # 子目录说明
        ├── baseline_inference.py         # 基线推理
        ├── benchmark.py                  # 性能测试
        ├── evaluate_accuracy.py          # 精度评估
        ├── vllm_inference.py             # vLLM 推理
        │
        ├── ceval_part.jsonl              # C-Eval 部分测试集
        ├── ceval_subset.jsonl            # C-Eval 完整测试集
        ├── prompts.jsonl                 # Prompt 集合（1.1KB）
        ├── requirements.txt              # 依赖列表
        │
        ├── results_baseline.json         # 基线结果
        ├── sig.json                      # 签名文件
        ├── core                          # 核心文件（1.5MB）
        │
        ├── baseline.tar                  # 基线包（10KB）
        ├── mbaseline.tar.gz              # 基线压缩包
        │
        ├── baseline_/                    # 基线目录
        │   └── （包含 9 个文件）
        │
        ├── zllm_main/                    # 主分支实现（17 个文件）
        ├── .sii/
        └── __pycache__/
```


---

## 📁 各目录详细介绍

### 1️⃣ `baseline/` - 基线实现目录

**功能定位**：提供 LLM 推理的基础实现和性能基准，作为所有优化的对比基线。

**核心文件详解**：

#### 🔹 [`baseline_inference.py`](baseline/baseline_inference.py) (3.9KB)
- **功能**：单条 prompt 的朴素推理实现
- **关键参数**：
  - `DEVICE = "cuda:0"` - 推理设备
  - `DTYPE = torch.float16` - 数据类型
  - `MAX_NEW_TOKENS = 256` - 最大生成长度
- **核心代码解析**：
  ```python
  # 模型加载
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=DTYPE,
      device_map=DEVICE
  )
  
  # Tokenizer 配置
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  
  # 推理生成
  inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
  outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
  ```
- **使用方法**：
  ```bash
  python baseline_inference.py --model_path /path/to/model
  ```
- **输出指标**：总延迟、吞吐率、峰值显存
- **性能参考**：
  - 延迟：~3800ms
  - 吞吐率：~28.4 tokens/sec
  - 显存：~29.0 GB

#### 🔹 [`benchmark.py`](baseline/benchmark.py) (4.9KB)
- **功能**：批量性能基准测试
- **测试流程**：
  1. 加载 `prompts.jsonl` 中的所有测试 prompt
  2. 逐条推理并记录延迟、TTFT、吞吐量
  3. 计算统计指标（平均值、P50、P95、P99）
  4. 记录峰值显存占用
- **核心代码解析**：
  ```python
  # 延迟统计
  latencies = []
  for prompt in prompts:
      start_time = time.time()
      output = infer_single(prompt)
      latency = (time.time() - start_time) * 1000
      latencies.append(latency)
  
  # 百分位计算
  p50 = np.percentile(latencies, 50)
  p95 = np.percentile(latencies, 95)
  p99 = np.percentile(latencies, 99)
  
  # 吞吐率计算
  throughput = total_output_tokens / wall_time_seconds
  ```
- **输出文件**：`results_baseline.json`
- **关键指标**：
  - `overall_throughput_tps` - 整体吞吐量
  - `avg_latency_ms` - 平均延迟
  - `p95_latency_ms` - 95 分位延迟
  - `peak_gpu_mem_gb` - 峰值显存
- **使用方法**：
  ```bash
  python benchmark.py --model_path /path/to/model --output results_baseline.json
  ```

#### 🔹 [`evaluate_accuracy.py`](baseline/evaluate_accuracy.py) (5.7KB)
- **功能**：C-Eval 数据集精度评估
- **评估流程**：
  1. 读取 `ceval_subset.jsonl` 测试数据
  2. 对每个问题生成答案
  3. 对比标准答案计算准确率
- **核心代码解析**：
  ```python
  # 加载测试数据
  with open(eval_file, 'r', encoding='utf-8') as f:
      test_data = [json.loads(line) for line in f]
  
  # 逐题评测
  correct = 0
  for item in test_data:
      question = item['question']
      answer = item['answer']
      pred = infer_single(question)
      if pred == answer:
          correct += 1
  
  accuracy = correct / len(test_data) * 100
  ```
- **输出文件**：`accuracy_baseline.json`
- **使用方法**：
  ```bash
  python evaluate_accuracy.py \
    --model_path /path/to/model \
    --eval_file ceval_subset.jsonl \
    --output accuracy_baseline.json
  ```

#### 🔹 [`vllm_inference.py`](baseline/vllm_inference.py) (3.2KB)
- **功能**：基于 vLLM 框架的推理实现
- **用途**：作为高性能推理的参考基线
- **核心代码解析**：
  ```python
  from vllm import LLM, SamplingParams
  
  # 初始化 vLLM 引擎
  llm = LLM(model=model_path)
  
  # 配置采样参数
  sampling_params = SamplingParams(
      temperature=0.7,
      max_tokens=256
  )
  
  # 批量推理
  outputs = llm.generate(prompts, sampling_params)
  ```
- **使用方法**：
  ```bash
  python vllm_inference.py --model_path /path/to/model --output results_vllm.json
  ```

**测试数据文件**：
- `prompts.jsonl` (11KB) - 统一测试 prompt 集合，包含 100+ 条测试用例
- `ceval_subset.jsonl` (402KB) - C-Eval 完整测试集，包含多个学科题目
- `ceval_part.jsonl` (45KB) - C-Eval 部分测试集（快速测试用）

**配置文件**：
- `requirements.txt` - Python 依赖列表
  ```txt
  torch>=2.0.0
  transformers>=4.35.0
  accelerate
  vllm
  autoawq
  hqq
  ```

**结果文件**：
- `results_baseline.json` - 基线性能测试结果
  ```json
  {
    "total_prompts": 100,
    "total_output_tokens": 12800,
    "wall_time_sec": 450.23,
    "overall_throughput_tps": 28.43,
    "avg_latency_ms": 3842.10,
    "p50_latency_ms": 3756.45,
    "p95_latency_ms": 4521.89,
    "p99_latency_ms": 4892.34,
    "avg_ttft_ms": 245.67,
    "peak_gpu_mem_gb": 29.021
  }
  ```

---

### 2️⃣ `baseline/submmit/` - 提交版本（优化实现）

**功能定位**：包含所有优化技术的最终提交版本，是项目的核心成果。

**核心优化模块**：

#### 🔹 [`optimizations/dynamic_batch_scheduler.py`](baseline/submmit/optimizations/dynamic_batch_scheduler.py) (16KB)
- **功能**：动态批处理调度器
- **核心机制**：
  - **请求队列管理**：维护待处理请求队列
  - **智能调度策略**：
    - 当队列长度 ≥ `max_batch_size` 时立即调度
    - 当最早请求等待时间 ≥ `max_wait_ms` 时调度
  - **Token 预算控制**：确保批次总 token 数不超过 `max_prompt_tokens`
- **核心代码解析**：
  ```python
  class DynamicBatchScheduler:
      def __init__(self, max_batch_size=16, max_wait_ms=20):
          self.max_batch_size = max_batch_size
          self.max_wait_ms = max_wait_ms
          self.request_queue = []
      
      def should_schedule(self):
          # 调度条件 1: 队列满
          if len(self.request_queue) >= self.max_batch_size:
              return True
          
          # 调度条件 2: 等待超时
          if self.request_queue:
              wait_time = (time.time() - self.request_queue[0].arrival_time) * 1000
              if wait_time >= self.max_wait_ms:
                  return True
          
          return False
      
      def schedule_batch(self):
          batch = self.request_queue[:self.max_batch_size]
          self.request_queue = self.request_queue[self.max_batch_size:]
          return batch
  ```
- **关键参数**：
  - `max_batch_size = 16` - 最大批处理大小
  - `max_wait_ms = 20` - 最大等待时间（毫秒）
  - `max_prompt_tokens = 16384` - 最大 prompt token 数
- **性能收益**：提升吞吐量 2-3 倍
- **使用方法**：
  ```python
  scheduler = DynamicBatchScheduler(max_batch_size=16)
  scheduler.add_request(prompt)
  if scheduler.should_schedule():
      batch = scheduler.schedule_batch()
      process_batch(batch)
  ```

#### 🔹 [`optimizations/kv_cache/paged_kv_cache_manager.py`](baseline/submmit/optimizations/kv_cache/paged_kv_cache_manager.py) (15KB)
- **功能**：PagedAttention KV Cache 管理器
- **核心机制**：
  - **分页存储**：将 KV Cache 切分为固定大小的 page（默认 16 tokens/page）
  - **非连续分配**：page 在显存中非连续存放，减少碎片
  - **LRU 淘汰**：基于 LRU 算法淘汰旧 page
  - **前缀共享**：相同前缀的序列共享 page，减少重复存储
- **核心代码解析**：
  ```python
  class PagedKVCacheManager:
      def __init__(self, max_entries=16, page_size_tokens=16):
          self.page_size = page_size_tokens
          self.max_entries = max_entries
          self.page_table = {}  # seq_id -> [page_ids]
          self.free_pages = []
          self.lru_queue = deque()
      
      def allocate_page(self):
          if not self.free_pages:
              self._evict_lru_page()
          return self.free_pages.pop()
      
      def _evict_lru_page(self):
          # LRU 淘汰最久未使用的 page
          oldest_seq = self.lru_queue.popleft()
          page_id = self.page_table[oldest_seq].pop()
          self.free_pages.append(page_id)
      
      def enable_prefix_sharing(self, seq1, seq2):
          # 共享相同前缀的 page
          common_prefix_len = self._compute_common_prefix(seq1, seq2)
          shared_pages = common_prefix_len // self.page_size
          # 复用 page_table 中的前缀 page
  ```
- **关键参数**：
  - `max_entries = 16` - 最大缓存条目数
  - `max_cache_tokens = 32768` - 最大缓存 token 数
  - `page_size_tokens = 16` - 每页 token 数
  - `enable_prefix_sharing = True` - 启用前缀共享
- **性能收益**：显存利用率提升 30-50%
- **使用方法**：
  ```python
  cache_manager = PagedKVCacheManager(page_size_tokens=16)
  cache_manager.allocate(seq_id, tokens)
  kv_cache = cache_manager.get_kv_cache(seq_id)
  ```

#### 🔹 [`optimizations/kv_cache/simple_kv_cache_manager.py`](baseline/submmit/optimizations/kv_cache/simple_kv_cache_manager.py) (3.6KB)
- **功能**：简化版 KV Cache 管理器
- **适用场景**：固定长度序列、资源充足场景
- **特点**：实现简单，开销小
- **核心代码解析**：
  ```python
  class SimpleKVCacheManager:
      def __init__(self, max_seq_len=2048):
          self.cache = {}
          self.max_seq_len = max_seq_len
      
      def store(self, seq_id, k_cache, v_cache):
          self.cache[seq_id] = (k_cache, v_cache)
      
      def retrieve(self, seq_id):
          return self.cache.get(seq_id)
      
      def clear(self, seq_id):
          if seq_id in self.cache:
              del self.cache[seq_id]
  ```

#### 🔹 [`optimizations/memory_optimizer.py`](baseline/submmit/optimizations/memory_optimizer.py) (1.9KB)
- **功能**：内存优化器
- **优化策略**：
  - 显存池管理
  - 惰性分配
  - 垃圾回收优化
- **核心代码解析**：
  ```python
  class MemoryOptimizer:
      def __init__(self):
          self.memory_pool = {}
          self.lazy_alloc = True
      
      def allocate_lazy(self, size):
          # 惰性分配：仅在真正使用时分配显存
          if self.lazy_alloc:
              return LazyTensor(size)
          else:
              return torch.empty(size, device='cuda')
      
      def garbage_collect(self):
          # 手动触发垃圾回收
          torch.cuda.empty_cache()
          gc.collect()
      
      def get_memory_stats(self):
          return {
              'allocated': torch.cuda.memory_allocated(),
              'cached': torch.cuda.memory_reserved()
          }
  ```
- **性能收益**：降低峰值显存 10-20%

#### 🔹 [`zllm_inference_awq.py`](baseline/submmit/zllm_inference_awq.py) (21KB)
- **功能**：AWQ 量化推理实现
- **量化类型**：4bit/8bit AWQ 量化
- **特点**：激活值感知的权重量化，保持重要权重精度
- **核心代码解析**：
  ```python
  from awq import AutoAWQForCausalLM
  
  # 加载 AWQ 量化模型
  model = AutoAWQForCausalLM.from_quantized(
      model_path,
      w_bit=4,  # 4bit 量化
      group_size=128,
      zero_point=True
  )
  
  # 量化推理
  outputs = model.generate(
      input_ids,
      max_new_tokens=256,
      temperature=0.7
  )
  ```
- **使用方法**：
  ```bash
  python zllm_inference_awq.py \
    --model_path /path/to/Qwen2.5-14B-Instruct-AWQ \
    --w_bit 4 \
    --output results_awq.json
  ```

**运行脚本**：
- `test.sh` - 通用测试脚本
  ```bash
  #!/bin/bash
  python benchmark.py --model_path $MODEL_PATH
  ```
- `zllm.sh` - ZLLM 主运行脚本
- `zllm_accuracy.sh` - 精度测试脚本
- `zllm_benchmark.sh` - 性能测试脚本

---

### 3️⃣ `baseline/zllm_distributed/` - 分布式推理目录

**功能定位**：实现多卡张量并行推理，支持超大模型（如 70B+）的推理。

**核心文件详解**：

#### 🔹 [`distributed/tensor_parallel.py`](baseline/zllm_distributed/distributed/tensor_parallel.py) (12KB)
- **功能**：张量并行启动器和核心逻辑
- **并行策略**：
  - **流水线并行** (`pipeline_parallel_size = 4`)：4 卡各持 1/4 层
  - **张量并行**：单层内的矩阵乘法拆分到多卡
- **核心修复**：
  - 创建**持久后台 event loop**，避免多次调用导致的 EngineDeadError
  - 使用独立守护线程驱动 event loop
  - 所有推理请求通过 `asyncio.run_coroutine_threadsafe()` 提交
- **核心代码解析**：
  ```python
  from vllm import LLM
  import asyncio
  from threading import Thread
  
  class TensorParallelEngine:
      def __init__(self, tensor_parallel_size=4):
          self.tp_size = tensor_parallel_size
          self.llm = LLM(
              model=model_path,
              tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=0.8
          )
          
          # 创建持久 event loop
          self.loop = asyncio.new_event_loop()
          self.loop_thread = Thread(target=self._run_event_loop, daemon=True)
          self.loop_thread.start()
      
      def _run_event_loop(self):
          asyncio.set_event_loop(self.loop)
          self.loop.run_forever()
      
      def generate(self, prompts):
          # 通过 event loop 提交请求
          future = asyncio.run_coroutine_threadsafe(
              self.llm.generate_async(prompts),
              self.loop
          )
          return future.result()
  ```
- **关键参数**：
  - `gpu_memory_utilization = 0.8` - GPU 显存利用率
  - `enable_prefix_caching = True` - 启用前缀缓存
  - `max_model_len = 2048` - 最大模型长度
  - `max_num_batched_tokens = 8192` - 最大批处理 token 数
- **使用方法**：
  ```bash
  python zllm_inference.py \
    --model_path /path/to/model \
    --tensor_parallel_size 4
  ```

#### 🔹 [`zllm_inference.py`](baseline/zllm_distributed/zllm_inference.py) (4.1KB)
- **功能**：分布式推理主程序
- **接口**：与 baseline 保持一致的单条推理接口 `infer_single()`
- **核心代码解析**：
  ```python
  def infer_single(prompt):
      engine = TensorParallelEngine(tensor_parallel_size=4)
      output = engine.generate([prompt])
      return output[0].text
  ```

#### 🔹 [`zllm_drain_inference.py`](baseline/zllm_distributed/zllm_drain_inference.py) (14.5KB)
- **功能**：Drain 模式推理
- **应用场景**：优雅关闭、模型切换等场景
- **核心机制**：
  - 停止接收新请求
  - 等待当前批次完成
  - 释放资源

**运行脚本**：
- `zllm.sh` - 启动分布式推理
  ```bash
  #!/bin/bash
  python zllm_inference.py \
    --model_path $MODEL_PATH \
    --tensor_parallel_size 4
  ```
- `test.sh` - 分布式测试

---

### 4️⃣ `baseline/zllm_main/` - 主分支目录

**功能定位**：主要的优化实现分支，整合了各项优化技术。

**核心文件**：
- [`vllm_inference.py`](baseline/zllm_main/vllm_inference.py) (16.4KB) - vLLM 推理集成
  - 支持流式输出
  - 支持多请求并发
- [`benchmark.py`](baseline/zllm_main/benchmark.py) (6.3KB) - 性能基准测试
  - 支持自定义测试参数
  - 输出详细性能报告
- `test.tar.gz` (64KB) - 测试数据包

**优化模块**：
- `optimizations/kv_cache/` - KV Cache 管理
- `optimizations/dynamic_batch_scheduler.py` - 动态批处理
- `optimizations/memory_optimizer.py` - 内存优化

---

### 5️⃣ `baseline/zllm_quant/` - 量化专用目录

**功能定位**：专注于量化技术的优化实现，包括 AWQ、GPTQ 等量化方案。

**核心文件**：
- [`zllm_inference_awq.py`](baseline/zllm_quant/zllm_inference_awq.py) (21KB) - AWQ 量化推理
  - 支持 4bit/8bit 量化
  - 激活值感知量化
- `zllm.tar.gz` (59KB) - 量化模型包

**优化模块**：
- `optimizations/kv_cache/` - KV Cache 管理
- `optimizations/dynamic_batch_scheduler.py` - 动态批处理
- `optimizations/memory_optimizer.py` - 内存优化

---

### 6️⃣ `quantize/` - 量化主目录

**功能定位**：量化相关的完整实现，包括多种量化技术和评估工具。

**量化技术文件**：

#### 🔹 [`HQQ.py`](quantize/HQQ.py) (2.3KB)
- **功能**：Half-Quadratic Quantization（半二次量化）实现
- **特点**：混合精度量化，通道级缩放因子
- **优势**：无需校准数据
- **核心代码解析**：
  ```python
  from hqq.core.quantize import BaseQuantizeConfig, HQQBackend
  
  # 配置量化参数
  quant_config = BaseQuantizeConfig(
      nbits=4,
      group_size=64,
      quant_zero=False,
      quant_scale=False,
      axis=1
  )
  
  # 应用量化
  model = HQQQuantizer.quantize_model(model, quant_config)
  ```

#### 🔹 [`awq.py`](quantize/awq.py) (1KB)
- **功能**：AWQ 量化实现入口
- **原理**：Activation-aware Weight Quantization
- **核心代码解析**：
  ```python
  from awq import AutoAWQForCausalLM
  
  # 量化模型
  quantized_model = AutoAWQForCausalLM.from_pretrained(model_path)
  quantized_model.quantize(
      calib_dataset=calib_data,
      w_bit=4,
      q_group_size=128
  )
  quantized_model.save_quantized(save_path)
  ```

#### 🔹 [`gptq.py`](quantize/gptq.py) (5.7KB)
- **功能**：GPTQ 量化实现
- **原理**：逐层量化 + 误差补偿
- **压缩比**：3-4bit 保持较好精度
- **核心代码解析**：
  ```python
  from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
  
  # 配置量化
  quantize_config = BaseQuantizeConfig(
      bits=4,
      group_size=128,
      desc_act=False
  )
  
  # 量化模型
  model = AutoGPTQForCausalLM.from_pretrained(
      model_path,
      quantize_config=quantize_config
  )
  model.quantize(calib_dataset)
  model.save_quantized(save_path)
  ```

**基准测试文件**：
- [`benchmark.py`](quantize/benchmark.py) (11.8KB) - 通用基准测试
- [`benchmark_hqq.py`](quantize/benchmark_hqq.py) (11.9KB) - HQQ 专用基准测试
- [`benchmark_l.py`](quantize/benchmark_l.py) (10.8KB) - 基准测试变体

**精度评估文件**：
- [`evaluate_accuracy.py`](quantize/evaluate_accuracy.py) (5.7KB) - 通用精度评估
- [`evaluate_accuracy_hqq.py`](quantize/evaluate_accuracy_hqq.py) (6.3KB) - HQQ 精度评估
- [`evaluate_accuracy_l.py`](quantize/evaluate_accuracy_l.py) (11.9KB) - 精度评估变体

**其他工具**：
- [`eva_vllm.py`](quantize/eva_vllm.py) (1.7KB) - vLLM 评估工具
- [`pre_data.py`](quantize/pre_data.py) (1.3KB) - 数据预处理
- [`train.py`](quantize/train.py) (13.6KB) - 模型训练脚本

**数据文件**：
- `ceval_converted.json` (625KB) - C-Eval 转换格式数据
- `test.txt` (1.1MB) - 测试文本文件

---

### 7️⃣ `baseline_batch_cpugpu_pagedattention/` - CPU+GPU 批处理 +PagedAttention

**功能定位**：探索 CPU 和 GPU 协同的批处理与 PagedAttention 结合方案。

**子目录**：

#### 🔹 `gpu_PagedAttention/` - GPU 版 PagedAttention 实现
- **核心文件**：
  - `baseline_inference.py` - 基线推理
  - `benchmark.py` - 性能测试
  - `evaluate_accuracy.py` - 精度评估
  - `vllm_inference.py` - vLLM 推理
- **数据文件**：
  - `prompts.jsonl` (1.1KB)
  - `ceval_subset.jsonl` (402KB)
  - `sig.json` - 签名文件
- **归档文件**：
  - `baseline.tar` (10KB)
  - `mbaseline.tar.gz`
- **子目录**：
  - `baseline_/` - 基线目录（9 个文件）
  - `zllm_main/` - 主分支实现（17 个文件）

---

## 🎯 评分指标体系

### 硬件配置
- **GPU**: 单卡 NVIDIA H100（或同等算力显卡）
- **显存**: ≥80GB（推荐）
- **CPU**: 多核处理器（建议 16 核以上）
- **内存**: ≥64GB

### 软件环境
- **Python**: 3.9+
- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.35+

### 模型要求
- **基础模型**: Qwen2.5-14B-Instruct
- **量化模型**: 
  - Qwen2.5-14B-Instruct-AWQ（4bit/8bit）
  - Qwen2.5-14B-Instruct-GPTQ（4bit/8bit）
  - Qwen2.5-14B-Instruct-HQQ（混合精度量化）

### 依赖安装

```bash
cd baseline
pip install -r requirements.txt
```

核心依赖包括：
- `torch` >= 2.0.0
- `transformers` >= 4.35.0
- `accelerate`
- `vllm` (可选，用于 vLLM 对比实验)
- `autoawq` (AWQ 量化支持)
- `hqq` (HQQ 量化支持)

---

## 🚀 快速开始

### 1. 基线推理验证

#### 单条 Prompt 推理
```bash
cd baseline
python baseline_inference.py --model_path /path/to/Qwen2.5-14B-Instruct
```

**预期输出**：
```
[INFO] 加载模型：/path/to/Qwen2.5-14B-Instruct
[INFO] 设备: cuda:0 | 数据类型：torch.float16
[INFO] 加载完成 | 参数量：14.00B | 显存占用：28.50 GB
[INFO] 生成完成 | 输出长度：128 tokens

=== 推理结果 ===
Prompt: 请用三句话解释大语言模型推理中 KV Cache 的作用。
Output: KV Cache 在大语言模型推理中扮演着关键角色...

=== 性能指标 ===
总延迟  : 3842.10 ms
吞吐率  : 28.4 tokens/sec
峰值显存：29.021 GB
```

#### 批量基准测试
```bash
python benchmark.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --output results_baseline.json
```

**输出示例** (`results_baseline.json`)：
```json
{
  "total_prompts": 100,
  "total_output_tokens": 12800,
  "wall_time_sec": 450.23,
  "overall_throughput_tps": 28.43,
  "avg_latency_ms": 3842.10,
  "p50_latency_ms": 3756.45,
  "p95_latency_ms": 4521.89,
  "p99_latency_ms": 4892.34,
  "avg_ttft_ms": 245.67,
  "peak_gpu_mem_gb": 29.021
}
```

### 2. 精度评估

```bash
python evaluate_accuracy.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --eval_file ceval_subset.jsonl \
  --output accuracy_baseline.json
```

**说明**：
- `ceval_subset.jsonl` 包含完整的 C-Eval 测试集
- 如需快速测试，可截取前 200-500 条数据
- 输出 `accuracy_baseline.json` 包含准确率结果

### 3. 量化优化推理

#### AWQ 量化推理
```bash
cd quantize/submmit
python zllm_inference_awq.py \
  --model_path /path/to/Qwen2.5-14B-Instruct-AWQ \
  --w_bit 4 \
  --output results_awq.json
```

#### HQQ 量化推理
```bash
cd quantize
python HQQ.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --bits 4 \
  --group_size 128
```

### 4. 分布式张量并行推理

```bash
cd quantize/zllm_distributed
python zllm_inference.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --tensor_parallel_size 2
```

### 5. vLLM 推理（对比基线）

```bash
cd baseline
python vllm_inference.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --output results_vllm.json
```

---

## 📊 评分指标体系

所有指标以**单卡 cuda:0** 结果为准。

| 指标 | 单位 | 方向 | 说明 | 基线参考值 |
|------|------|------|------|-----------|
| `overall_throughput_tps` | tokens/sec | ↑越高越好 | 总输出 token 数 / 总耗时 | ~28.4 |
| `p95_latency_ms` | ms | ↓越低越好 | 95 分位延迟 | ~4500 |
| `avg_latency_ms` | ms | ↓越低越好 | 单条请求平均端到端延迟 | ~3800 |
| `average_ttft_ms` | ms | ↓越低越好 | 首 token 延迟 (Time To First Token) | ~250 |
| `accuracy` | % | 损失≤5% | C-Eval 评测准确率 | 基准±5% |
| `peak_gpu_mem_gb` | GB | ↓越低越好 | 推理峰值显存占用 | ~29.0 |

### 指标计算说明

1. **整体吞吐率** (overall_throughput_tps):
   ```
   overall_throughput = total_output_tokens / wall_time_seconds
   ```

2. **延迟百分位**:
   - P50: 中位数延迟
   - P95: 95% 请求的延迟不超过此值
   - P99: 99% 请求的延迟不超过此值

3. **TTFT** (Time To First Token):
   - 从请求提交到第一个 token 生成的时间
   - 流式输出场景下的重要指标

---

## 🎯 优化方向与技术路线

### 1. KV Cache 优化

#### PagedAttention 实现
- **位置**: `quantize/*/optimizations/kv_cache/paged_kv_cache_manager.py`
- **原理**: 借鉴操作系统分页思想，非连续分配 KV Cache 块
- **收益**: 减少显存碎片，提升显存利用率 30-50%

#### 简单 KV Cache 管理
- **位置**: `quantize/*/optimizations/kv_cache/simple_kv_cache_manager.py`
- **适用场景**: 固定长度序列场景

### 2. 动态批处理 (Dynamic Batching)

- **位置**: `quantize/*/optimizations/dynamic_batch_scheduler.py`
- **功能**: 
  - 实时合并多个请求为一个 batch
  - 支持不同长度的序列
  - 自动调整 batch 大小
- **收益**: 提升 GPU 利用率，增加吞吐量 2-3 倍

### 3. 内存优化

- **位置**: `quantize/*/optimizations/memory_optimizer.py`
- **策略**:
  - 显存池管理
  - 惰性分配
  - 垃圾回收优化
- **收益**: 降低峰值显存 10-20%

### 4. 量化技术

#### AWQ (Activation-aware Weight Quantization)
- **位置**: `quantize/submmit/zllm_inference_awq.py`
- **特点**: 保留重要权重精度，激活值感知
- **压缩比**: 4bit 可达 FP16 精度的 99%

#### GPTQ (Generative Pre-trained Transformer Quantization)
- **位置**: `quantize/gptq.py`
- **特点**: 逐层量化，误差补偿
- **压缩比**: 3-4bit 保持较好精度

#### HQQ (Half-Quadratic Quantization)
- **位置**: `quantize/HQQ.py`
- **特点**: 混合精度量化，通道级缩放
- **优势**: 无需校准数据

### 5. 分布式张量并行

- **位置**: `quantize/zllm_distributed/distributed/tensor_parallel.py`
- **实现**: 
  - 模型权重切分到多卡
  - 并行矩阵乘法
  - 通信优化
- **扩展性**: 支持 2-8 卡并行

---

## 📁 核心文件说明

### Baseline 目录

| 文件 | 功能 | 行数 |
|------|------|------|
| [`baseline_inference.py`](baseline/baseline_inference.py) | 单条 prompt 推理验证 | ~127 |
| [`benchmark.py`](baseline/benchmark.py) | 批量性能基准测试 | ~153 |
| [`evaluate_accuracy.py`](baseline/evaluate_accuracy.py) | C-Eval 精度评估 | ~150+ |
| [`vllm_inference.py`](baseline/vllm_inference.py) | vLLM 推理集成 | ~100+ |

### Quantize 目录

| 文件/目录 | 功能 |
|----------|------|
| `submmit/` | 最终提交版本（包含所有优化） |
| `zllm_distributed/` | 分布式张量并行实现 |
| `zllm_main/` | 主分支优化代码 |
| `zllm_quant/` | 量化专用优化 |
| `optimizations/kv_cache/` | KV Cache 管理模块 |
| `dynamic_batch_scheduler.py` | 动态批处理调度器 |
| `memory_optimizer.py` | 内存优化器 |

---

## 🔬 实验流程

### Step 1: 建立基线

```bash
# 1. 运行基线推理
cd baseline
python baseline_inference.py --model_path $MODEL_PATH

# 2. 运行基线基准测试
python benchmark.py --model_path $MODEL_PATH --output results_baseline.json

# 3. 运行基线精度评估
python evaluate_accuracy.py \
  --model_path $MODEL_PATH \
  --eval_file ceval_subset.jsonl \
  --output accuracy_baseline.json
```

### Step 2: 实施优化

选择以下一个或多个优化方向：

#### 方案 A: KV Cache 优化
```bash
cd quantize/submmit
# 修改 paged_kv_cache_manager.py 实现 PagedAttention
# 运行测试
python benchmark.py --model_path $MODEL_PATH --output results_paged_attn.json
```

#### 方案 B: 量化优化
```bash
cd quantize
# 使用 AWQ 量化
python awq.py --model_path $MODEL_PATH --w_bit 4

# 运行量化模型推理
cd submmit
python zllm_inference_awq.py --model_path $MODEL_PATH_AWQ --output results_awq.json
```

#### 方案 C: 分布式并行
```bash
cd quantize/zllm_distributed
# 配置张量并行
python zllm_inference.py --model_path $MODEL_PATH --tensor_parallel_size 2
```

### Step 3: 对比分析

```bash
# 比较优化前后的性能指标
python -c "
import json
with open('results_baseline.json') as f1, open('results_optimized.json') as f2:
    baseline = json.load(f1)
    optimized = json.load(f2)
    
print(f'吞吐率提升：{(optimized[\"overall_throughput_tps\"] / baseline[\"overall_throughput_tps\"] - 1) * 100:.2f}%')
print(f'显存降低：{(baseline[\"peak_gpu_mem_gb\"] - optimized[\"peak_gpu_mem_gb\"]) / baseline[\"peak_gpu_mem_gb\"] * 100:.2f}%')
"
```

---

## 📝 代码提交要求

### 必需材料

1. **优化后的代码**
   - 可基于 baseline 修改，或新建目录
   - 保持代码结构清晰，注释完整

2. **README.md**
   - 项目简介
   - 运行方法
   - 优化技术说明

3. **流式输出演示**（可选）
   - 单独的脚本展示流式输出效果
   - 示例：`streaming_inference.py`

4. **性能对比数据**
   - `results_baseline.json` - 基线性能
   - `results_optimized.json` - 优化后性能

5. **精度对比数据**
   - `accuracy_baseline.json` - 基线精度
   - `accuracy_optimized.json` - 优化后精度

6. **实验报告**
   - 技术路线与原理
   - 实验设计与对比
   - 结果分析与总结

### 提交结构示例

```
submission/
├── src/                          # 优化后源代码
│   ├── inference.py
│   ├── optimizations/
│   └── ...
├── results/                      # 实验结果
│   ├── results_baseline.json
│   ├── results_optimized.json
│   ├── accuracy_baseline.json
│   └── accuracy_optimized.json
├── scripts/                      # 运行脚本
│   ├── run_baseline.sh
│   └── run_optimized.sh
├── streaming_demo/               # 流式输出演示
│   └── streaming_inference.py
├── README.md                     # 项目说明
└── report.pdf                    # 实验报告
```

---

## 🛠️ 常见问题 FAQ

### Q1: 显存不足怎么办？
**A**: 可以尝试以下方法：
1. 使用量化模型（4bit AWQ/GPTQ）
2. 启用 CPU offload（将部分权重放到 CPU 内存）
3. 使用梯度检查点技术
4. 减小 `max_new_tokens` 配置

### Q2: 如何选择合适的量化位数？
**A**: 
- **4bit**: 最高压缩比，适合资源受限场景
- **8bit**: 平衡精度和压缩比，推荐生产使用
- **混合精度**: 关键层保持 FP16，其他层量化

### Q3: PagedAttention 相比传统 KV Cache 有什么优势？
**A**:
- 显存利用率提升 30-50%
- 支持更长的上下文窗口
- 减少显存碎片
- 更好的并发性能

### Q4: 动态批处理的 batch size 如何选择？
**A**:
- 受限于显存容量
- 受限于序列长度分布
- 建议通过 auto-tuning 找到最优值
- 典型范围：8-64

### Q5: 精度下降过多如何解决？
**A**:
1. 检查量化参数设置
2. 使用更大量化位数（如 4bit → 8bit）
3. 对敏感层保持 FP16 精度
4. 增加校准数据集规模

---

## 📚 参考资料

### 论文
- [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/pdf/2309.06180)
- [AWQ: Activation-aware Weight Quantization for LLM Compression](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [FlashAttention: Fast and Memory-Efficient Exact Attention from IO-Awareness](https://arxiv.org/abs/2205.14135)

### 开源项目
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理框架
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ 实现的 LLM 推理
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - AWQ 量化工具
- [HQQ](https://github.com/mobiusml/hqq) - Half-Quadratic Quantization
- [LMCache](https://github.com/LMCache/LMCache) - KV Cache 引擎

### 文档
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [NVIDIA CUDA 编程指南](https://developer.nvidia.com/cuda-toolkit)

---

## 📞 联系方式

如有问题，请通过以下方式联系：
- Email: [lshi@cs.ecnu.edu.cn](mailto:lshi@cs.ecnu.edu.cn)
- GitHub Issues: 在项目仓库中提 Issue

---

## 📄 许可证

本项目遵循 MIT 开源协议。

---

