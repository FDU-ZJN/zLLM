import time
from enum import Enum


class DynamicBatchScheduler:

    def __init__(self, max_batch_size=16, max_wait_ms=20, max_prompt_tokens=16384):
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = int(max_wait_ms)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.queue = []

        self._total_dispatched = 0
        self._total_batches = 0
        self._total_queue_wait_ms = 0.0

    def _now_ms(self):
        return time.time() * 1000.0

    def add_request(self, request_id, prompt, prompt_tokens, gen_config=None):
        item = {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_tokens": int(prompt_tokens),
            "gen_config": gen_config or {},
            "enqueue_ms": self._now_ms(),
        }
        self.queue.append(item)

    def _should_dispatch(self):
        if len(self.queue) == 0:
            return False
        if len(self.queue) >= self.max_batch_size:
            return True

        oldest = self.queue[0]
        waited = self._now_ms() - oldest["enqueue_ms"]
        if waited >= self.max_wait_ms:
            return True
        return False

    def _build_batch(self):
        if not self.queue:
            return []

        self.queue.sort(key=lambda x: x["prompt_tokens"])

        batch = []
        token_budget = 0
        while self.queue and len(batch) < self.max_batch_size:
            candidate = self.queue[0]
            candidate_tokens = int(candidate["prompt_tokens"])
            if token_budget + candidate_tokens > self.max_prompt_tokens and len(batch) > 0:
                break
            token_budget += candidate_tokens
            batch.append(self.queue.pop(0))

        if not batch and self.queue:
            batch.append(self.queue.pop(0))

        return batch

    def poll_batch(self):
        if not self._should_dispatch():
            return []

        batch = self._build_batch()
        now = self._now_ms()

        for item in batch:
            self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])

        self._total_dispatched += len(batch)
        self._total_batches += 1
        return batch

    def flush_all(self):
        out = []
        while self.queue:
            batch = self._build_batch()
            now = self._now_ms()
            for item in batch:
                self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])
            self._total_dispatched += len(batch)
            self._total_batches += 1
            out.append(batch)
        return out

    def stats(self):
        avg_batch_size = (float(self._total_dispatched) / float(self._total_batches)) if self._total_batches > 0 else 0.0
        avg_wait = (float(self._total_queue_wait_ms) / float(self._total_dispatched)) if self._total_dispatched > 0 else 0.0
        return {
            "queued": len(self.queue),
            "total_dispatched": self._total_dispatched,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch_size, 4),
            "avg_queue_wait_ms": round(avg_wait, 4),
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "max_prompt_tokens": self.max_prompt_tokens,
        }

class FIFODynamicBatchScheduler:

    def __init__(self, max_batch_size=16, max_wait_ms=20, max_prompt_tokens=16384):
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = int(max_wait_ms)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.queue = []

        self._total_dispatched = 0
        self._total_batches = 0
        self._total_queue_wait_ms = 0.0

    def _now_ms(self):
        return time.time() * 1000.0

    def add_request(self, request_id, prompt, prompt_tokens, gen_config=None):
        item = {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_tokens": int(prompt_tokens),
            "gen_config": gen_config or {},
            "enqueue_ms": self._now_ms(),
        }
        self.queue.append(item)

    def _should_dispatch(self):
        if len(self.queue) == 0:
            return False
        if len(self.queue) >= self.max_batch_size:
            return True
        oldest = self.queue[0]
        waited = self._now_ms() - oldest["enqueue_ms"]
        return waited >= self.max_wait_ms

    def _build_batch(self):
        if not self.queue:
            return []

        batch = []
        token_budget = 0
        # FIFO：从队头依次 pop，不再 sort
        while self.queue and len(batch) < self.max_batch_size:
            candidate = self.queue[0]
            candidate_tokens = int(candidate["prompt_tokens"])
            if token_budget + candidate_tokens > self.max_prompt_tokens and len(batch) > 0:
                break
            token_budget += candidate_tokens
            batch.append(self.queue.pop(0))

        if not batch and self.queue:
            batch.append(self.queue.pop(0))

        return batch

    def poll_batch(self):
        if not self._should_dispatch():
            return []

        batch = self._build_batch()
        now = self._now_ms()

        for item in batch:
            self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])

        self._total_dispatched += len(batch)
        self._total_batches += 1
        return batch

    def flush_all(self):
        out = []
        while self.queue:
            batch = self._build_batch()
            now = self._now_ms()
            for item in batch:
                self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])
            self._total_dispatched += len(batch)
            self._total_batches += 1
            out.append(batch)
        return out

    def stats(self):
        avg_batch_size = (
            float(self._total_dispatched) / float(self._total_batches)
            if self._total_batches > 0
            else 0.0
        )
        avg_wait = (
            float(self._total_queue_wait_ms) / float(self._total_dispatched)
            if self._total_dispatched > 0
            else 0.0
        )
        return {
            "queued": len(self.queue),
            "total_dispatched": self._total_dispatched,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch_size, 4),
            "avg_queue_wait_ms": round(avg_wait, 4),
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "max_prompt_tokens": self.max_prompt_tokens,
        }



class AgingDynamicBatchScheduler:
    """带老化防饥饿机制的动态 Batch 调度器。

    在按 prompt_tokens 排序的基础上引入等待时间衰减，排序 key 为：
        effective_score = prompt_tokens - aging_rate * waited_ms
    等待越久 effective_score 越低（优先级越高），从而避免长请求被持续跳过。
    同时设置 max_starve_ms 作为硬保底：超过该时限的请求无条件优先调度。
    """

    def __init__(
        self,
        max_batch_size=16,
        max_wait_ms=20,
        max_prompt_tokens=16384,
        aging_rate=0.1,
        max_starve_ms=500,
    ):
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = int(max_wait_ms)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.aging_rate = float(aging_rate)
        self.max_starve_ms = float(max_starve_ms)
        self.queue = []

        self._total_dispatched = 0
        self._total_batches = 0
        self._total_queue_wait_ms = 0.0
        self._total_starved_promoted = 0

    def _now_ms(self):
        return time.time() * 1000.0

    def add_request(self, request_id, prompt, prompt_tokens, gen_config=None):
        item = {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_tokens": int(prompt_tokens),
            "gen_config": gen_config or {},
            "enqueue_ms": self._now_ms(),
        }
        self.queue.append(item)

    def _should_dispatch(self):
        if len(self.queue) == 0:
            return False
        if len(self.queue) >= self.max_batch_size:
            return True
        now = self._now_ms()
        oldest_wait = max(now - item["enqueue_ms"] for item in self.queue)
        return oldest_wait >= self.max_wait_ms

    def _build_batch(self):
        if not self.queue:
            return []

        now = self._now_ms()

        starved = []
        normal = []
        for item in self.queue:
            waited = now - item["enqueue_ms"]
            if waited >= self.max_starve_ms:
                starved.append(item)
            else:
                normal.append(item)

        self._total_starved_promoted += len(starved)

        starved.sort(key=lambda x: x["enqueue_ms"])
        normal.sort(
            key=lambda x: x["prompt_tokens"] - self.aging_rate * (now - x["enqueue_ms"])
        )

        candidates = starved + normal

        batch = []
        token_budget = 0
        picked_set = set()

        for item in candidates:
            if len(batch) >= self.max_batch_size:
                break
            candidate_tokens = int(item["prompt_tokens"])
            if token_budget + candidate_tokens > self.max_prompt_tokens and len(batch) > 0:
                break
            token_budget += candidate_tokens
            batch.append(item)
            picked_set.add(id(item))

        if not batch and self.queue:
            item = self.queue[0]
            batch.append(item)
            picked_set.add(id(item))

        self.queue = [item for item in self.queue if id(item) not in picked_set]

        return batch

    def poll_batch(self):
        if not self._should_dispatch():
            return []

        batch = self._build_batch()
        now = self._now_ms()

        for item in batch:
            self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])

        self._total_dispatched += len(batch)
        self._total_batches += 1
        return batch

    def flush_all(self):
        out = []
        while self.queue:
            batch = self._build_batch()
            now = self._now_ms()
            for item in batch:
                self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])
            self._total_dispatched += len(batch)
            self._total_batches += 1
            out.append(batch)
        return out

    def stats(self):
        avg_batch_size = (
            float(self._total_dispatched) / float(self._total_batches)
            if self._total_batches > 0
            else 0.0
        )
        avg_wait = (
            float(self._total_queue_wait_ms) / float(self._total_dispatched)
            if self._total_dispatched > 0
            else 0.0
        )
        return {
            "queued": len(self.queue),
            "total_dispatched": self._total_dispatched,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch_size, 4),
            "avg_queue_wait_ms": round(avg_wait, 4),
            "starved_promoted": self._total_starved_promoted,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "max_prompt_tokens": self.max_prompt_tokens,
            "aging_rate": self.aging_rate,
            "max_starve_ms": self.max_starve_ms,
        }


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"


class ContinuousBatchScheduler:
    """Continuous Batching (iteration-level) 调度器。

    与 Dynamic Batch 在批次粒度调度不同，Continuous Batch 在每个 decode step 级别调度：
    - 每个 decode step 结束后，已完成的请求立即被驱逐
    - 等待队列中的新请求立即填入空闲 slot，无需等待整个 batch 结束
    - 消除了 "batch bubble"（短请求等待长请求完成的空转），最大化 GPU 利用率

    典型用法：
        scheduler = ContinuousBatchScheduler(max_batch_size=8)
        for idx, prompt in enumerate(prompts):
            scheduler.add_request(idx, prompt, token_count)

        while scheduler.has_unfinished():
            new_reqs, active_reqs = scheduler.schedule()
            # prefill new_reqs, decode active_reqs (逐 token)
            scheduler.finish_requests([rid for rid in done_ids])
    """

    def __init__(self, max_batch_size=8, max_new_tokens=256):
        self.max_batch_size = int(max_batch_size)
        self.max_new_tokens = int(max_new_tokens)

        self.waiting = []
        self.running = {}

        self._total_finished = 0
        self._total_decode_steps = 0
        self._total_queue_wait_ms = 0.0
        self._total_tokens_generated = 0
        self._sum_active_per_step = 0

    def _now_ms(self):
        return time.time() * 1000.0

    def add_request(self, request_id, prompt, prompt_tokens, gen_config=None):
        item = {
            "request_id": request_id,
            "prompt": prompt,
            "prompt_tokens": int(prompt_tokens),
            "gen_config": gen_config or {},
            "enqueue_ms": self._now_ms(),
            "state": RequestState.WAITING,
        }
        self.waiting.append(item)

    def schedule(self):
        """每个 iteration 调用一次。

        Returns:
            (new_requests, active_requests)
            - new_requests: 本轮新从 waiting 进入 running 的请求（需要 prefill）
            - active_requests: 所有处于 running 的请求（包含新请求，prefill 之后它们也需要 decode）
        """
        free_slots = self.max_batch_size - len(self.running)
        new_requests = []

        while self.waiting and free_slots > 0:
            item = self.waiting.pop(0)
            now = self._now_ms()
            self._total_queue_wait_ms += max(0.0, now - item["enqueue_ms"])
            item["state"] = RequestState.PREFILL
            item["admit_ms"] = now
            self.running[item["request_id"]] = item
            new_requests.append(item)
            free_slots -= 1

        active_requests = list(self.running.values())

        self._total_decode_steps += 1
        self._sum_active_per_step += len(active_requests)

        return new_requests, active_requests

    def finish_requests(self, finished_ids):
        """将完成的请求从 running 中移除。"""
        for rid in finished_ids:
            req = self.running.pop(rid, None)
            if req is not None:
                req["state"] = RequestState.FINISHED
                self._total_finished += 1

    def mark_decoding(self, request_id):
        """将 prefill 完成的请求标记为 DECODING 状态。"""
        if request_id in self.running:
            self.running[request_id]["state"] = RequestState.DECODING

    def has_unfinished(self):
        return len(self.waiting) > 0 or len(self.running) > 0

    def num_waiting(self):
        return len(self.waiting)

    def num_running(self):
        return len(self.running)

    def stats(self):
        avg_active = (
            float(self._sum_active_per_step) / float(self._total_decode_steps)
            if self._total_decode_steps > 0
            else 0.0
        )
        avg_wait = (
            float(self._total_queue_wait_ms) / float(self._total_finished)
            if self._total_finished > 0
            else 0.0
        )
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "total_finished": self._total_finished,
            "total_decode_steps": self._total_decode_steps,
            "avg_active_per_step": round(avg_active, 4),
            "avg_queue_wait_ms": round(avg_wait, 4),
            "max_batch_size": self.max_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
