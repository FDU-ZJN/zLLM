import gc
import torch


class MemoryOptimizer:

    def __init__(self, clear_interval=16, force_gc=True, cuda_empty_cache=True):
        self.clear_interval = int(clear_interval)
        self.force_gc = bool(force_gc)
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self._step = 0

    def before_infer(self):
        self._step += 1

    def after_infer(self):
        if self.clear_interval <= 0:
            return
        if self._step % self.clear_interval != 0:
            return

        if self.force_gc:
            gc.collect()
        if self.cuda_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def try_set_per_process_memory_fraction(self, fraction, device_index=0):
        if not torch.cuda.is_available():
            return False
        value = float(fraction)
        torch.cuda.set_per_process_memory_fraction(value, device=device_index)
        return True

    def memory_snapshot(self):
        if not torch.cuda.is_available():
            return {
                "cuda": False,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "max_reserved_gb": 0.0,
            }

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        max_reserved = torch.cuda.max_memory_reserved() / 1e9

        return {
            "cuda": True,
            "allocated_gb": round(allocated, 4),
            "reserved_gb": round(reserved, 4),
            "max_allocated_gb": round(max_allocated, 4),
            "max_reserved_gb": round(max_reserved, 4),
        }

    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
