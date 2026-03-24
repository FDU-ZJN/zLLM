import time
from collections import OrderedDict


class KVCacheManager:

    @staticmethod
    def normalize_key(key):
        if not isinstance(key, str):
            return key
        import re
        key = key.strip()
        key = re.sub(r'[\u3000\s]+', ' ', key)
        return key

    def __init__(self, max_entries=16, max_cache_tokens=32768, ttl_seconds=600):
        self.max_entries = int(max_entries)
        self.max_cache_tokens = int(max_cache_tokens)
        self.ttl_seconds = int(ttl_seconds)

        self._store = OrderedDict()
        self._total_tokens = 0

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _now(self):
        return time.time()

    def _is_expired(self, item):
        return (self._now() - item["created_at"]) > self.ttl_seconds

    def _evict_one(self):
        key, item = self._store.popitem(last=False)
        self._total_tokens = max(0, self._total_tokens - int(item.get("token_count", 0)))
        self._evictions += 1
        return key

    def _evict_expired(self):
        if not self._store:
            return
        keys = list(self._store.keys())
        for key in keys:
            item = self._store.get(key)
            if item is None:
                continue
            if self._is_expired(item):
                removed = self._store.pop(key)
                self._total_tokens = max(0, self._total_tokens - int(removed.get("token_count", 0)))
                self._evictions += 1

    def _evict_to_budget(self):
        while len(self._store) > self.max_entries:
            self._evict_one()
        while self._total_tokens > self.max_cache_tokens and len(self._store) > 0:
            self._evict_one()

    def get(self, key):
        key = self.normalize_key(key)
        self._evict_expired()
        item = self._store.get(key)
        if item is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return item["payload"]

    def put(self, key, payload, token_count):
        key = self.normalize_key(key)
        token_count = int(token_count)
        if token_count < 0:
            token_count = 0

        old = self._store.pop(key, None)
        if old is not None:
            self._total_tokens = max(0, self._total_tokens - int(old.get("token_count", 0)))

        self._store[key] = {
            "payload": payload,
            "token_count": token_count,
            "created_at": self._now(),
        }
        self._store.move_to_end(key)
        self._total_tokens += token_count

        self._evict_expired()
        self._evict_to_budget()

    def delete(self, key):
        key = self.normalize_key(key)
        old = self._store.pop(key, None)
        if old is not None:
            self._total_tokens = max(0, self._total_tokens - int(old.get("token_count", 0)))
            return True
        return False

    def clear(self):
        self._store.clear()
        self._total_tokens = 0

    def stats(self):
        total = self._hits + self._misses
        hit_rate = (float(self._hits) / float(total)) if total > 0 else 0.0
        return {
            "entries": len(self._store),
            "total_tokens": self._total_tokens,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "max_entries": self.max_entries,
            "max_cache_tokens": self.max_cache_tokens,
            "ttl_seconds": self.ttl_seconds,
        }