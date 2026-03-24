import time
from collections import OrderedDict
import math
import inspect


class PagedKVCacheManager:

    @staticmethod
    def normalize_key(key):
        if not isinstance(key, str):
            return key
        import re
        key = key.strip()
        key = re.sub(r'[\u3000\s]+', ' ', key)
        return key

    def __init__(
        self,
        max_entries=16,
        max_cache_tokens=32768,
        ttl_seconds=600,
        page_size_tokens=16,
        enable_prefix_sharing=True,
        max_pages=None,
    ):
        self.max_entries = int(max_entries)
        self.max_cache_tokens = int(max_cache_tokens)
        self.page_size_tokens = max(1, int(page_size_tokens))
        self.ttl_seconds = int(ttl_seconds)
        self.enable_prefix_sharing = bool(enable_prefix_sharing)
        derived_pages = int(math.ceil(float(self.max_cache_tokens) / float(self.page_size_tokens)))
        self.max_pages = max(1, int(max_pages)) if max_pages is not None else max(1, derived_pages)

        self._pages = {}
        self._page_lru = OrderedDict()
        self._token_sig_to_page = {}
        self._store = OrderedDict()
        self._total_tokens = 0

        self._next_page_id = 1
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._prefix_reuse_pages = 0

    def _now(self):
        return time.time()

    def _token_signature(self, tokens):
        if not self.enable_prefix_sharing:
            return None
        if tokens is None:
            return None
        return tuple(int(t) for t in tokens)

    def _touch_page(self, page_id):
        if page_id in self._page_lru:
            self._page_lru.move_to_end(page_id)
        else:
            self._page_lru[page_id] = 1

    def _is_expired(self, created_at):
        return (self._now() - float(created_at)) > self.ttl_seconds

    def _drop_page(self, page_id):
        page = self._pages.pop(page_id, None)
        self._page_lru.pop(page_id, None)
        if page is None:
            return
        token_sig = page.get("token_sig")
        if token_sig is not None and self._token_sig_to_page.get(token_sig) == page_id:
            self._token_sig_to_page.pop(token_sig, None)

    def _release_pages(self, page_ids):
        for page_id in page_ids:
            page = self._pages.get(page_id)
            if page is None:
                continue
            ref_count = int(page.get("ref_count", 0)) - 1
            if ref_count <= 0:
                self._drop_page(page_id)
            else:
                page["ref_count"] = ref_count

    def _evict_one(self):
        key, meta = self._store.popitem(last=False)
        self._total_tokens = max(0, self._total_tokens - int(meta.get("token_count", 0)))
        self._release_pages(meta.get("pages", []))
        self._evictions += 1
        return key

    def _evict_if_needed(self):
        self._evict_expired_keys()
        while len(self._store) > self.max_entries and self._store:
            self._evict_one()
        while self._total_tokens > self.max_cache_tokens and self._store:
            self._evict_one()
        while len(self._pages) > self.max_pages and self._store:
            self._evict_one()

    def _evict_expired_keys(self):
        if not self._store:
            return
        for key in list(self._store.keys()):
            meta = self._store.get(key)
            if meta is None:
                continue
            if self._is_expired(meta.get("created_at", 0.0)):
                removed = self._store.pop(key)
                self._total_tokens = max(0, self._total_tokens - int(removed.get("token_count", 0)))
                self._release_pages(removed.get("pages", []))
                self._evictions += 1

    def _iter_token_pages(self, token_ids):
        token_ids = self._normalize_token_ids(token_ids)
        if not token_ids:
            return []
        chunks = []
        for start in range(0, len(token_ids), self.page_size_tokens):
            chunks.append(token_ids[start : start + self.page_size_tokens])
        return chunks

    def _normalize_token_ids(self, token_ids):
        if token_ids is None:
            return []

        if hasattr(token_ids, "tolist"):
            values = token_ids.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = values[0]
            if isinstance(values, list):
                return [int(x) for x in values]

        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                return [int(x) for x in token_ids[0]]
            return [int(x) for x in token_ids]

        if isinstance(token_ids, tuple):
            return [int(x) for x in token_ids]

        return list(token_ids)

    def _extract_token_ids(self, payload):
        if not isinstance(payload, dict):
            return None
        input_ids = payload.get("input_ids")
        if input_ids is None:
            return None
        if hasattr(input_ids, "tolist"):
            values = input_ids.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list):
                return values[0]
            if isinstance(values, list):
                return values
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return input_ids[0]
            return input_ids
        return None

    def _extract_token_count(self, payload):
        if not isinstance(payload, dict):
            return None
        value = payload.get("input_len")
        if value is None:
            return None
        try:
            value = int(value)
        except Exception:
            return None
        return max(0, value)

    def _build_pages_from_token_count(self, token_count):
        pages = []
        remain = int(max(0, token_count))
        while remain > 0:
            chunk_size = min(self.page_size_tokens, remain)
            page_id = self._next_page_id
            self._next_page_id += 1
            self._pages[page_id] = {
                "token_sig": None,
                "token_count": int(chunk_size),
                "payload": None,
                "ref_count": 1,
                "created_at": self._now(),
            }
            self._touch_page(page_id)
            pages.append(page_id)
            remain -= chunk_size
        return pages

    def _iter_page_ranges(self, token_count):
        token_count = int(max(0, token_count))
        ranges = []
        start = 0
        while start < token_count:
            end = min(start + self.page_size_tokens, token_count)
            ranges.append((start, end))
            start = end
        return ranges

    def _build_page_payload(self, payload_builder, token_chunk, page_index, start, end):
        if not callable(payload_builder):
            return None
        try:
            sig = inspect.signature(payload_builder)
            positional = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(positional) >= 4:
                return payload_builder(token_chunk, page_index, start, end)
            if len(positional) >= 2:
                return payload_builder(token_chunk, page_index)
            return payload_builder(token_chunk)
        except Exception:
            return payload_builder(token_chunk)

    def put_sequence(self, sequence_key, token_ids=None, payload=None, payload_builder=None, token_count=None):
        sequence_key = self.normalize_key(str(sequence_key))

        if token_count is None:
            extracted_count = self._extract_token_count(payload)
            if extracted_count is not None:
                token_count = extracted_count

        if token_ids is None and self.enable_prefix_sharing:
            token_ids = self._extract_token_ids(payload)
        token_ids = self._normalize_token_ids(token_ids)

        if token_count is None:
            token_count = len(token_ids)
        token_count = int(token_count)
        if token_count < 0:
            token_count = 0

        old = self._store.pop(sequence_key, None)
        if old is not None:
            self._total_tokens = max(0, self._total_tokens - int(old.get("token_count", 0)))
            self._release_pages(old.get("pages", []))

        if not self.enable_prefix_sharing:
            pages = self._build_pages_from_token_count(token_count)
            page_ranges = self._iter_page_ranges(token_count)
            if callable(payload_builder):
                for idx, page_id in enumerate(pages):
                    page = self._pages.get(page_id)
                    if page is None:
                        continue
                    start, end = page_ranges[idx] if idx < len(page_ranges) else (0, 0)
                    token_chunk = []
                    if token_ids:
                        token_chunk = token_ids[start:end]
                    page["payload"] = self._build_page_payload(payload_builder, token_chunk, idx, start, end)
            self._store[sequence_key] = {
                "pages": pages,
                "token_count": token_count,
                "payload": payload,
                "created_at": self._now(),
            }
            self._store.move_to_end(sequence_key)
            self._total_tokens += token_count
            self._evict_if_needed()
            return

        if not token_ids and token_count > 0:
            token_ids = [0] * token_count

        pages = []
        page_payloads = []
        token_chunks = self._iter_token_pages(token_ids)
        page_ranges = self._iter_page_ranges(token_count)
        if callable(payload_builder):
            for idx, token_chunk in enumerate(token_chunks):
                start, end = page_ranges[idx] if idx < len(page_ranges) else (0, 0)
                page_payloads.append(self._build_page_payload(payload_builder, token_chunk, idx, start, end))

        for token_chunk in token_chunks:
            token_sig = self._token_signature(token_chunk)
            reused_page_id = None

            if token_sig is not None:
                reused_page_id = self._token_sig_to_page.get(token_sig)
                if reused_page_id is not None and reused_page_id in self._pages:
                    self._prefix_reuse_pages += 1

            if reused_page_id is not None:
                page = self._pages[reused_page_id]
                page["ref_count"] = int(page.get("ref_count", 0)) + 1
                pages.append(reused_page_id)
                self._touch_page(reused_page_id)
                continue

            page_id = self._next_page_id
            self._next_page_id += 1

            self._pages[page_id] = {
                "token_sig": token_sig,
                "token_count": len(token_chunk),
                "payload": None,
                "ref_count": 1,
                "created_at": self._now(),
            }
            if token_sig is not None:
                self._token_sig_to_page[token_sig] = page_id
            self._touch_page(page_id)
            pages.append(page_id)

        self._store[sequence_key] = {
            "pages": pages,
            "token_count": token_count,
            "payload": payload,
            "created_at": self._now(),
        }
        self._store.move_to_end(sequence_key)

        if page_payloads:
            meta = self._store.get(sequence_key)
            if meta is not None:
                for idx, page_id in enumerate(meta.get("pages", [])):
                    page = self._pages.get(page_id)
                    if page is None:
                        continue
                    if idx < len(page_payloads):
                        page["payload"] = page_payloads[idx]

        self._total_tokens += token_count
        self._evict_if_needed()

    def get_sequence(self, sequence_key, touch_stats=True):
        sequence_key = self.normalize_key(str(sequence_key))
        self._evict_expired_keys()
        meta = self._store.get(sequence_key)
        if meta is None:
            if touch_stats:
                self._misses += 1
            return None

        for page_id in meta.get("pages", []):
            if page_id not in self._pages:
                if touch_stats:
                    self._misses += 1
                return None
            self._touch_page(page_id)

        self._store.move_to_end(sequence_key)

        if touch_stats:
            self._hits += 1
        return meta.get("payload")

    def get_sequence_pages(self, sequence_key, touch_stats=True):
        sequence_key = self.normalize_key(str(sequence_key))
        self._evict_expired_keys()
        meta = self._store.get(sequence_key)
        if meta is None:
            if touch_stats:
                self._misses += 1
            return None

        pages = []
        for page_id in list(meta.get("pages", [])):
            page = self._pages.get(page_id)
            if page is None:
                if touch_stats:
                    self._misses += 1
                return None
            pages.append({
                "page_id": page_id,
                "token_count": int(page.get("token_count", 0)),
                "payload": page.get("payload"),
            })
            self._touch_page(page_id)
        self._store.move_to_end(sequence_key)
        if touch_stats:
            self._hits += 1
        return pages

    def delete_sequence(self, sequence_key):
        sequence_key = self.normalize_key(str(sequence_key))
        old = self._store.pop(sequence_key, None)
        if old is None:
            return False
        self._total_tokens = max(0, self._total_tokens - int(old.get("token_count", 0)))
        self._release_pages(old.get("pages", []))
        return True

    def clear(self):
        self._pages.clear()
        self._page_lru.clear()
        self._token_sig_to_page.clear()
        self._store.clear()
        self._total_tokens = 0

    def stats(self):
        total = self._hits + self._misses
        hit_rate = (float(self._hits) / float(total)) if total > 0 else 0.0
        return {
            "entries": len(self._store),
            "total_tokens": self._total_tokens,
            "pages": len(self._pages),
            "page_size_tokens": self.page_size_tokens,
            "max_pages": self.max_pages,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "prefix_reuse_pages": self._prefix_reuse_pages,
            "enable_prefix_sharing": self.enable_prefix_sharing,
            "max_entries": self.max_entries,
            "max_cache_tokens": self.max_cache_tokens,
        }