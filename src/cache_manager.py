"""
cache_manager.py
================
LangChain caching: save model answers so an identical request does not need
a second (slow, paid) API call.

Two cache types are demonstrated:

  InMemoryCache  - stored in RAM. Fastest, but wiped when the app restarts.
  SQLiteCache    - stored in a .db file on disk. Slightly slower, but survives
                   restarts and can be reused across sessions.

How it works: we register ONE cache with `set_llm_cache(...)`. From then on
LangChain automatically checks it before every model call.
"""

import time
from typing import Iterator

from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from src import config

try:  # InMemoryCache lives in langchain-core in newer releases
    from langchain_community.cache import InMemoryCache, SQLiteCache
except ImportError:  # pragma: no cover
    from langchain_community.cache import SQLiteCache
    from langchain_core.caches import InMemoryCache


class CountingCache(BaseCache):
    """
    A thin wrapper around a real cache that counts hits and misses.

    The counters let the UI show "⚡ answered from cache" and let the tests
    prove the cache is really used. It forwards everything to the inner cache.
    """

    def __init__(self, inner: BaseCache, name: str):
        self.inner = inner
        self.name = name
        self.hits = 0
        self.misses = 0

    def lookup(self, prompt, llm_string):
        result = self.inner.lookup(prompt, llm_string)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def update(self, prompt, llm_string, return_val):
        self.inner.update(prompt, llm_string, return_val)

    def clear(self, **kwargs):
        self.inner.clear(**kwargs)


# One cache object per mode, created once and then re-used. (Creating a new
# InMemoryCache on every Streamlit rerun would throw the saved answers away.)
_CACHES: dict = {}


def _build_cache(mode: str) -> CountingCache:
    if mode == config.CACHE_MEMORY:
        return CountingCache(InMemoryCache(), mode)
    if mode == config.CACHE_SQLITE:
        config.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return CountingCache(SQLiteCache(database_path=str(config.SQLITE_PATH)), mode)
    raise ValueError(f"Unknown cache mode: {mode}")


def configure_cache(mode: str):
    """
    Switch caching on or off for the whole app.
    Returns the active CountingCache (or None when caching is Off).
    """
    if mode == config.CACHE_OFF or mode not in config.CACHE_MODES:
        set_llm_cache(None)
        return None
    if mode not in _CACHES:
        _CACHES[mode] = _build_cache(mode)
    set_llm_cache(_CACHES[mode])  # <- the one line that switches caching on
    return _CACHES[mode]


def clear_cache(mode: str) -> None:
    """Delete every saved answer of the given cache type."""
    if mode not in (config.CACHE_MEMORY, config.CACHE_SQLITE):
        return
    if mode not in _CACHES:
        _CACHES[mode] = _build_cache(mode)
    _CACHES[mode].clear()
    _CACHES[mode].hits = 0
    _CACHES[mode].misses = 0


def get_stats() -> dict:
    """Current hit / miss counters of the active cache (zeros when Off)."""
    cache = get_llm_cache()
    if isinstance(cache, CountingCache):
        return {"mode": cache.name, "hits": cache.hits, "misses": cache.misses}
    return {"mode": config.CACHE_OFF, "hits": 0, "misses": 0}


def cached_stream(prompt_key: str, llm_string: str, producer) -> Iterator[str]:
    """
    Stream text with caching.

    LangChain's `.stream()` does not read the global cache by itself, so we
    use the same cache object manually:
      * cache HIT  -> replay the saved text instantly (still as a stream)
      * cache MISS -> stream live from the model, then save the full text

    `producer` is a function that returns the live stream (an iterator of str).
    """
    cache = get_llm_cache()

    if cache is not None:
        hit = cache.lookup(prompt_key, llm_string)
        if hit:
            text = hit[0].text
            for i in range(0, len(text), 24):
                yield text[i : i + 24]
                time.sleep(0.001)  # tiny pause keeps the typing effect smooth
            return

    parts = []
    for chunk in producer():
        parts.append(chunk)
        yield chunk

    if cache is not None and parts:
        cache.update(prompt_key, llm_string, [Generation(text="".join(parts))])


CACHE_COMPARISON = [
    ("Where is it stored?", "RAM (memory)", "A file on disk (.db)"),
    ("Speed", "Fastest", "Fast, slightly slower"),
    ("Survives an app restart?", "No", "Yes"),
    ("Best for", "One running session", "Re-using answers across sessions"),
]
