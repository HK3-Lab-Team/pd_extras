import functools
from typing import Any, Callable


def lazy_property(f: Callable[..., Any]):
    return property(functools.lru_cache()(f))
