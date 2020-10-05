import functools
from typing import Any, Callable, Sequence


def lazy_property(f: Callable[..., Any]):
    return property(functools.lru_cache()(f))


def is_sequence_and_not_str(obj: Any) -> bool:
    """Return True if ``obj`` is a sequence object but not a string.

    Parameters
    ----------
    obj : Any
        Object to check

    Returns
    -------
    bool
        True if ``obj`` is a sequence object but not a string
    """
    return not isinstance(obj, str) and isinstance(obj, Sequence)
