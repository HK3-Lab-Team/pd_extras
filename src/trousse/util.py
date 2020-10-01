import functools
from typing import Any, Callable, Iterable, List, Sequence


def lazy_property(f: Callable[..., Any]):
    return property(functools.lru_cache()(f))


def tolist(iterable: Iterable) -> List[Any]:
    """Convert any iterable to a list. A string is converted to a single element list.

    Parameters
    ----------
    iterable : Iterable[Any]
        The object to convert

    Returns
    -------
    List[Any]
        List after conversion
    """
    if isinstance(iterable, str):
        iterable = [iterable]
    elif not isinstance(iterable, list):
        iterable = list(iterable)
    return iterable


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
