import functools
from typing import Any, Callable, Iterable, List


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
