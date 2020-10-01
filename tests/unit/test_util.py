import numpy as np
import pytest
from trousse import util


@pytest.mark.parametrize(
    "iterable, expected_list",
    [
        ("str", ["str"]),
        (["hello", "hello2"], ["hello", "hello2"]),
        (("hello", "hello2"), ["hello", "hello2"]),
        (np.array([0, 1, 2]), [0, 1, 2]),
    ],
)
def test_tolist(iterable, expected_list):
    list_ = util.tolist(iterable)

    assert type(list_) == list
    assert list_ == expected_list


def but_it_raises_typeerror_with_none():
    with pytest.raises(TypeError) as err:
        util.tolist(None)

    assert isinstance(err.value, TypeError)
    assert str(err.value) == "'NoneType' object is not iterable"


@pytest.mark.parametrize(
    "obj, expected_result",
    (("nan", False), (dict(), False), (set(), False), ([1, 2, 3], True), ([], True)),
)
def test_is_sequence_and_not_str(obj, expected_result):
    result = util.is_sequence_and_not_str(obj)

    assert isinstance(result, bool)
    assert result == expected_result
