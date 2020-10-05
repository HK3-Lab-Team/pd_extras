import pytest
from trousse import util


@pytest.mark.parametrize(
    "obj, expected_result",
    (("nan", False), (dict(), False), (set(), False), ([1, 2, 3], True), ([], True)),
)
def test_is_sequence_and_not_str(obj, expected_result):
    result = util.is_sequence_and_not_str(obj)

    assert isinstance(result, bool)
    assert result == expected_result
