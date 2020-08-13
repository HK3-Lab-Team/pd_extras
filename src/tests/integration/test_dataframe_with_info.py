from datetime import date

import pandas as pd
import pytest

from ...pd_extras.dataframe_with_info import (
    DataFrameWithInfo,
    _split_columns_by_type_parallel,
)
from ..unitutil import DataFrameMock, function_mock


class Describe_DataFrameWithInfo:
    @pytest.mark.parametrize(
        "nan_ratio, n_columns, expected_many_nan_columns",
        [
            (0.8, 2, {"nan_0", "nan_1"}),
            (0.8, 1, {"nan_0"}),
            (0.8, 0, set()),
            (
                0.0,
                2,
                {
                    "nan_0",
                    "nan_1",
                    "not_nan_0",
                    "not_nan_1",
                    "not_nan_2",
                    "not_nan_3",
                    "not_nan_4",
                },
            ),
            (1.0, 2, {"nan_0", "nan_1"}),
        ],
    )
    def test_many_nan_columns(
        self, request, nan_ratio, n_columns, expected_many_nan_columns
    ):
        df = DataFrameMock.df_many_nans(nan_ratio, n_columns)
        df_info = DataFrameWithInfo(
            df_object=df, nan_percentage_threshold=nan_ratio - 0.01
        )

        many_nan_columns = df_info.many_nan_columns

        assert len(many_nan_columns) == len(expected_many_nan_columns)
        assert isinstance(many_nan_columns, set)
        assert many_nan_columns == expected_many_nan_columns

    @pytest.mark.parametrize(
        "n_columns, expected_same_value_columns",
        [(2, {"same_0", "same_1"}), (1, {"same_0"}), (0, set()),],
    )
    def test_same_value_columns(self, request, n_columns, expected_same_value_columns):
        df = DataFrameMock.df_same_value(n_columns)
        df_info = DataFrameWithInfo(df_object=df)

        same_value_columns = df_info.same_value_cols

        assert len(same_value_columns) == len(expected_same_value_columns)
        assert isinstance(same_value_columns, set)
        assert same_value_columns == expected_same_value_columns


def mock_responses(responses, default_response=None):
    """
    Return function that will return different values based on the input argument.

    Based on the dictionary (map) that the user inputs through ``responses``
    argument, the function ``corresponding_response`` (and returned by this function)
    will return the value related to the second argument that is input when
    ``corresponding_response`` is called

    Parameters
    ----------
    responses: Dict
        Map of the values that connect the second argument of the returned function
        ``corresponding_response`` to the value it must return
    default_response: Any
        Default value that will be returned by the returned function
        ``corresponding_response`` when the input argument is not among the keys
        of ``responses``

    Example
    -------
    >>> my_mock.foo.side_effect = mock_responses(
    >>>   {
    >>>      'x': 42,
    >>>      'y': [1,2,3]
    >>>    })
    >>> my_mock.foo('x')
    42
    >>> my_mock.foo('y')
    [1,2,3]
    >>> my_mock.foo('unknown')  # ``default_response`` is returned
    None
    """

    def corresponding_response(df, input):
        return responses[input] if input in responses else default_response

    return corresponding_response


def test_split_columns_by_type_parallel_unittest(request):
    # Input Arguments
    df_by_type = DataFrameMock.df_multi_type()
    col_list = df_by_type.columns
    # Internal Functions
    _find_single_column_type = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_single_column_type"
    )
    _find_single_column_type.return_value = mock_responses(
        {
            pd.Series([True, False, True, True, False]): {
                "col_name": "bool_col_0",
                "col_type": "bool_col",
            },
            pd.Series(["value_0", "value_1", "value_2", "value_3", "value_4"]): {
                "col_name": "string_col_0",
                "col_type": "string_col",
            },
            pd.Series(
                ["category_1", "category_1", "category_0", "category_1", "category_0"]
            ): {"col_name": "categorical_col_0", "col_type": "string_col"},
            pd.Series([0.05 * i for i in range(5)]): {
                "col_name": "numerical_col_0",
                "col_type": "numerical_col",
            },
            pd.Series(
                [
                    pd.Interval(0, 1),
                    pd.Interval(1, 5),
                    pd.Interval(2, 5),
                    pd.Interval(1, 4),
                    None,
                ]
            ): {"col_name": "interval_col_0", "col_type": "numerical_col"},
            pd.Series([date.today() for i in range(5)]): {
                "col_name": "datetime_col_0",
                "col_type": "other_col",
            },
            pd.Series([1, 2, 3, 4, "value_0"]): {
                "col_name": "mixed_type_col_0",
                "col_type": "mixed_type_col",
            },
        }
    )
    # RETURN MIXED COLUMNS
    _find_columns_by_type_mixed = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_columns_by_type"
    )
    _find_columns_by_type_mixed.return_value = {"mixed_type_col_0"}
    # RETURN NUMERICAL COLUMNS
    _find_columns_by_type_numerical = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_columns_by_type"
    )
    _find_columns_by_type_numerical.return_value = {"numerical_col_0", "interval_col_0"}
    # RETURN STRING COLUMNS
    _find_columns_by_type_string = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_columns_by_type"
    )
    _find_columns_by_type_string.return_value = {"string_col_0", "categorical_col_0"}
    # RETURN OTHER COLUMNS
    _find_columns_by_type_other = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_columns_by_type"
    )
    _find_columns_by_type_other.return_value = {"datetime_col_0"}
    # RETURN BOOL COLUMNS
    _find_columns_by_type_bool = function_mock(
        request, "src.pd_extras.dataframe_with_info._find_columns_by_type"
    )
    _find_columns_by_type_bool.return_value = {"bool_col_0"}

    cols_by_type_tuple = _split_columns_by_type_parallel(df_by_type, col_list)

    # TODO: Complete this part with appropriate DataFrame (at the moment it is created
    #   during parallel process so the sample order is not fixed, and it can't
    #   be predicted)
    _find_columns_by_type_mixed.assert_called_once()  # _with(..., "mixed_type_col")
    _find_columns_by_type_numerical.assert_called_once()  # _with(..., "numerical_col")
    _find_columns_by_type_string.assert_called_once()  # _with(..., "string_col")
    _find_columns_by_type_other.assert_called_once()  # _with(..., "other_col")
    _find_columns_by_type_bool.assert_called_once()  # _with(..., "bool_col")

    assert cols_by_type_tuple == (
        {"mixed_type_col_0"},
        {"numerical_col_0", "interval_col_0"},
        {"string_col_0", "categorical_col_0"},
        {"datetime_col_0"},
        {"bool_col_0"},
    )
