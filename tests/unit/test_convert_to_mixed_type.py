import copy
import datetime

import numpy as np
import pandas as pd
import pytest
from trousse.convert_to_mixed_type import _ConvertDfToMixedType, _StrColumnToConvert

from ..unitutil import ANY, initializer_mock


class Describe_StrColumnToConvert:
    @pytest.mark.parametrize(
        "original_values, dtype, converted_values, expected_coerce_dtype_conv",
        [
            (
                pd.Series([], dtype="object"),
                "object",
                pd.Series([], dtype="object"),
                True,
            ),
            (
                pd.Series([1, 2, 3, 4], dtype="float"),
                "int",
                pd.Series([pd.NA] * 4, dtype="object"),
                True,
            ),
            (
                pd.Series(["1", "2", "3", "4"], dtype="object"),
                "float",
                pd.Series([pd.NA] * 4, dtype="object"),
                True,
            ),
            (  # Non-forced dtype
                pd.Series(["1", "2", "3", "4"], dtype="object"),
                "float",
                pd.Series([pd.NA] * 4, dtype="object"),
                True,
            ),
        ],
    )
    def it_constructs_from_args(
        self, dtype, original_values, converted_values, expected_coerce_dtype_conv
    ):
        mixed_col = _StrColumnToConvert(values=original_values, dtype=dtype)

        assert isinstance(mixed_col, _StrColumnToConvert)
        assert mixed_col._dtype == dtype
        assert mixed_col._coerce_dtype_conversion == expected_coerce_dtype_conv
        pd.testing.assert_series_equal(mixed_col._original_values, original_values)
        pd.testing.assert_series_equal(mixed_col._converted_values, converted_values)

    @pytest.mark.parametrize(
        "original_values",
        [
            pd.Series([], dtype="object"),
            pd.Series([1, 2, 3, 4], dtype="float"),
            pd.Series(["1", "2", "3", "4"], dtype="object"),
        ],
    )
    def it_knows_its_original_values(self, original_values):
        mixed_col = _StrColumnToConvert(values=original_values)

        original_values_ = mixed_col.original_values

        assert original_values_ is not original_values
        assert type(original_values_) == pd.Series
        pd.testing.assert_series_equal(original_values_, original_values)

    @pytest.mark.parametrize(
        "dtype, expected_dtype",
        [
            (None, "object"),
            ("float", "float"),
            ("int", pd.Int32Dtype()),
        ],
    )
    def it_knows_its_dtype(self, dtype, expected_dtype):
        mixed_col = _StrColumnToConvert(values=pd.Series(), dtype=dtype)

        dtype_ = mixed_col.dtype

        # The instance type cannot be checked because pandas ExtensionDtype
        # implies forward declaration and this an error would be raised:
        # "Forward references cannot be used with isinstance()"
        assert dtype_ == expected_dtype

    @pytest.mark.parametrize(
        "original_values, expected_converted_values",
        [
            (pd.Series([]), pd.Series([], dtype="object")),
            (
                pd.Series([1, 2, 3, 4], dtype="float"),
                pd.Series([pd.NA] * 4, dtype="object"),
            ),
        ],
    )
    def it_knows_its_converted_values(self, original_values, expected_converted_values):
        mixed_col = _StrColumnToConvert(values=original_values)

        converted_values_ = mixed_col.converted_values

        assert type(converted_values_) == pd.Series
        pd.testing.assert_series_equal(converted_values_, expected_converted_values)

    @pytest.mark.parametrize(
        "original_values, converted_values, dtype, expected_orig_with_conv_values",
        [
            (
                pd.Series(["1", "2", "str0", 3, 4, None], dtype="object"),
                pd.Series([1, 2, pd.NA, 3, 4, pd.NA], dtype="object"),
                "Int32",
                pd.Series([1, 2, "str0", 3, 4, None], dtype="object"),
            ),
            (
                pd.Series(["1", "2", "str0", 3, 4, None], dtype="object"),
                pd.Series([1, 2, pd.NA, 3, 4, pd.NA], dtype="object"),
                "int",
                pd.Series([1, 2, "str0", 3, 4, None], dtype="object"),
            ),
            (
                pd.Series(
                    ["1", "2", "str0", "True", "3/04/2020", None], dtype="object"
                ),
                pd.Series(
                    [1, 2, pd.NA, True, datetime.date(2000, 8, 1), pd.NA],
                    dtype="object",
                ),
                "object",
                pd.Series(
                    [1, 2, "str0", True, datetime.date(2000, 8, 1), None],
                    dtype="object",
                ),
            ),
        ],
    )
    def it_knows_its_original_with_converted_values(
        self, original_values, converted_values, dtype, expected_orig_with_conv_values
    ):
        # Create original values
        mixed_col = _StrColumnToConvert(values=original_values, dtype=dtype)
        # Modify converted values
        mixed_col._converted_values = converted_values

        original_with_converted_values_ = mixed_col.original_with_converted_values

        assert isinstance(original_with_converted_values_, pd.Series)
        pd.testing.assert_series_equal(
            original_with_converted_values_, expected_orig_with_conv_values
        )

    @pytest.mark.parametrize(
        "original_values, converted_values, new_converted, dtype, expected_dtype",
        [
            (  # NOT modifying dtype attribute
                pd.Series(["1", "2", True, False, 4, None], dtype="object"),
                pd.Series([pd.NA, pd.NA, True, False, pd.NA, pd.NA], dtype="object"),
                pd.Series([1, 2, 1, 0, 4, np.nan], dtype="float"),
                None,
                None,
            ),
            (  # Modifying dtype attribute, no dtype set
                pd.Series(["1", "2", "3", 4, 5, None], dtype="object"),
                pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="object"),
                pd.Series([1, 2, 3, 4, 5, np.nan], dtype="float"),
                None,
                "float",
            ),
            (  # Modifying dtype attribute, previous dtype set
                pd.Series(["1", "2", "3", 4, 5, None], dtype="object"),
                pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="object"),
                pd.Series([1, 2, 3, 4, 5, np.nan], dtype="float"),
                "int",
                "int",
            ),
        ],
    )
    def it_can_update_dtype(
        self, original_values, converted_values, new_converted, dtype, expected_dtype
    ):
        # Create original values
        mixed_col = _StrColumnToConvert(values=original_values, dtype=dtype)
        # Set previously converted values
        mixed_col._converted_values = converted_values

        dtype_ = mixed_col._updated_dtype(new_converted)

        assert dtype_ == expected_dtype

    @pytest.mark.parametrize(
        "original_values, converted_values, new_converted, dtype, expected_converted_values, expected_dtype",
        [
            (  # With previously converted values but avoiding overwriting
                pd.Series(["1", "2", True, False, 4, None], dtype="object"),
                pd.Series([pd.NA, pd.NA, True, False, pd.NA, pd.NA], dtype="object"),
                pd.Series([1, 2, 1, 0, 4, np.nan], dtype="float"),
                None,
                pd.Series([1, 2, True, False, 4, pd.NA], dtype="object"),
                "object",
            ),
            (  # With NO previously converted values
                pd.Series(["1", "2", "True", 4, 5, None], dtype="object"),
                pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="object"),
                pd.Series([1, 2, np.nan, 4, 5, np.nan], dtype="float"),
                None,
                pd.Series([1, 2, pd.NA, 4, 5, pd.NA], dtype="object"),
                "object",
            ),
        ],
    )
    def it_can_add_converted_values(
        self,
        original_values,
        converted_values,
        new_converted,
        dtype,
        expected_converted_values,
        expected_dtype,
    ):
        # Create original values
        mixed_col = _StrColumnToConvert(values=original_values, dtype=dtype)
        # Set previously converted values
        mixed_col._converted_values = converted_values

        mixed_col_ = mixed_col.add_converted_values(new_converted)

        assert mixed_col_.dtype == expected_dtype
        assert isinstance(mixed_col_.converted_values, pd.Series)
        pd.testing.assert_series_equal(
            mixed_col_.converted_values, expected_converted_values
        )

    def it_can_safe_convert_to_dtype(self):
        pass

    def it_can_deepcopy_itself(self):
        mixed_col = _StrColumnToConvert(
            pd.Series(["1", "2", True, False], dtype="object"), "float"
        )
        mixed_col._converted_values = pd.Series(
            [pd.NA, pd.NA, True, False, pd.NA, pd.NA], dtype="object"
        )

        copied_ = copy.deepcopy(mixed_col)

        assert copied_._original_values is not mixed_col._original_values
        pd.testing.assert_series_equal(
            copied_._original_values, mixed_col._original_values
        )
        assert copied_._converted_values is not mixed_col._converted_values
        pd.testing.assert_series_equal(
            copied_._converted_values, mixed_col._converted_values
        )
        # The builtin types share the memory everytime they contain
        # the same value ('float' is 'float')
        assert copied_._dtype == mixed_col._dtype
        assert copied_._coerce_dtype_conversion == mixed_col._coerce_dtype_conversion


class Describe_ConvertDfToMixedType:
    def it_constructs(self, request):
        _init_ = initializer_mock(request, _ConvertDfToMixedType)

        df_mixed_type_converter = _ConvertDfToMixedType(
            column="str_column", derived_column="mixed_column"
        )

        _init_.assert_called_once_with(
            ANY, column="str_column", derived_column="mixed_column"
        )
        assert isinstance(df_mixed_type_converter, _ConvertDfToMixedType)

    def it_constructs_from_args(self):
        df_mixed_type_converter = _ConvertDfToMixedType(
            column="str_column", derived_column="mixed_column"
        )

        assert isinstance(df_mixed_type_converter, _ConvertDfToMixedType)
        assert df_mixed_type_converter.column == "str_column"
        assert df_mixed_type_converter.derived_column == "mixed_column"

    # @pytest.mark.parametrize(
    #     "expected, expected_converted_values"
    # )
    # def it_can_analyze_numeric_values(self, request, column):
    #     conv_maybe_update_col_dtype_ = method_mock(
    #         request, _ConvertDfToMixedType, "_maybe_update_col_dtype"
    #     )
    #     conv_update_converted_values_ = method_mock(
    #         request, _ConvertDfToMixedType, "_update_converted_values"
    #     )
    #     multi_type = DataFrameMock.df_multi_type(100)
    #     mixed_type_converter = _ConvertDfToMixedType(column=column)
    #     pd_to_numeric = function_mock(request, "pandas.to_numeric")
    #     pd_to_numeric.return_value = pd.Series([0, 1, 3, np.nan, np.nan])

    #     mixed_type_converter._convert_numeric_values(
    #         pd.Series(["0", "1", "3", "str_0", None])
    #     )

    #     # assert mixed_type_converter._converted_values ==

    # def it_can_create_derived_column(self, request):
    #     df_multi_type = DataFrameMock.df_multi_type(100)
    #     df_mixed_type_converter = _ConvertDfToMixedType(
    #         column="str_column", derived_column="mixed_derived_col"
    #     )
