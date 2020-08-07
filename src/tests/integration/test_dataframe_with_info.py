import pytest

from ...pd_extras.dataframe_with_info import DataFrameWithInfo
from ..unitutil import DataFrameMock


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
