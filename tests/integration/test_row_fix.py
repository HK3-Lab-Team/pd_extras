import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.trousse.dataframe_with_info import DataFrameWithInfo
from trousse.row_fix import RowFix


class Describe_RowFix:
    @pytest.mark.parametrize(
        "column, input_df, expected_df, dry_run",
        [
            (
                "col_a",
                pd.DataFrame(
                    {"col_a": [0, 1, 2, "e", "3", 5], "col_b": [0, 1, 2, "e", "3", 5]}
                ),
                pd.DataFrame(
                    {"col_a": [0, 1, 2, "e", "3", 5], "col_b": [0, 1, 2, "e", "3", 5]}
                ),
                True,
            ),
            (
                "col_a",
                pd.DataFrame(
                    {"col_a": [0, 1, 2, "e", "3", 5], "col_b": [0, 1, 2, "e", "3", 5]}
                ),
                pd.DataFrame(
                    {"col_a": [0, 1, 2, np.nan, 3, 5], "col_b": [0, 1, 2, "e", "3", 5]}
                ),
                False,
            ),
        ],
    )
    def test_forced_conversion_to_numeric(self, column, input_df, expected_df, dry_run):
        dfinfo_converted = RowFix({}, {}).forced_conversion_to_numeric(
            DataFrameWithInfo(df_object=input_df),
            columns=[
                column,
            ],
            dry_run=dry_run,
        )

        assert isinstance(dfinfo_converted, DataFrameWithInfo)
        assert_frame_equal(dfinfo_converted.df, expected_df)
