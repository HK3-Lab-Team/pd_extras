import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock
from ..unitutil import function_mock


@pytest.mark.parametrize(
    "columns, derived_columns, expected_df",
    [
        (
            ["nan_0"],
            ["filled_nan_0"],
            DataFrameMock.df_nans_filled(["filled_nan_0"]),
        ),
        (
            ["nan_0"],
            None,
            DataFrameMock.df_nans_filled(["nan_0"]),
        ),
    ],
)
def test_fillna(request, columns, derived_columns, expected_df):
    df = DataFrameMock.df_many_nans(nan_ratio=0.5, n_columns=3)
    get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
    get_df_from_csv_.return_value = df
    dataset = Dataset(data_file="fake/path0")
    fillna = fop.FillNA(columns=columns, derived_columns=derived_columns, value=1)

    filled_dataset = fillna(dataset)

    assert filled_dataset is not dataset
    pd.testing.assert_frame_equal(filled_dataset.data, expected_df)
