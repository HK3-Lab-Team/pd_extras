import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock
from ..fixtures import CSV
from ..unitutil import function_mock
from ..util import load_expectation


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


@pytest.mark.parametrize(
    "csv, columns, derived_columns, expected_csv",
    (
        (
            CSV.generic,
            ["col0"],
            None,
            "csv/generic-replaced-d-a-col0-inplace",
        ),
        (CSV.generic, ["col0"], ["col4"], "csv/generic-replaced-d-a-col0-col4"),
    ),
)
def test_replace_strings(csv, columns, derived_columns, expected_csv):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")
    replace_strings = fop.ReplaceStrings(
        columns=columns, derived_columns=derived_columns, replacement_map={"d": "a"}
    )

    replaced_dataset = replace_strings(dataset)

    pd.testing.assert_frame_equal(replaced_dataset.data, expected_df)


@pytest.mark.parametrize(
    "csv, columns, derived_columns, expected_csv",
    (
        (
            CSV.generic,
            ["col3"],
            None,
            "csv/generic-replaced-substrings-r-c-col3-inplace",
        ),
        (
            CSV.generic,
            ["col3"],
            ["col4"],
            "csv/generic-replaced-substrings-r-c-col3-col4",
        ),
    ),
)
def test_replace_substrings(csv, columns, derived_columns, expected_csv):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")
    replace_substrings = fop.ReplaceSubstrings(
        columns=columns, derived_columns=derived_columns, replacement_map={"r": "c"}
    )

    replaced_dataset = replace_substrings(dataset)

    pd.testing.assert_frame_equal(replaced_dataset.data, expected_df)


@pytest.mark.parametrize(
    "csv, columns, derived_columns, expected_csv",
    (
        (
            CSV.generic,
            ["col3"],
            None,
            "csv/generic-ordinal-encoded-col3-inplace",
        ),
        (
            CSV.generic,
            ["col3"],
            ["col4"],
            "csv/generic-ordinal-encoded-col3-col4",
        ),
    ),
)
def test_ordinal_encoder(csv, columns, derived_columns, expected_csv):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")
    ordinal_encoder = fop.OrdinalEncoder(
        columns=columns, derived_columns=derived_columns
    )

    encoded_dataset = ordinal_encoder(dataset)

    pd.testing.assert_frame_equal(encoded_dataset.data, expected_df)


@pytest.mark.parametrize(
    "csv, columns, derived_column_suffix, expected_csv",
    (
        (
            CSV.generic,
            ["col3"],
            "_enc",
            "csv/generic-one-hot-encoded-col3-enc",
        ),
        (
            CSV.generic,
            ["col3"],
            "_encoded",
            "csv/generic-one-hot-encoded-col3-encoded",
        ),
        (
            CSV.generic,
            ["col0"],
            "_enc",
            "csv/generic-one-hot-encoded-col0-enc",
        ),
        (
            CSV.generic_na,
            ["col4"],
            "_enc",
            "csv/generic-na-one-hot-encoded-col4-enc",
        ),
    ),
)
def test_one_hot_encoder(csv, columns, derived_column_suffix, expected_csv):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")
    one_hot_encoder = fop.OneHotEncoder(
        columns=columns,
        derived_column_suffix=derived_column_suffix,
    )

    encoded_dataset = one_hot_encoder(dataset)
    print(encoded_dataset.data)

    pd.testing.assert_frame_equal(encoded_dataset.data, expected_df, check_dtype=False)
