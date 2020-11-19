import pandas as pd
import pytest

import trousse.feature_fix as ffx
from trousse.dataset import Dataset

from ..fixtures import CSV
from ..util import load_expectation


@pytest.mark.parametrize(
    "csv, column, derived_column, expected_csv",
    (
        (
            CSV.generic,
            "col3",
            ["col3_enc"],
            "csv/generic-ordinal-encoded-col3-col3_enc",
        ),
    ),
)
def test_ordinal_encode_column(csv, column, derived_column, expected_csv):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")

    encoded_dataset, new_cols = ffx._ordinal_encode_column(dataset, column)

    pd.testing.assert_frame_equal(encoded_dataset.data, expected_df)
    assert derived_column == new_cols


@pytest.mark.parametrize(
    "csv, column, drop_one_new_column, expected_new_cols, expected_csv",
    (
        (
            CSV.generic,
            "col3",
            True,
            ["col3_abc_enc", "col3_abr_enc"],
            "csv/generic-one-hot-encoded-col3-enc",
        ),
        (
            CSV.generic,
            "col0",
            False,
            ["col0_a_enc", "col0_c_enc", "col0_d_enc"],
            "csv/generic-one-hot-encoded-col0-enc",
        ),
    ),
)
def test_one_hot_encode_column(
    csv, column, drop_one_new_column, expected_new_cols, expected_csv
):
    dataset = Dataset(data_file=csv)
    expected_df = load_expectation(expected_csv, type_="csv")

    encoded_dataset, new_cols = ffx._one_hot_encode_column(
        dataset, column, drop_one_new_column
    )

    assert expected_new_cols == new_cols
    pd.testing.assert_frame_equal(encoded_dataset.data, expected_df, check_dtype=False)
