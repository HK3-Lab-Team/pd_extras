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

    encoded_df, _, new_cols = ffx._ordinal_encode_column(dataset.data, column, False)

    pd.testing.assert_frame_equal(encoded_df, expected_df)
    assert derived_column == new_cols
