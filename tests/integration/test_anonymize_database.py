import os

import pandas as pd
import pytest

from trousse.anonymize_database import anonymize_data

from ..dataset_util import DataFrameMock


@pytest.mark.parametrize(
    "private_cols_to_remove, private_cols_to_map, "
    "expected_anonym_df, expected_private_df",
    [
        (
            ["private_col_a", "private_col_b"],
            ["private_col_a", "private_col_b", "private_col_c"],
            pd.DataFrame(
                {
                    "private_col_c": {
                        0: "col_2_value_0",
                        1: "col_2_value_1",
                        2: "col_2_value_2",
                        3: "col_2_value_3",
                        4: "col_2_value_3",
                    },
                    "data_col_0": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                    "data_col_1": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                    "ID_OWNER": {
                        0: "467ef2006da06554f248d74bf537a2e5a5270321c35963eace344feb32d"
                        "d7b31",
                        1: "42d7ba97aaf0368c3b2e66ac7bb88787480d22ff3e0694a805647cdce1e"
                        "cac73",
                        2: "e605c6ffcbfcb25f252e269b04b77df4a9514effe10d9885b366dfceae8"
                        "2aa24",
                        3: "be7c8a1fc7ff3c143455fb8d2774369ff6e756d804cb1e1765aca079b1a"
                        "0778a",
                        4: "be7c8a1fc7ff3c143455fb8d2774369ff6e756d804cb1e1765aca079b1a"
                        "0778a",
                    },
                }
            ),
            pd.DataFrame(
                {
                    "private_col_a": {
                        0: "col_0_value_0",
                        1: "col_0_value_1",
                        2: "col_0_value_2",
                        3: "col_0_value_3",
                    },
                    "private_col_b": {
                        0: "col_1_value_0",
                        1: "col_1_value_1",
                        2: "col_1_value_2",
                        3: "col_1_value_3",
                    },
                    "private_col_c": {
                        0: "col_2_value_0",
                        1: "col_2_value_1",
                        2: "col_2_value_2",
                        3: "col_2_value_3",
                    },
                    "ID_OWNER": {
                        0: "467ef2006da06554f248d74bf537a2e5a5270321c35963eace344feb32d"
                        "d7b31",
                        1: "42d7ba97aaf0368c3b2e66ac7bb88787480d22ff3e0694a805647cdce1e"
                        "cac73",
                        2: "e605c6ffcbfcb25f252e269b04b77df4a9514effe10d9885b366dfceae8"
                        "2aa24",
                        3: "be7c8a1fc7ff3c143455fb8d2774369ff6e756d804cb1e1765aca079b1a"
                        "0778a",
                    },
                }
            ),
        )
    ],
)
def test_anonymize_data(
    temporary_data_dir,
    private_cols_to_remove,
    private_cols_to_map,
    expected_anonym_df,
    expected_private_df,
):
    original_df = DataFrameMock.df_with_private_info(private_cols=private_cols_to_map)

    anonym_df, private_df = anonymize_data(
        df=original_df,
        file_name="test_original_db_anonymize",
        private_cols_to_remove=private_cols_to_remove,
        private_cols_to_map=private_cols_to_map,
        dest_path=str(temporary_data_dir),
        random_seed=42,
    )

    pd.testing.assert_frame_equal(anonym_df, expected_anonym_df)
    pd.testing.assert_frame_equal(private_df, expected_private_df)


def but_it_raises_filenotfounderror_with_wrong_dest_path():
    original_df = DataFrameMock.df_with_private_info(private_cols=["private_col_a"])

    with pytest.raises(FileNotFoundError) as err:
        anonym_df, private_df = anonymize_data(
            df=original_df,
            file_name="test_original_db_anonymize",
            private_cols_to_remove=["private_col_a"],
            private_cols_to_map=["private_col_a"],
            dest_path="path/fake",
            random_seed=42,
        )
    assert isinstance(err.value, FileNotFoundError)
    assert (
        "[Errno 2] No such file or directory: '"
        + os.path.join("path", "fake", "test_original_db_anonymize_private_info.csv")
        + "'"
    ) == str(err.value)
