import pandas as pd
from trousse.dataset import Dataset
from trousse.row_fix import RowFix

from ..preprocessing_util import CSVMock

# is_failed = False

# try:
#     pd.testing.assert_frame_equal(df_info1.df, df_info2.df)
# except AssertionError as e:
#     print(e)
#     is_failed = True

# if is_failed:
#     raise AssertionError("The errors are listed above")


def assert_featureoperations_equal(df_info1: Dataset, df_info2: Dataset):
    # Check the keys of the two feature operations (i.e. the columns that have been elaborated)
    assert set(df_info1.feature_elaborations) == set(df_info2.feature_elaborations)

    for col_name, feat_op_list_1 in df_info1.feature_elaborations.items():
        # Retrieve the list of operations executed on column `col_name` from `df_info2`
        feat_op_list_2 = df_info2.feature_elaborations[col_name]
        assert len(feat_op_list_1) == len(feat_op_list_2)
        # Compare each operation (since the order must be the same)
        for op1 in feat_op_list_1:
            assert any(
                [op1 == op2 for op2 in feat_op_list_2]
            ), f"The operation {op1} is missing in df_info2 and was not executed on column {col_name} "


def assert_dataframewithinfo_equal(df_info1: Dataset, df_info2: Dataset):

    assert assert_featureoperations_equal(df_info1, df_info2)
    pd.testing.assert_frame_equal(df_info1.df, df_info2.df)


def test_initial_formatting(tmpdir):
    """
    Compare dataframe after fixing invalid substring and invalid strings
    """
    (
        dataframe_to_fix_dir,
        dataframe_after_fix_dir,
    ) = CSVMock.csv_with_nans_strings_substrings(
        sample_size=1000, wrong_values_count=20, csv_path=tmpdir
    )
    dataset_to_fix = Dataset(metadata_cols=(), data_file=dataframe_to_fix_dir)
    expected_dataset = Dataset(metadata_cols=(), data_file=dataframe_after_fix_dir)

    fix_tool = RowFix()
    fixed_dataset = fix_tool.fix_common_errors(
        dataset_to_fix, set_to_correct_dtype=True, verbose=True
    )
    list_diff_bool = [
        expected_dataset.df.iloc[s, 5] == fixed_dataset.df.iloc[s, 5]
        for s in range(expected_dataset.df.shape[0])
    ]

    import pdb

    pdb.set_trace()
    print(f"Expected:\n{expected_dataset.df.iloc[list_diff_bool,3]}")
    print(f"Expected:\n{fixed_dataset.df.iloc[list_diff_bool,3]}")

    pd.testing.assert_frame_equal(
        expected_dataset.df, fixed_dataset.df, check_dtype=False
    )
