import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.trousse.dataframe_with_info import DataFrameWithInfo
from trousse.feature_enum import OperationTypeEnum
from trousse.feature_fix import split_continuous_column_into_bins
from trousse.feature_operation import FeatureOperation


@pytest.mark.parametrize(
    "col_name, input_df, bin_thresholds, infer_upper_lower_bounds, "
    "extra_padding_ratio, expected_df, expected_feat_op,",
    [
        (
            "col_a",
            pd.DataFrame(
                {"col_a": list(range(100)), "col_b": [0.01 * i for i in range(100)]}
            ),
            [30, 70],
            True,
            0.2,
            pd.DataFrame(
                {
                    "col_a": list(range(100)),
                    "col_b": [0.01 * i for i in range(100)],
                    "col_a_bin_id": pd.Series(
                        [0] * 30 + [1] * 40 + [2] * 30, dtype="Int16"
                    ),
                }
            ),
            FeatureOperation(
                original_columns="col_a",
                operation_type=OperationTypeEnum.BIN_SPLITTING,
                encoded_values_map={
                    0: [-19.8, 30],
                    1: [30, 70],
                    2: [70, 118.8],
                },
                derived_columns="col_a_bin_id",
            ),
        ),
        (
            # No range boundary inference, bin_thresholds from min to max value
            "col_a",
            pd.DataFrame(
                {"col_a": list(range(100)), "col_b": [0.01 * i for i in range(100)]}
            ),
            [0, 30, 70, 100],
            False,
            0.2,
            pd.DataFrame(
                {
                    "col_a": list(range(100)),
                    "col_b": [0.01 * i for i in range(100)],
                    "col_a_bin_id": pd.Series(
                        [0] * 30 + [1] * 40 + [2] * 30, dtype="Int16"
                    ),
                }
            ),
            FeatureOperation(
                original_columns="col_a",
                operation_type=OperationTypeEnum.BIN_SPLITTING,
                encoded_values_map={0: [0, 30], 1: [30, 70], 2: [70, 100]},
                derived_columns="col_a_bin_id",
            ),
        ),
        (
            # No range boundary inference, bin_thresholds including a subset of values only,
            # so few values are out of bound and they are converted to pd.NA
            "col_a",
            pd.DataFrame(
                {"col_a": list(range(100)), "col_b": [0.01 * i for i in range(100)]}
            ),
            [10, 30, 70, 110],
            False,
            0.2,
            pd.DataFrame(
                {
                    "col_a": list(range(100)),
                    "col_b": [0.01 * i for i in range(100)],
                    "col_a_bin_id": pd.Series(
                        [pd.NA] * 10 + [0] * 20 + [1] * 40 + [2] * 30, dtype="Int16"
                    ),
                }
            ),
            FeatureOperation(
                original_columns="col_a",
                operation_type=OperationTypeEnum.BIN_SPLITTING,
                encoded_values_map={0: [10, 30], 1: [30, 70], 2: [70, 110]},
                derived_columns="col_a_bin_id",
            ),
        ),
    ],
)
def test_split_continuous_column_into_bins(
    col_name,
    input_df,
    bin_thresholds,
    infer_upper_lower_bounds,
    extra_padding_ratio,
    expected_df,
    expected_feat_op,
):
    dfinfo_converted = split_continuous_column_into_bins(
        DataFrameWithInfo(df_object=input_df),
        col_name=col_name,
        bin_thresholds=bin_thresholds,
        infer_upper_lower_bounds=infer_upper_lower_bounds,
        extra_padding_ratio=extra_padding_ratio,
    )

    new_column_feat_ops = dfinfo_converted.feature_elaborations[f"{col_name}_bin_id"]
    assert len(new_column_feat_ops) == 1
    feat_op = new_column_feat_ops[0]
    assert feat_op.__dict__ == expected_feat_op.__dict__
    assert isinstance(feat_op, FeatureOperation)
    assert isinstance(dfinfo_converted, DataFrameWithInfo)
    assert_frame_equal(dfinfo_converted.df, expected_df)
