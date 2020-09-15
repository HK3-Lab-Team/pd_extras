import pytest

from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock


class DescribeDataset:
    @pytest.mark.parametrize(
        "metadata_cols",
        [
            (("metadata_num_col")),
            (("metadata_num_col, string_col")),
            (()),
        ],
    )
    def it_knows_its_metadata_cols(self, metadata_cols):

        df = DataFrameMock.df_multi_type(10)
        dataset = Dataset(df_object=df, metadata_cols=metadata_cols)

        metadata_cols_ = dataset.metadata_cols

        assert type(metadata_cols_) == set
        assert metadata_cols_ == set(metadata_cols)

    @pytest.mark.parametrize(
        "metadata_cols, feature_cols, expected_feature_cols",
        [
            (  # metadata and features non overlapping
                ("metadata_num_col",),
                ("str_categorical_col",),
                {"str_categorical_col"},
            ),
            (  # metadata and features overlapping
                ("metadata_num_col", "string_col"),
                ("metadata_num_col", "str_categorical_col"),
                {"metadata_num_col", "str_categorical_col"},
            ),
            (  # features None --> features are all columns but metadata
                ("metadata_num_col",),
                None,
                {
                    "numerical_col",
                    "same_col",
                    "bool_col",
                    "mixed_type_col",
                    "nan_col",
                    "num_categorical_col",
                    "string_col",
                    "str_categorical_col",
                    "interval_col",
                    "datetime_col",
                },
            ),
        ],
    )
    def it_knows_its_feature_cols(
        self, metadata_cols, feature_cols, expected_feature_cols
    ):
        df = DataFrameMock.df_multi_type(10)
        dataset = Dataset(
            df_object=df, metadata_cols=metadata_cols, feature_cols=feature_cols
        )

        feature_cols_ = dataset.feature_cols

        assert type(feature_cols_) == set
        assert feature_cols_ == expected_feature_cols
