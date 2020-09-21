import pytest

from trousse.dataset import Dataset, _ColumnListByType

from ..dataset_util import DataFrameMock
from ..unitutil import initializer_mock, property_mock


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

    def it_knows_its_mixed_type_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(mixed_type_cols={"mixed0", "mixed1"})
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        mixed_type_columns_ = dataset.mixed_type_columns

        assert type(mixed_type_columns_) == set
        assert mixed_type_columns_ == {"mixed0", "mixed1"}
        _columns_type.assert_called_once()

    def it_knows_its_numerical_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(
            numerical_cols={"numerical0", "numerical1"}
        )
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        numerical_columns_ = dataset.numerical_columns

        assert type(numerical_columns_) == set
        assert numerical_columns_ == {"numerical0", "numerical1"}
        _columns_type.assert_called_once()

    def it_knows_its_med_exam_col_list(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(med_exam_col_list={"med0", "med1"})
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        med_exam_col_list_ = dataset.med_exam_col_list

        assert type(med_exam_col_list_) == set
        assert med_exam_col_list_ == {"med0", "med1"}
        _columns_type.assert_called_once()

    def it_knows_its_str_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(str_cols={"str0", "str1"})
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        str_columns_ = dataset.str_columns

        assert type(str_columns_) == set
        assert str_columns_ == {"str0", "str1"}
        _columns_type.assert_called_once()

    def it_knows_its_str_categorical_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(
            str_categorical_cols={"strcat0", "strcat1"}
        )
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        str_categorical_columns_ = dataset.str_categorical_columns

        assert type(str_categorical_columns_) == set
        assert str_categorical_columns_ == {"strcat0", "strcat1"}
        _columns_type.assert_called_once()

    def it_knows_its_num_categorical_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(
            num_categorical_cols={"numcat0", "numcat1"}
        )
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        num_categorical_columns_ = dataset.num_categorical_columns

        assert type(num_categorical_columns_) == set
        assert num_categorical_columns_ == {"numcat0", "numcat1"}
        _columns_type.assert_called_once()

    def it_knows_its_bool_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(bool_cols={"bool0", "bool1"})
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        bool_columns_ = dataset.bool_columns

        assert type(bool_columns_) == set
        assert bool_columns_ == {"bool0", "bool1"}
        _columns_type.assert_called_once()

    def it_knows_its_other_type_columns(self, request):
        _columns_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(other_cols={"other0", "other1"})
        _columns_type.return_value = column_list_by_type
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")

        other_type_columns_ = dataset.other_type_columns

        assert type(other_type_columns_) == set
        assert other_type_columns_ == {"other0", "other1"}
        _columns_type.assert_called_once()
