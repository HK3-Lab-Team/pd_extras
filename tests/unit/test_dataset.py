import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import Dataset, _ColumnListByType
from trousse.operations_list import OperationsList

from ..dataset_util import DataFrameMock
from ..unitutil import ANY, function_mock, initializer_mock, method_mock, property_mock


class DescribeDataset:
    @pytest.mark.parametrize(
        "metadata_cols",
        [(("metadata_num_col")), (("metadata_num_col", "string_col")), (())],
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
                    "int_col",
                    "float_col",
                    "same_col",
                    "bool_col",
                    "mixed_type_col",
                    "nan_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "string_col",
                    "str_categorical_col",
                    "str_forced_categorical_col",
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

    def it_knows_its_str(self, request):
        column_list_by_type = _ColumnListByType(
            mixed_type_cols={"mixed0", "mixed1"},
            constant_cols={"constant"},
            numerical_cols={"numerical0", "numerical1"},
            med_exam_col_list={"med0", "med1", "med2"},
            str_cols={"str0", "str1"},
            str_categorical_cols={"strcat0", "strcat1"},
            num_categorical_cols={"numcat0", "numcat1"},
            bool_cols={"bool0", "bool1"},
            other_cols={"other0", "other1"},
        )
        expected_str = (
            "Columns with:\n\t1.\tMixed types: \t\t2\n\t2.\tNumerical types"
            " (float/int): \t2\n\t3.\tString types: \t\t2\n\t4.\tBool types: "
            "\t\t2\n\t5.\tOther types: \t\t2\nAmong these categories:\n\t1.\tString "
            "categorical columns: 2\n\t2.\tNumeric categorical columns: 2\n\t3."
            "\tMedical Exam columns (numerical, no metadata): 3\n\t4.\tOne repeated "
            "value: 1"
        )

        str_ = str(column_list_by_type)

        assert type(str_) == str
        assert str_ == expected_str

    def it_knows_its_data(self, request):
        expected_df = DataFrameMock.df_generic(10)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = expected_df
        dataset = Dataset(data_file="fake/path")

        data = dataset.data

        assert isinstance(data, pd.DataFrame)
        pd.testing.assert_frame_equal(data, expected_df)

    def it_knows_its_operations_history(self, request):
        function_mock(request, "trousse.dataset.get_df_from_csv")
        dataset = Dataset(data_file="fake/path")

        history = dataset.operations_history

        assert isinstance(history, OperationsList)
        assert len(history) == 0

    @pytest.mark.parametrize(
        "metadata_cols, derived_columns, expected_metadata_cols",
        [
            ((), None, set()),
            (
                tuple(["metadata_num_col"]),
                ["derived_metadata_num_col"],
                {"metadata_num_col", "derived_metadata_num_col"},
            ),
            # TODO: add case with a FeatureOperation that can accept more than
            # one derived_column
        ],
    )
    def it_knows_how_to_track_history(
        self, request, metadata_cols, derived_columns, expected_metadata_cols
    ):
        operations_list_iadd_ = method_mock(request, OperationsList, "__iadd__")
        expected_df = DataFrameMock.df_generic(10)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = expected_df
        dataset = Dataset(data_file="fake/path", metadata_cols=metadata_cols)
        feat_op = fop.FillNA(
            columns=["metadata_num_col"], derived_columns=derived_columns, value=0
        )

        dataset.track_history(feat_op)

        assert dataset.metadata_cols == expected_metadata_cols
        operations_list_iadd_.assert_called_once_with(ANY, feat_op)

    @pytest.mark.parametrize(
        "op_from_original_column_ret_value, expected_columns",
        [
            (
                [
                    fop.OrdinalEncoder(
                        columns=["col"], derived_columns=["encoded_col"]
                    ),
                    fop.OrdinalEncoder(columns=["col"], derived_columns=None),
                ],
                ["encoded_col", "col"],
            ),
            (
                [
                    fop.OrdinalEncoder(
                        columns=["col"], derived_columns=["encoded_col"]
                    ),
                    fop.OrdinalEncoder(columns=["col"], derived_columns=None),
                    fop.OrdinalEncoder(
                        columns=["col"], derived_columns=["encoded_col2"]
                    ),
                ],
                ["encoded_col", "col", "encoded_col2"],
            ),
            (
                [],
                [],
            ),
        ],
    )
    def it_knows_how_to_get_encoded_columns_from_original(
        self, request, op_from_original_column_ret_value, expected_columns
    ):
        op_list__op_from_original_column_ = method_mock(
            request, OperationsList, "operations_from_original_column"
        )
        op_list__op_from_original_column_.return_value = (
            op_from_original_column_ret_value
        )
        expected_df = DataFrameMock.df_generic(10)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = expected_df
        dataset = Dataset(data_file="fake/path")

        columns = dataset.encoded_columns_from_original("col")

        assert type(columns) == list
        assert columns == expected_columns
        op_list__op_from_original_column_.assert_called_once_with(
            dataset.operations_history, "col", [fop.OrdinalEncoder, fop.OneHotEncoder]
        )


class DescribeColumnListByType:
    def it_knows_its_str(self, request):
        column_list_by_type_str = (
            "Columns with:\n\t1.\tMixed types: \t\t2\n\t2.\tNumerical types"
            " (float/int): \t2\n\t3.\tString types: \t\t2\n\t4.\tBool types: "
            "\t\t2\n\t5. \tOther types: \t\t2\nAmong these categories:\n\t1.\t"
            "String categorical columns: 2\n\t2.\tNumeric categorical columns: "
            "2\n\t3.\tMedical Exam columns (numerical, no metadata): 3\n\t4.\tOne "
            "repeated value: 1"
        )
        _column_list_by_type_str = method_mock(request, _ColumnListByType, "__str__")
        _column_list_by_type_str.return_value = column_list_by_type_str
        _column_list_by_type = property_mock(request, Dataset, "_columns_type")
        column_list_by_type = _ColumnListByType(
            mixed_type_cols={"mixed0", "mixed1"},
            constant_cols={"constant"},
            numerical_cols={"numerical0", "numerical1"},
            med_exam_col_list={"med0", "med1", "med2"},
            str_cols={"str0", "str1"},
            str_categorical_cols={"strcat0", "strcat1"},
            num_categorical_cols={"numcat0", "numcat1"},
            bool_cols={"bool0", "bool1"},
            other_cols={"other0", "other1"},
        )
        _column_list_by_type.return_value = column_list_by_type
        _nan_columns = method_mock(request, Dataset, "nan_columns")
        _nan_columns.return_value = {"nan0", "nan1"}
        initializer_mock(request, Dataset)
        dataset = Dataset(data_file="fake/path")
        expected_str = column_list_by_type_str + "\nColumns with many NaN: 2"

        str_ = str(dataset)

        assert type(str_) == str
        assert str_ == expected_str
        _column_list_by_type.assert_called_once
        _nan_columns.assert_called_once_with(dataset, 0.999)
