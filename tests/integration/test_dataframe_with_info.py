import logging
import shelve
from pathlib import Path
from typing import Tuple

import pytest

from pd_extras.dataframe_with_info import (
    ColumnListByType,
    DataFrameWithInfo,
    FeatureOperation,
    _find_samples_by_type,
    _find_single_column_type,
    _split_columns_by_type_parallel,
    copy_df_info_with_new_df,
    read_file,
)
from pd_extras.exceptions import MultipleOperationsFoundError, NotShelveFileError
from pd_extras.feature_enum import EncodingFunctions, OperationTypeEnum

from ..dataframewithinfo_util import DataFrameMock, SeriesMock
from ..featureoperation_util import eq_featureoperation_combs


class Describe_DataFrameWithInfo:
    @pytest.mark.parametrize(
        "nan_ratio, n_columns, expected_many_nan_columns",
        [
            (0.8, 2, {"nan_0", "nan_1"}),
            (0.8, 1, {"nan_0"}),
            (0.8, 0, set()),
            (
                0.0,
                2,
                {
                    "nan_0",
                    "nan_1",
                    "not_nan_0",
                    "not_nan_1",
                    "not_nan_2",
                    "not_nan_3",
                    "not_nan_4",
                },
            ),
            (1.0, 2, {"nan_0", "nan_1"}),
        ],
    )
    def test_many_nan_columns(
        self, request, nan_ratio, n_columns, expected_many_nan_columns
    ):
        df = DataFrameMock.df_many_nans(nan_ratio, n_columns)
        df_info = DataFrameWithInfo(
            df_object=df, nan_percentage_threshold=nan_ratio - 0.01
        )

        many_nan_columns = df_info.many_nan_columns

        assert len(many_nan_columns) == len(expected_many_nan_columns)
        assert isinstance(many_nan_columns, set)
        assert many_nan_columns == expected_many_nan_columns

    @pytest.mark.parametrize(
        "n_columns, expected_same_value_columns",
        [(2, {"same_0", "same_1"}), (1, {"same_0"}), (0, set())],
    )
    def test_same_value_columns(self, request, n_columns, expected_same_value_columns):
        df = DataFrameMock.df_same_value(n_columns)
        df_info = DataFrameWithInfo(df_object=df)

        same_value_columns = df_info.same_value_cols

        assert len(same_value_columns) == len(expected_same_value_columns)
        assert isinstance(same_value_columns, set)
        assert same_value_columns == expected_same_value_columns

    @pytest.mark.parametrize(
        "n_columns, expected_trivial_columns",
        [
            (4, {"nan_0", "nan_1", "same_0", "same_1"}),
            (2, {"nan_0", "same_0"}),
            (0, set()),
        ],
    )
    def test_trivial_columns(self, request, n_columns, expected_trivial_columns):
        df = DataFrameMock.df_trivial(n_columns)
        df_info = DataFrameWithInfo(df_object=df)

        trivial_columns = df_info.trivial_columns

        assert len(trivial_columns) == len(expected_trivial_columns)
        assert isinstance(trivial_columns, set)
        assert trivial_columns == expected_trivial_columns

    @pytest.mark.parametrize(
        "sample_size, expected_categ_cols",
        [
            (
                50,
                {
                    "numerical_3",
                    "numerical_5",
                    "string_3",
                    "string_5",
                    "mixed_3",
                    "mixed_5",
                },
            ),
            (
                100,
                {
                    "numerical_3",
                    "numerical_5",
                    "string_3",
                    "string_5",
                    "mixed_3",
                    "mixed_5",
                },
            ),
            (
                3000,
                {
                    "numerical_3",
                    "numerical_5",
                    "numerical_8",
                    "string_3",
                    "string_5",
                    "string_8",
                    "mixed_3",
                    "mixed_5",
                    "mixed_8",
                },
            ),
            (
                15000,
                {
                    "numerical_3",
                    "numerical_5",
                    "numerical_8",
                    "numerical_40",
                    "string_3",
                    "string_5",
                    "string_8",
                    "string_40",
                    "mixed_3",
                    "mixed_5",
                    "mixed_8",
                    "mixed_40",
                },
            ),
        ],
    )
    def test_get_categorical_cols(self, request, sample_size, expected_categ_cols):
        df_categ = DataFrameMock.df_categorical_cols(sample_size)
        df_info = DataFrameWithInfo(df_object=df_categ)

        categ_cols = df_info._get_categorical_cols(col_list=df_categ.columns)

        assert isinstance(categ_cols, set)
        assert categ_cols == expected_categ_cols

    @pytest.mark.parametrize(
        "metadata_as_features, expected_column_list_type",
        [
            (
                True,
                ColumnListByType(
                    mixed_type_cols={"mixed_type_col"},
                    same_value_cols={"same_col"},
                    numerical_cols={
                        "numerical_col",
                        "num_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                        "metadata_num_col",
                    },
                    med_exam_col_list={
                        "numerical_col",
                        "num_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                        "metadata_num_col",
                    },
                    str_cols={"string_col", "str_categorical_col"},
                    str_categorical_cols={"str_categorical_col"},
                    num_categorical_cols={"num_categorical_col", "nan_col"},
                    other_cols={"datetime_col"},
                    bool_cols={"bool_col"},
                ),
            ),
            (
                False,
                ColumnListByType(
                    mixed_type_cols={"mixed_type_col"},
                    same_value_cols={"same_col"},
                    numerical_cols={
                        "numerical_col",
                        "num_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                    },
                    med_exam_col_list={
                        "numerical_col",
                        "num_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                    },
                    str_cols={"string_col", "str_categorical_col"},
                    str_categorical_cols={"str_categorical_col"},
                    num_categorical_cols={"num_categorical_col", "nan_col"},
                    other_cols={"datetime_col"},
                    bool_cols={"bool_col"},
                ),
            ),
        ],
    )
    def test_column_list_by_type(self, metadata_as_features, expected_column_list_type):
        df_multi_type = DataFrameMock.df_multi_type(sample_size=200)
        df_info = DataFrameWithInfo(
            df_object=df_multi_type,
            metadata_cols=("metadata_num_col",),
            metadata_as_features=metadata_as_features,
        )

        col_list_by_type = df_info.column_list_by_type

        assert isinstance(col_list_by_type, ColumnListByType)
        assert col_list_by_type == expected_column_list_type

    @pytest.mark.parametrize(
        "metadata_as_features, expected_med_exam_col_list",
        [
            (
                True,
                {
                    "numerical_col",
                    "num_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                    "metadata_num_col",
                },
            ),
            (
                False,
                {
                    "numerical_col",
                    "num_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                },
            ),
        ],
    )
    def test_med_exam_col_list(self, metadata_as_features, expected_med_exam_col_list):
        df_multi_type = DataFrameMock.df_multi_type(sample_size=200)
        df_info = DataFrameWithInfo(
            df_object=df_multi_type,
            metadata_cols=("metadata_num_col",),
            metadata_as_features=metadata_as_features,
        )

        med_exam_col_list = df_info.med_exam_col_list

        assert isinstance(med_exam_col_list, set)
        assert med_exam_col_list == expected_med_exam_col_list

    @pytest.mark.parametrize(
        "nan_threshold, expected_least_nan_cols",
        [
            (10, {"0nan_col"}),
            (101, {"0nan_col", "50nan_col"}),
            (199, {"0nan_col", "50nan_col"}),
            (200, {"0nan_col", "50nan_col", "99nan_col"}),
        ],
    )
    def test_least_nan_cols(self, request, nan_threshold, expected_least_nan_cols):
        df_multi_type = DataFrameMock.df_multi_nan_ratio(sample_size=200)
        df_info = DataFrameWithInfo(df_object=df_multi_type)

        least_nan_cols = df_info.least_nan_cols(nan_threshold)

        assert isinstance(least_nan_cols, set)
        assert least_nan_cols == expected_least_nan_cols

    @pytest.mark.parametrize(
        "duplicated_cols_count, expected_contains_dupl_cols_bool",
        [(0, False), (4, True), (2, True)],
    )
    def test_contains_duplicated_features(
        self, request, duplicated_cols_count, expected_contains_dupl_cols_bool
    ):
        df_duplicated_cols = DataFrameMock.df_duplicated_columns(duplicated_cols_count)
        df_info = DataFrameWithInfo(df_object=df_duplicated_cols)

        contains_duplicated_features = df_info.check_duplicated_features()

        assert isinstance(contains_duplicated_features, bool)
        assert contains_duplicated_features is expected_contains_dupl_cols_bool

    def test_show_columns_type(self, request):
        # df_col_names_by_type = DataFrameMock.df_column_names_by_type()
        # expected_cols_to_type_map = {
        #     "bool_col_0": "bool_col",
        #     "bool_col_1": "bool_col",
        #     "string_col_0": "string_col",
        #     "string_col_1": "string_col",
        #     "string_col_2": "string_col",
        #     "numerical_col_0": "numerical_col",
        #     "other_col_0": "other_col",
        #     "mixed_type_col_0": "mixed_type_col",
        #     "mixed_type_col_1": "mixed_type_col",
        #     "mixed_type_col_2": "mixed_type_col",
        #     "mixed_type_col_3": "mixed_type_col",
        # }

        # TODO: Check "print" output or make the method easy to test and then complete test
        pass

    @pytest.mark.parametrize(
        "original_columns, derived_columns, operation_type, encoded_values_map, "
        "encoder, details",
        [
            (  # Case 1: Only metadata columns as original_columns
                ["metadata_num_col", "metadata_str_col"],
                ["exam_num_col_0", "exam_str_col_0"],
                OperationTypeEnum.BIN_SPLITTING,
                {0: "value_0", 1: "value_1"},
                EncodingFunctions.ONEHOT.value(),
                {"key_2": "value_2", "key_3": "value_3"},
            ),
            (  # Case 2: Only one metadata column as original_columns
                ["metadata_num_col", "exam_num_col_0"],
                ["exam_num_col_1", "exam_str_col_0"],
                OperationTypeEnum.CATEGORICAL_ENCODING,
                None,
                None,
                None,
            ),
            (  # Case 3: One of the derived columns is not present in df
                ["metadata_num_col", "exam_num_col_0"],
                ["exam_str_col_0", "exam_str_col_1"],
                OperationTypeEnum.FEAT_COMBOS_ENCODING,
                {0: "value_0", 1: "value_1"},
                EncodingFunctions.ORDINAL.value(),
                {"key_2": "value_2", "key_3": "value_3"},
            ),
            (  # Case 4: Only one derived column, no metadata columns involved
                ["exam_num_col_0", "exam_num_col_1"],
                ["exam_str_col_0"],
                OperationTypeEnum.CATEGORICAL_ENCODING,
                {0: "value_0", 1: "value_1"},
                EncodingFunctions.ONEHOT.value(),
                {"key_2": "value_2", "key_3": "value_3"},
            ),
        ],
    )
    def test_add_operation(
        self,
        request,
        original_columns,
        derived_columns,
        operation_type,
        encoded_values_map,
        encoder,
        details,
    ):
        df = DataFrameMock.df_generic(10)
        df_info = DataFrameWithInfo(
            df_object=df, metadata_cols=("metadata_num_col", "metadata_str_col"),
        )
        feat_op = FeatureOperation(
            original_columns=original_columns,
            derived_columns=derived_columns,
            operation_type=operation_type,
            encoded_values_map=encoded_values_map,
            encoder=encoder,
            details=details,
        )

        df_info.add_operation(feat_op)

        for orig_column in original_columns:
            # Check if the operation is added to each column
            assert feat_op in df_info.feature_elaborations[orig_column]
        for deriv_column in derived_columns:
            # Check that the derived_columns are inserted in the derived_columns
            # attribute of DataFrameWithInfo instance
            assert deriv_column in df_info.derived_columns
            # Check if the operation is added to each column
            assert feat_op in df_info.feature_elaborations[deriv_column]
            # If original cols are all metadata_cols, check if they are
            # added to metadata_cols
            if original_columns == ["metadata_num_col", "metadata_str_col"]:
                assert deriv_column in df_info.metadata_cols

    def test_add_operation_on_previous_one(self, request, df_info_with_operations):
        # Test `add_operation` method when some columns already have other previous
        # FeatureOperation instances associated
        origin_column = "fop_original_col_0"
        deriv_column = "fop_derived_col_1"
        feat_op = FeatureOperation(
            original_columns=[origin_column],
            derived_columns=[deriv_column],
            operation_type=OperationTypeEnum.BIN_SPLITTING,
            encoded_values_map={0: "value_0", 1: "value_1"},
            encoder=EncodingFunctions.ONEHOT.value(),
            details={"key_2": "value_2", "key_3": "value_3"},
        )

        df_info_with_operations.add_operation(feat_op)

        # Check if the previous operations are still present
        assert len(df_info_with_operations.feature_elaborations[origin_column]) == 6
        assert (
            FeatureOperation(
                OperationTypeEnum.BIN_SPLITTING,
                original_columns=["fop_original_col_0", "fop_original_col_1"],
                derived_columns=["fop_derived_col_0", "fop_derived_col_1"],
            )
            in df_info_with_operations.feature_elaborations[origin_column]
        )
        # Check if the operation is added to each column
        assert feat_op in df_info_with_operations.feature_elaborations[origin_column]
        # Check that they are inserted in derived cols attribute
        assert deriv_column in df_info_with_operations.derived_columns
        # Check if the operation is added to each column
        assert feat_op in df_info_with_operations.feature_elaborations[deriv_column]

    @pytest.mark.parametrize(
        "searched_feat_op, expected_found_feat_op",
        [
            (
                {  # Case 1: Encoder attribute not specified
                    "original_columns": ("fop_original_col_0", "fop_original_col_1"),
                    "derived_columns": ["fop_derived_col_0"],
                    "operation_type": OperationTypeEnum.BIN_SPLITTING,
                    "encoded_values_map": None,
                    "encoder": None,
                    "details": None,
                },
                {
                    "original_columns": ("fop_original_col_0", "fop_original_col_1"),
                    "derived_columns": ["fop_derived_col_0"],
                    "operation_type": OperationTypeEnum.BIN_SPLITTING,
                    "encoded_values_map": {0: "value_0", 1: "value_1"},
                    "encoder": None,
                    "details": None,
                },
            ),
            (
                {  # Case 2: Derived_columns not specified
                    "original_columns": ("fop_original_col_0", "fop_original_col_1"),
                    "derived_columns": None,
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "encoded_values_map": None,
                    "encoder": None,
                    "details": None,
                },
                {
                    "original_columns": ("fop_original_col_0", "fop_original_col_1"),
                    "derived_columns": ("fop_derived_col_1",),
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "encoded_values_map": {0: "value_3", 1: "value_4"},
                    "encoder": EncodingFunctions.ORDINAL.value(),
                    "details": {"key_A": "value_A", "key_B": "value_B"},
                },
            ),
            (
                {  # Case 3: Original columns not specified
                    "original_columns": None,
                    "derived_columns": ["fop_derived_col_1"],
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "encoder": EncodingFunctions.ORDINAL.value(),
                    "encoded_values_map": None,
                    "details": None,
                },
                {
                    "original_columns": ("fop_original_col_0", "fop_original_col_1"),
                    "derived_columns": ("fop_derived_col_1",),
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "encoded_values_map": {0: "value_3", 1: "value_4"},
                    "encoder": EncodingFunctions.ORDINAL.value(),
                    "details": {"key_A": "value_A", "key_B": "value_B"},
                },
            ),
            (
                {  # Case 4: Only derived_columns specified
                    "operation_type": OperationTypeEnum.BIN_SPLITTING,
                    "original_columns": None,
                    "derived_columns": ["fop_derived_col_0"],
                    "encoded_values_map": None,
                    "encoder": None,
                    "details": None,
                },
                {
                    "operation_type": OperationTypeEnum.BIN_SPLITTING,
                    "original_columns": ["fop_original_col_0", "fop_original_col_1"],
                    "derived_columns": ["fop_derived_col_0"],
                    "encoded_values_map": {0: "value_0", 1: "value_1"},
                    "encoder": None,
                    "details": None,
                },
            ),
            (
                {  # Case 5: Original, Derived_columns and encoder specified
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "original_columns": ("fop_original_col_0",),
                    "derived_columns": ("fop_derived_col_0",),
                    "encoded_values_map": None,
                    "encoder": EncodingFunctions.ONEHOT.value(),
                    "details": None,
                },
                {
                    "original_columns": ("fop_original_col_0",),
                    "derived_columns": ("fop_derived_col_0",),
                    "operation_type": OperationTypeEnum.CATEGORICAL_ENCODING,
                    "encoded_values_map": {0: "value_0", 1: "value_1"},
                    "encoder": EncodingFunctions.ONEHOT.value(),
                    "details": {"key_2": "value_2", "key_3": "value_3"},
                },
            ),
        ],
    )
    def test_find_operation_in_column(
        self, request, searched_feat_op, expected_found_feat_op, df_info_with_operations
    ):
        feat_op = FeatureOperation(
            original_columns=searched_feat_op["original_columns"],
            derived_columns=searched_feat_op["derived_columns"],
            operation_type=searched_feat_op["operation_type"],
            encoded_values_map=searched_feat_op["encoded_values_map"],
            encoder=searched_feat_op["encoder"],
            details=searched_feat_op["details"],
        )
        expected_found_feat_op = FeatureOperation(
            original_columns=expected_found_feat_op["original_columns"],
            derived_columns=expected_found_feat_op["derived_columns"],
            operation_type=expected_found_feat_op["operation_type"],
            encoded_values_map=expected_found_feat_op["encoded_values_map"],
            encoder=expected_found_feat_op["encoder"],
            details=expected_found_feat_op["details"],
        )

        found_feat_operat = df_info_with_operations.find_operation_in_column(
            feat_operation=feat_op
        )

        assert isinstance(found_feat_operat, FeatureOperation)
        assert found_feat_operat == expected_found_feat_op

    def test_find_operation_in_column_not_found(self, request, df_info_with_operations):
        feat_op = FeatureOperation(
            original_columns=("fop_original_col_0",),
            derived_columns=("fop_derived_col_0",),
            operation_type=OperationTypeEnum.BIN_SPLITTING,
        )

        found_feat_operat = df_info_with_operations.find_operation_in_column(
            feat_operation=feat_op
        )

        assert found_feat_operat is None

    def test_find_operation_in_column_raise_error(
        self, request, df_info_with_operations
    ):
        feat_op = FeatureOperation(
            OperationTypeEnum.BIN_SPLITTING,
            original_columns=["fop_original_col_0", "fop_original_col_1"],
            derived_columns=None,
        )
        with pytest.raises(MultipleOperationsFoundError) as err:
            df_info_with_operations.find_operation_in_column(feat_operation=feat_op)

        assert isinstance(err.value, MultipleOperationsFoundError)
        assert (
            "Multiple operations were found. Please provide additional information"
            in str(err.value)
        )

    @pytest.mark.parametrize(
        "original_column, encoder, expected_encoded_columns",
        [
            (  # Case 1: Everything specified and found
                "fop_original_col_0",
                EncodingFunctions.ONEHOT.value(),
                ("fop_derived_col_0",),
            ),
            # Case 2: No encoder specified
            ("fop_original_col_2", None, ("fop_derived_col_2", "fop_derived_col_3")),
        ],
    )
    def test_get_enc_column_from_original_one_col_found(
        self,
        df_info_with_operations,
        original_column,
        encoder,
        expected_encoded_columns,
    ):
        encoded_columns = df_info_with_operations.get_enc_column_from_original(
            column_name=original_column, encoder=encoder
        )

        assert isinstance(encoded_columns, tuple)
        assert len(encoded_columns) == len(expected_encoded_columns)
        assert set(encoded_columns) == set(expected_encoded_columns)

    @pytest.mark.parametrize(
        "original_column",
        [
            ("fop_derived_col_1"),  # Case 1: column_name in derived_columns
            ("fop_original_col_10"),  # Case 2: No operation associated
        ],
    )
    def test_get_enc_column_from_original_not_found(
        self, df_info_with_operations, original_column,
    ):
        encoded_columns = df_info_with_operations.get_enc_column_from_original(
            column_name=original_column, encoder=None
        )

        assert encoded_columns is None

    def test_get_enc_column_from_original_raise_multicolfound_error(
        self, df_info_with_operations
    ):
        with pytest.raises(MultipleOperationsFoundError) as err:
            df_info_with_operations.get_enc_column_from_original(
                column_name="fop_original_col_0"
            )

        assert isinstance(err.value, MultipleOperationsFoundError)
        assert (
            "Multiple operations were found. Please provide additional information"
            in str(err.value)
        )

    @pytest.mark.parametrize(
        "encoded_column, encoder, expected_original_columns",
        [
            (  # Case 1: Everything specified and found
                "fop_derived_col_0",
                EncodingFunctions.ONEHOT.value(),
                ("fop_original_col_0",),
            ),
            # Case 2: No encoder specified
            ("fop_derived_col_1", None, ("fop_original_col_0", "fop_original_col_1")),
        ],
    )
    def test_get_original_from_enc_column_one_col_found(
        self,
        df_info_with_operations,
        encoded_column,
        encoder,
        expected_original_columns,
    ):
        original_columns = df_info_with_operations.get_original_from_enc_column(
            column_name=encoded_column, encoder=encoder
        )

        assert isinstance(original_columns, tuple)
        assert len(original_columns) == len(expected_original_columns)
        assert set(original_columns) == set(expected_original_columns)

    @pytest.mark.parametrize(
        "encoded_column",
        [
            ("fop_derived_col_10"),  # Case 1: No operation associated
            ("fop_original_col_2"),  # Case 2: Column_name in original_columns
        ],
    )
    def test_get_original_from_enc_column_not_found(
        self, df_info_with_operations, encoded_column,
    ):
        original_columns = df_info_with_operations.get_original_from_enc_column(
            column_name=encoded_column, encoder=None
        )

        assert original_columns is None

    def test_get_original_from_enc_column_raise_multicolfound_error(
        self, df_info_with_operations
    ):
        with pytest.raises(MultipleOperationsFoundError) as err:
            df_info_with_operations.get_original_from_enc_column(
                column_name="fop_derived_col_0"
            )

        assert isinstance(err.value, MultipleOperationsFoundError)
        assert (
            "Multiple operations were found. Please provide additional information"
            in str(err.value)
        )


class Describe_FeatureOperation:
    @pytest.mark.parametrize(
        "feat_op_1_dict, feat_op_2_dict, is_equal_label", eq_featureoperation_combs()
    )
    def test_featureoperation_equals(
        self, feat_op_1_dict, feat_op_2_dict, is_equal_label
    ):
        feat_op_1 = FeatureOperation(
            operation_type=feat_op_1_dict["operation_type"],
            original_columns=feat_op_1_dict["original_columns"],
            derived_columns=feat_op_1_dict["derived_columns"],
            encoder=feat_op_1_dict["encoder"],
        )
        feat_op_2 = FeatureOperation(
            operation_type=feat_op_2_dict["operation_type"],
            original_columns=feat_op_2_dict["original_columns"],
            derived_columns=feat_op_2_dict["derived_columns"],
            encoder=feat_op_2_dict["encoder"],
        )

        are_feat_ops_equal = feat_op_1 == feat_op_2

        assert are_feat_ops_equal == is_equal_label

    def test_featureoperation_equals_with_different_instance_types(self, request):
        feat_op_1 = FeatureOperation(
            operation_type=OperationTypeEnum.BIN_SPLITTING,
            original_columns=("original_column_2",),
            derived_columns=("derived_column_1", "derived_column_2"),
            encoder=EncodingFunctions.ONEHOT.value(),
        )
        feat_op_2 = dict(
            operation_type=OperationTypeEnum.BIN_SPLITTING,
            original_columns=("original_column_2",),
            derived_columns=("derived_column_1", "derived_column_2"),
            encoder=EncodingFunctions.ONEHOT.value(),
        )

        are_feat_ops_equal = feat_op_1 == feat_op_2

        assert are_feat_ops_equal is False


@pytest.mark.parametrize(
    "series_type, expected_col_type_dict",
    [
        ("bool", {"col_name": "column_name", "col_type": "bool_col"}),
        ("string", {"col_name": "column_name", "col_type": "string_col"}),
        ("category", {"col_name": "column_name", "col_type": "string_col"}),
        ("float", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("int", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("float_int", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("interval", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("date", {"col_name": "column_name", "col_type": "other_col"}),
        ("mixed_0", {"col_name": "column_name", "col_type": "mixed_type_col"}),
        ("mixed_1", {"col_name": "column_name", "col_type": "mixed_type_col"}),
        ("mixed_2", {"col_name": "column_name", "col_type": "mixed_type_col"}),
    ],
)
def test_find_single_column_type(request, series_type, expected_col_type_dict):
    serie = SeriesMock.series_by_type(series_type)

    col_type_dict = _find_single_column_type(serie)

    assert col_type_dict == expected_col_type_dict


@pytest.mark.parametrize(
    "col_type, expected_column_single_type_set",
    [
        ("bool_col", {"bool_col_0", "bool_col_1"}),
        ("string_col", {"string_col_0", "string_col_1", "string_col_2"}),
        ("numerical_col", {"numerical_col_0"}),
        ("other_col", {"other_col_0"}),
        (
            "mixed_type_col",
            {
                "mixed_type_col_0",
                "mixed_type_col_1",
                "mixed_type_col_2",
                "mixed_type_col_3",
            },
        ),
    ],
)
def test_find_columns_by_type(request, col_type, expected_column_single_type_set):
    df_col_names_by_type = DataFrameMock.df_column_names_by_type()

    column_single_type_set = _find_samples_by_type(df_col_names_by_type, col_type)

    assert column_single_type_set == expected_column_single_type_set


def test_split_columns_by_type_parallel(request):
    df_by_type = DataFrameMock.df_multi_type(sample_size=10)
    col_list = df_by_type.columns

    cols_by_type_tuple = _split_columns_by_type_parallel(df_by_type, col_list)

    assert cols_by_type_tuple == (
        {"mixed_type_col"},
        {
            "numerical_col",
            "interval_col",
            "num_categorical_col",
            "nan_col",
            "metadata_num_col",
            "same_col",
        },
        {"string_col", "str_categorical_col"},
        {"bool_col"},
        {"datetime_col"},
    )


def test_copy_df_info_with_new_df(df_info_with_operations):
    new_df = DataFrameMock.df_generic(10)

    new_df_info = copy_df_info_with_new_df(
        df_info=df_info_with_operations, new_pandas_df=new_df
    )

    assert isinstance(new_df_info, DataFrameWithInfo)
    conserved_attributes = new_df_info.__dict__.keys() - {"df"}
    for k in conserved_attributes:
        assert new_df_info.__dict__[k] == df_info_with_operations.__dict__[k]
    assert new_df_info.df.equals(new_df)


def test_copy_df_info_with_new_df_log_warning(caplog, df_info_with_operations):
    new_df = DataFrameMock.df_generic(10)
    reduced_new_df = new_df.drop(["exam_num_col_0"], axis=1)

    copy_df_info_with_new_df(
        df_info=df_info_with_operations, new_pandas_df=reduced_new_df
    )

    assert caplog.record_tuples == [
        (
            "root",
            logging.WARNING,
            "Some columns of the previous DataFrameWithInfo instance "
            + "are being lost, but information about operation on them "
            + "is still present",
        )
    ]


def test_to_file(df_info_with_operations, tmpdir):
    filename = tmpdir.join("export_raise_fileexistserr")

    df_info_with_operations.to_file(filename)

    my_shelf = shelve.open(str(filename))
    assert len(my_shelf.keys()) == 1
    exported_df_info = list(my_shelf.values())[0]
    my_shelf.close()
    assert isinstance(exported_df_info, DataFrameWithInfo)
    # This is to identify attribute errors easier
    conserved_attributes = exported_df_info.__dict__.keys() - {"df"}
    for k in conserved_attributes:
        assert exported_df_info.__dict__[k] == df_info_with_operations.__dict__[k]
    assert exported_df_info.df.equals(df_info_with_operations.df)


def test_to_file_raise_fileexistserror(df_info_with_operations, create_generic_file):
    filename = create_generic_file

    with pytest.raises(FileExistsError) as err:
        df_info_with_operations.to_file(filename)

    assert isinstance(err.value, FileExistsError)
    assert (
        f"File {filename} already exists. If overwriting is not a problem, "
        + "set the 'overwrite' argument to True"
        == str(err.value)
    )


def test_read_file(export_df_info_with_operations_to_file_fixture):
    (
        expected_imported_df_info,
        exported_df_info_path,
    ) = export_df_info_with_operations_to_file_fixture

    imported_df_info = read_file(exported_df_info_path)

    assert isinstance(imported_df_info, DataFrameWithInfo)
    # This is to identify attribute errors easier
    conserved_attributes = imported_df_info.__dict__.keys() - {"df"}
    for k in conserved_attributes:
        assert imported_df_info.__dict__[k] == expected_imported_df_info.__dict__[k]
    assert imported_df_info.df.equals(expected_imported_df_info.df)


def test_read_file_raise_notshelvefileerror(create_generic_file):
    with pytest.raises(NotShelveFileError) as err:
        read_file(create_generic_file)

    assert isinstance(err.value, NotShelveFileError)
    assert (
        f"The file {create_generic_file} was not created by 'shelve' module or no "
        f"db type could be determined" == str(err.value)
    )


def test_read_file_raise_typeerror(create_generic_shelve_file):
    with pytest.raises(TypeError) as err:
        read_file(create_generic_shelve_file)

    assert isinstance(err.value, TypeError)
    assert (
        "The object is not a DataFrameWithInfo instance, but it is <class 'str'>"
        == str(err.value)
    )


# ====================
#      FIXTURES
# ====================


@pytest.fixture()
def create_generic_file(tmpdir) -> Path:
    """
    Create and store a generic file using Python built-in functions.

    At the end of tests, this file is removed by the finalizer of the
    'tmpdir' fixture.

    Returns
    -------
    pathlib.Path
        Path of the saved file
    """
    filename = tmpdir.join("generic_file_with_string")
    text_file = open(filename, "w")
    text_file.write("Generic File")
    text_file.close()
    return filename


@pytest.fixture()
def create_generic_shelve_file(tmpdir) -> Path:
    """
    Create and store a generic file using 'shelve' module.

    At the end of tests, this file is removed by the finalizer of the
    'tmpdir' fixture.

    Returns
    -------
    pathlib.Path
        Path of the saved file
    """
    filename = tmpdir.join("generic_shelve_file_with_string")
    my_shelf = shelve.open(str(filename), "n")  # 'n' for new
    my_shelf["shelve_data"] = "Generic File"
    my_shelf.close()
    return filename


@pytest.fixture(scope="function")
def df_info_with_operations() -> DataFrameWithInfo:
    """
    Create DataFrameWithInfo instance with not empty ``feature_elaborations`` attribute.

    The returned DataFrameWithInfo will have ``feature_elaborations`` attribute
    initialized with ``feature_elaborations`` argument.

    Returns
    -------
    DataFrameWithInfo
        DataFrameWithInfo instance containing FeatureOperation instances
        in the `feature_elaborations` attribute
    """
    df_info = DataFrameWithInfo(df_object=DataFrameMock.df_generic(10))

    feat_operat_list = [
        FeatureOperation(
            OperationTypeEnum.BIN_SPLITTING,
            original_columns=["fop_original_col_0", "fop_original_col_1"],
            derived_columns=["fop_derived_col_0", "fop_derived_col_1"],
        ),
        FeatureOperation(
            OperationTypeEnum.BIN_SPLITTING,
            original_columns=["fop_original_col_0", "fop_original_col_1"],
            derived_columns=["fop_derived_col_0"],
            encoded_values_map={0: "value_0", 1: "value_1"},
        ),
        FeatureOperation(
            original_columns=("fop_original_col_0", "fop_original_col_1"),
            derived_columns=("fop_derived_col_1",),
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            encoded_values_map={0: "value_3", 1: "value_4"},
            encoder=EncodingFunctions.ORDINAL.value(),
            details={"key_A": "value_A", "key_B": "value_B"},
        ),
        FeatureOperation(
            original_columns=("fop_original_col_0",),
            derived_columns=("fop_derived_col_0",),
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            encoded_values_map={0: "value_0", 1: "value_1"},
            encoder=EncodingFunctions.ONEHOT.value(),
            details={"key_2": "value_2", "key_3": "value_3"},
        ),
        FeatureOperation(
            original_columns=("fop_original_col_0",),
            derived_columns=("fop_derived_col_0",),
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            encoded_values_map={0: "value_0", 1: "value_1"},
            encoder=EncodingFunctions.ORDINAL.value(),
            details={"key_2": "value_2", "key_3": "value_3"},
        ),
        FeatureOperation(
            original_columns=("fop_original_col_2",),
            derived_columns=("fop_derived_col_2", "fop_derived_col_3"),
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            encoded_values_map={0: "value_3", 1: "value_4"},
            encoder=EncodingFunctions.ONEHOT.value(),
            details={"key_A": "value_A", "key_B": "value_B"},
        ),
    ]
    for feat_op in feat_operat_list:
        for orig_col in feat_op.original_columns:
            df_info.feature_elaborations[orig_col].append(feat_op)

        for der_col in feat_op.derived_columns:
            df_info.feature_elaborations[der_col].append(feat_op)

    return df_info


@pytest.fixture
def export_df_info_with_operations_to_file_fixture(
    df_info_with_operations, tmpdir
) -> Tuple[DataFrameWithInfo, Path]:
    """
    Export a DataFrameWithInfo instance to a file.

    The DataFrameWithInfo instance is created by the fixture ``df_info_with_operations``
    and it is exported using "shelve" module to a file named ``exported_df_info_ops_fixture`` inside
    the folder returned by the fixture ``tmpdir``.

    Returns
    -------
    DataFrameWithInfo
        DataFrameWithInfo instance (created by ``df_info_with_operations`` fixture)
        that is exported to the file
    Path
        Path of the directory where the DataFrameWithInfo instance is saved

    """
    exported_df_info_path = tmpdir / "exported_df_info_ops_fixture"
    my_shelf = shelve.open(str(exported_df_info_path), "n")
    my_shelf["df_info"] = df_info_with_operations
    my_shelf.close()

    return df_info_with_operations, exported_df_info_path
