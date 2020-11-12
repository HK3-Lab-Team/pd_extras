import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as sk_preproc

from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock
from ..unitutil import (ANY, function_mock, initializer_mock, instance_mock,
                        method_mock)


class DescribeFeatureOperation:
    @pytest.mark.parametrize(
        "columns, expected_length",
        [
            (["nan", "nan"], 2),
            ([], 0),
        ],
    )
    def it_knows_how_to_validate_columns_valueerror(
        self, columns, expected_length, is_sequence_and_not_str_
    ):
        # overriding __abstractmethods__ lets you instantiate an abstract class
        # (PEP 3119)
        fop.FeatureOperation.__abstractmethods__ = set()
        feature_operation = fop.FeatureOperation()
        is_sequence_and_not_str_.return_value = True

        with pytest.raises(ValueError) as err:
            feature_operation._validate_single_element_columns(columns)

        assert isinstance(err.value, ValueError)
        assert f"Length of columns must be 1, found {expected_length}" == str(err.value)

    @pytest.mark.parametrize(
        "columns, expected_type",
        [("nan", "str"), (dict(), "dict"), (set(), "set")],
    )
    def it_knows_how_to_validate_columns_typeerror(
        self, columns, expected_type, is_sequence_and_not_str_
    ):
        # overriding __abstractmethods__ lets you instantiate an abstract class
        # (PEP 3119)
        fop.FeatureOperation.__abstractmethods__ = set()
        feature_operation = fop.FeatureOperation()
        is_sequence_and_not_str_.return_value = False

        with pytest.raises(TypeError) as err:
            feature_operation._validate_single_element_columns(columns)

        assert isinstance(err.value, TypeError)
        assert f"columns parameter must be a list, found {expected_type}" == str(
            err.value
        )

    @pytest.mark.parametrize(
        "derived_columns, expected_length",
        [
            (["nan", "nan"], 2),
            ([], 0),
        ],
    )
    def it_knows_how_to_validate_derived_columns_valueerror(
        self, derived_columns, expected_length, is_sequence_and_not_str_
    ):
        # overriding __abstractmethods__ lets you instantiate an abstract class
        # (PEP 3119)
        fop.FeatureOperation.__abstractmethods__ = set()
        feature_operation = fop.FeatureOperation()
        is_sequence_and_not_str_.return_value = True

        with pytest.raises(ValueError) as err:
            feature_operation._validate_single_element_derived_columns(derived_columns)

        assert isinstance(err.value, ValueError)
        assert f"Length of derived_columns must be 1, found {expected_length}" == str(
            err.value
        )

    @pytest.mark.parametrize(
        "derived_columns, expected_type",
        [("nan", "str"), (dict(), "dict"), (set(), "set")],
    )
    def it_knows_how_to_validate_derived_columns_typeerror(
        self, derived_columns, expected_type, is_sequence_and_not_str_
    ):
        # overriding __abstractmethods__ lets you instantiate an abstract class
        # (PEP 3119)
        fop.FeatureOperation.__abstractmethods__ = set()
        feature_operation = fop.FeatureOperation()
        is_sequence_and_not_str_.return_value = False

        with pytest.raises(TypeError) as err:
            feature_operation._validate_single_element_derived_columns(derived_columns)

        assert isinstance(err.value, TypeError)
        assert (
            f"derived_columns parameter must be a list, found {expected_type}"
            == str(err.value)
        )

    # ====================
    #      FIXTURES
    # ====================

    @pytest.fixture
    def is_sequence_and_not_str_(self, request):
        return function_mock(
            request, "trousse.feature_operations.is_sequence_and_not_str"
        )


class DescribeTrousse:
    @pytest.mark.parametrize(
        "operations",
        [
            (),
            (
                fop.FillNA(
                    columns=["nan"],
                    value=0,
                ),
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"a": "b"},
                ),
                fop.FillNA(columns=["nan"], value=0),
            ),
        ],
    )
    def it_contructs_from_args(self, request, operations):
        _init_ = initializer_mock(request, fop.Trousse)

        trousse = fop.Trousse(*operations)

        _init_.assert_called_once_with(ANY, *operations)
        assert isinstance(trousse, fop.Trousse)

    def it_knows_its_operations(
        self,
        replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b,
        fillna_col0_col1,
    ):
        trousse = fop.Trousse(
            replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b, fillna_col0_col1
        )

        operations = trousse.operations

        assert type(operations) == tuple
        assert operations == (
            replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b,
            fillna_col0_col1,
        )

    def it_knows_how_to_call(
        self,
        request,
        replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b,
        fillna_col0_col1,
    ):
        dataset_in = instance_mock(request, Dataset, "in")
        dataset_out_1 = instance_mock(request, Dataset, "1")
        dataset_out_2 = instance_mock(request, Dataset, "2")
        _call_replacestrings = method_mock(request, fop.ReplaceStrings, "__call__")
        _call_replacestrings.return_value = dataset_out_1
        _call_fillna = method_mock(request, fop.FillNA, "__call__")
        _call_fillna.return_value = dataset_out_2
        trousse = fop.Trousse(
            replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b, fillna_col0_col1
        )

        new_dataset = trousse(dataset_in)

        _call_replacestrings.assert_called_once_with(
            replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b, dataset_in
        )
        _call_fillna.assert_called_once_with(fillna_col0_col1, dataset_out_1)
        assert isinstance(new_dataset, Dataset)
        assert new_dataset == dataset_out_2

    def it_knows_its_str(self, fillna_col0_col1, fillna_col1_col4):
        trousse = fop.Trousse(fillna_col0_col1, fillna_col1_col4)

        _str = str(trousse)

        assert type(_str) == str
        assert _str == (
            "Trousse: (FillNA(\n\tcolumns=['col0'],\n\tvalue=0,\n\t"
            "derived_columns=['col1'],\n), FillNA(\n\tcolumns=['col1']"
            ",\n\tvalue=0,\n\tderived_columns=['col4'],\n))"
        )


class DescribeFillNa:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.FillNA)

        fillna = fop.FillNA(columns=["nan"], derived_columns=["filled"], value=0)

        _init_.assert_called_once_with(
            ANY, columns=["nan"], derived_columns=["filled"], value=0
        )
        assert isinstance(fillna, fop.FillNA)

    def and_it_validates_its_arguments(self, request):
        validate_columns_ = method_mock(
            request, fop.FillNA, "_validate_single_element_columns"
        )
        validate_derived_columns_ = method_mock(
            request, fop.FillNA, "_validate_single_element_derived_columns"
        )

        fillna = fop.FillNA(columns=["nan"], derived_columns=["filled"], value=0)

        validate_columns_.assert_called_once_with(fillna, ["nan"])
        validate_derived_columns_.assert_called_once_with(fillna, ["filled"])

    @pytest.mark.parametrize(
        "columns, derived_columns, expected_new_columns, expected_inplace",
        [
            (["nan_0"], ["filled_nan_0"], ["filled_nan_0"], False),
            (["nan_0"], None, [], True),
        ],
    )
    def it_can_apply_fillna(
        self, request, columns, derived_columns, expected_new_columns, expected_inplace
    ):
        df = DataFrameMock.df_many_nans(nan_ratio=0.5, n_columns=3)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        pd_fillna_ = method_mock(request, pd.Series, "fillna")
        pd_fillna_.return_value = pd.Series([0] * 100)
        fillna = fop.FillNA(columns=columns, derived_columns=derived_columns, value=0)

        filled_dataset = fillna._apply(dataset)

        assert filled_dataset is not None
        assert filled_dataset is not dataset
        assert isinstance(filled_dataset, Dataset)
        for col in expected_new_columns:
            assert col in filled_dataset.data.columns
        get_df_from_csv_.assert_called_once_with("fake/path0")
        assert len(pd_fillna_.call_args_list) == len(columns)
        pd.testing.assert_series_equal(
            pd_fillna_.call_args_list[0][0][0], df[columns[0]]
        )
        assert pd_fillna_.call_args_list[0][1] == {"inplace": expected_inplace}

    def it_can_fillna_with_template_call(self, request):
        _apply_ = method_mock(request, fop.FillNA, "_apply")
        track_history_ = method_mock(request, Dataset, "track_history")
        df = DataFrameMock.df_many_nans(nan_ratio=0.5, n_columns=3)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset_in = Dataset(data_file="fake/path0")
        dataset_out = Dataset(data_file="fake/path0")
        _apply_.return_value = dataset_out
        fillna = fop.FillNA(
            columns=["nan_0"], derived_columns=["filled_nan_0"], value=0
        )

        filled_dataset = fillna(dataset_in)

        _apply_.assert_called_once_with(fillna, dataset_in)
        track_history_.assert_called_once_with(filled_dataset, fillna)
        assert filled_dataset is dataset_out

    @pytest.mark.parametrize(
        "other, expected_equal",
        [
            (fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0), True),
            (fop.FillNA(columns=["col9"], derived_columns=["col1"], value=0), False),
            (fop.FillNA(columns=["col0"], derived_columns=["col2"], value=1), False),
            (dict(), False),
        ],
    )
    def it_knows_if_equal(self, other, expected_equal):
        feat_op = fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)

        equal = feat_op == other

        assert type(equal) == bool
        assert equal == expected_equal

    def it_knows_its_str(self):
        feat_op = fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)

        _str = str(feat_op)

        assert type(_str) == str
        assert _str == (
            "FillNA(\n\tcolumns=['col0'],\n\tvalue=0,\n\tderived_columns=['col1'],\n)"
        )


class DescribeReplaceStrings:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.ReplaceStrings)

        replace_strings = fop.ReplaceStrings(
            columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )

        _init_.assert_called_once_with(
            ANY, columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )
        assert isinstance(replace_strings, fop.ReplaceStrings)

    def and_it_validates_its_arguments(self, request):
        validate_columns_ = method_mock(
            request, fop.ReplaceStrings, "_validate_single_element_columns"
        )
        validate_derived_columns_ = method_mock(
            request, fop.ReplaceStrings, "_validate_single_element_derived_columns"
        )
        validate_replacement_map = method_mock(
            request, fop.ReplaceStrings, "_validate_replacement_map"
        )

        replace_strings = fop.ReplaceStrings(
            columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )

        validate_columns_.assert_called_once_with(replace_strings, ["col0"])
        validate_derived_columns_.assert_called_once_with(replace_strings, ["col1"])
        validate_replacement_map.assert_called_once_with(replace_strings, {"a": "b"})

    @pytest.mark.parametrize(
        "columns, derived_columns, expected_new_columns, expected_inplace",
        [
            (["exam_num_col_0"], ["col1"], ["col1"], False),
            (["exam_num_col_0"], None, [], True),
        ],
    )
    def it_can_apply_replace_strings(
        self, request, columns, derived_columns, expected_new_columns, expected_inplace
    ):
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        pd_replace_ = method_mock(request, pd.Series, "replace")
        pd_replace_.return_value = pd.Series([0] * 100)
        replace_strings = replace_strings = fop.ReplaceStrings(
            columns=columns, derived_columns=derived_columns, replacement_map={"a": "b"}
        )

        replaced_dataset = replace_strings._apply(dataset)

        assert replaced_dataset is not None
        assert replaced_dataset is not dataset
        assert isinstance(replaced_dataset, Dataset)
        for col in expected_new_columns:
            assert col in replaced_dataset.data.columns
        get_df_from_csv_.assert_called_once_with("fake/path0")
        assert len(pd_replace_.call_args_list) == len(columns)
        pd.testing.assert_series_equal(
            pd_replace_.call_args_list[0][0][0], df[columns[0]]
        )
        assert pd_replace_.call_args_list[0][1] == {
            "inplace": expected_inplace,
            "to_replace": {"a": "b"},
        }

    def it_can_replace_with_template_call(self, request):
        _apply_ = method_mock(request, fop.ReplaceStrings, "_apply")
        track_history_ = method_mock(request, Dataset, "track_history")
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset_in = Dataset(data_file="fake/path0")
        dataset_out = Dataset(data_file="fake/path0")
        _apply_.return_value = dataset_out
        replace_strings = fop.ReplaceStrings(
            columns=["exam_num_col_0"],
            derived_columns=["replaced_exam_num_col_0"],
            replacement_map={"a": "b"},
        )

        replaced_dataset = replace_strings(dataset_in)

        _apply_.assert_called_once_with(replace_strings, dataset_in)
        track_history_.assert_called_once_with(replaced_dataset, replace_strings)
        assert replaced_dataset is dataset_out

    @pytest.mark.parametrize(
        "other, expected_equal",
        [
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                True,
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"c": "d", "a": "b"},
                ),
                True,
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_1"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                False,
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_1"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                False,
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={
                        "a": "b",
                    },
                ),
                False,
            ),
            (
                fop.ReplaceStrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={
                        "c": "b",
                    },
                ),
                False,
            ),
            (dict(), False),
        ],
    )
    def it_knows_if_equal(self, other, expected_equal):
        feat_op = fop.ReplaceStrings(
            columns=["exam_num_col_0"],
            derived_columns=["replaced_exam_num_col_0"],
            replacement_map={"a": "b", "c": "d"},
        )

        equal = feat_op == other

        assert type(equal) == bool
        assert equal == expected_equal

    def it_knows_its_str(self):
        feat_op = fop.ReplaceStrings(
            columns=["exam_num_col_0"],
            derived_columns=["replaced_exam_num_col_0"],
            replacement_map={"a": "b", "c": "d"},
        )

        _str = str(feat_op)

        assert type(_str) == str
        assert _str == (
            "ReplaceStrings(\n\tcolumns=['exam_num_col_0'],\n\treplacement_map="
            "{'a': 'b', 'c': 'd'},\n\tderived_columns=['replaced_exam_num_col_0'],\n)"
        )


class DescribeReplaceSubstrings:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.ReplaceSubstrings)

        replace_substrings = fop.ReplaceSubstrings(
            columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )

        _init_.assert_called_once_with(
            ANY, columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )
        assert isinstance(replace_substrings, fop.ReplaceSubstrings)

    def and_it_validates_its_arguments(self, request):
        validate_columns_ = method_mock(
            request, fop.ReplaceSubstrings, "_validate_single_element_columns"
        )
        validate_derived_columns_ = method_mock(
            request, fop.ReplaceSubstrings, "_validate_single_element_derived_columns"
        )
        validate_replacement_map = method_mock(
            request, fop.ReplaceSubstrings, "_validate_replacement_map"
        )

        replace_strings = fop.ReplaceSubstrings(
            columns=["col0"], derived_columns=["col1"], replacement_map={"a": "b"}
        )

        validate_columns_.assert_called_once_with(replace_strings, ["col0"])
        validate_derived_columns_.assert_called_once_with(replace_strings, ["col1"])
        validate_replacement_map.assert_called_once_with(replace_strings, {"a": "b"})

    @pytest.mark.parametrize(
        "replacement_map",
        [([]), ({}), ({"a": 1}), ({1: "a"})],
    )
    def it_knows_how_to_validate_replacement_map(self, request, replacement_map):
        initializer_mock(request, fop.ReplaceSubstrings)
        replace_strings = fop.ReplaceSubstrings(
            columns=["col0"], derived_columns=["col1"], replacement_map=replacement_map
        )

        with pytest.raises(TypeError) as err:
            replace_strings._validate_replacement_map(replacement_map)

        assert isinstance(err.value, TypeError)
        assert (
            "replacement_map must be a non-empty dict mapping string keys to string "
            "values" == str(err.value)
        )

    @pytest.mark.parametrize(
        "columns, derived_columns, expected_new_columns, expected_inplace",
        [
            (["exam_str_col_0"], ["col1"], ["col1"], False),
            (["exam_str_col_0"], None, [], True),
        ],
    )
    def it_can_apply_replace_strings(
        self, request, columns, derived_columns, expected_new_columns, expected_inplace
    ):
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        pd_str_replace_ = function_mock(request, "pandas.Series.str.replace")
        pd_str_replace_.return_value = pd.Series([0] * 100)
        replace_substrings = fop.ReplaceSubstrings(
            columns=columns, derived_columns=derived_columns, replacement_map={"a": "b"}
        )

        replaced_dataset = replace_substrings._apply(dataset)

        assert replaced_dataset is not None
        assert replaced_dataset is not dataset
        assert isinstance(replaced_dataset, Dataset)
        for col in expected_new_columns:
            assert col in replaced_dataset.data.columns
        get_df_from_csv_.assert_called_once_with("fake/path0")
        assert len(pd_str_replace_.call_args_list) == len(columns)
        pd.testing.assert_series_equal(
            pd_str_replace_.call_args_list[0][0][0][:], df[columns[0]]
        )
        assert pd_str_replace_.call_args_list[0][1] == {
            "pat": "a",
            "repl": "b",
        }

    def it_can_replace_with_template_call(self, request):
        _apply_ = method_mock(request, fop.ReplaceSubstrings, "_apply")
        track_history_ = method_mock(request, Dataset, "track_history")
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset_in = Dataset(data_file="fake/path0")
        dataset_out = Dataset(data_file="fake/path0")
        _apply_.return_value = dataset_out
        replace_substrings = fop.ReplaceSubstrings(
            columns=["exam_num_col_0"],
            derived_columns=["exam_str_col_0"],
            replacement_map={"a": "b"},
        )

        replaced_dataset = replace_substrings(dataset_in)

        _apply_.assert_called_once_with(replace_substrings, dataset_in)
        track_history_.assert_called_once_with(replaced_dataset, replace_substrings)
        assert replaced_dataset is dataset_out

    @pytest.mark.parametrize(
        "other, expected_equal",
        [
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                True,
            ),
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"c": "d", "a": "b"},
                ),
                True,
            ),
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_1"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                False,
            ),
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_1"],
                    replacement_map={"a": "b", "c": "d"},
                ),
                False,
            ),
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={
                        "a": "b",
                    },
                ),
                False,
            ),
            (
                fop.ReplaceSubstrings(
                    columns=["exam_num_col_0"],
                    derived_columns=["replaced_exam_num_col_0"],
                    replacement_map={
                        "c": "b",
                    },
                ),
                False,
            ),
            (dict(), False),
        ],
    )
    def it_knows_if_equal(self, other, expected_equal):
        feat_op = fop.ReplaceSubstrings(
            columns=["exam_num_col_0"],
            derived_columns=["replaced_exam_num_col_0"],
            replacement_map={"a": "b", "c": "d"},
        )

        equal = feat_op == other

        assert type(equal) == bool
        assert equal == expected_equal

    def it_knows_its_str(self):
        feat_op = fop.ReplaceSubstrings(
            columns=["exam_num_col_0"],
            derived_columns=["replaced_exam_num_col_0"],
            replacement_map={"a": "b", "c": "d"},
        )

        _str = str(feat_op)

        assert type(_str) == str
        assert _str == (
            "ReplaceSubstrings(\n\tcolumns=['exam_num_col_0'],\n\treplacement_map="
            "{'a': 'b', 'c': 'd'},\n\tderived_columns=['replaced_exam_num_col_0'],\n)"
        )


class DescribeOrdinalEncoder:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.OrdinalEncoder)

        ordinal_encoder = fop.OrdinalEncoder(columns=["col0"], derived_columns=["col1"])

        _init_.assert_called_once_with(
            ANY,
            columns=["col0"],
            derived_columns=["col1"],
        )
        assert isinstance(ordinal_encoder, fop.OrdinalEncoder)

    def and_it_validates_its_arguments(self, request):
        validate_columns_ = method_mock(
            request, fop.OrdinalEncoder, "_validate_single_element_columns"
        )
        validate_derived_columns_ = method_mock(
            request, fop.OrdinalEncoder, "_validate_single_element_derived_columns"
        )

        ordinal_encoder = fop.OrdinalEncoder(columns=["col0"], derived_columns=["col1"])

        validate_columns_.assert_called_once_with(ordinal_encoder, ["col0"])
        validate_derived_columns_.assert_called_once_with(ordinal_encoder, ["col1"])

    def it_knows_its_encoder(self):
        ordinal_encoder = fop.OrdinalEncoder(columns=["col0"], derived_columns=["col1"])

        encoder_attr = ordinal_encoder.encoder

        assert isinstance(encoder_attr, sk_preproc.OrdinalEncoder)

    @pytest.mark.parametrize(
        "columns, derived_columns, expected_new_columns",
        [
            (
                ["exam_str_col_0"],
                ["col1"],
                ["col1"],
            ),
            (
                ["exam_str_col_0"],
                None,
                [],
            ),
        ],
    )
    def it_can_apply_ordinal_encoder(
        self,
        request,
        columns,
        derived_columns,
        expected_new_columns,
    ):
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        sk_fit_transform_ = method_mock(
            request, sk_preproc.OrdinalEncoder, "fit_transform"
        )
        sk_fit_transform_.return_value = pd.Series(range(100))
        ordinal_encoder = fop.OrdinalEncoder(
            columns=columns,
            derived_columns=derived_columns,
        )

        encoded_dataset = ordinal_encoder._apply(dataset)

        assert encoded_dataset is not None
        assert encoded_dataset is not dataset
        assert isinstance(encoded_dataset, Dataset)
        for col in expected_new_columns:
            assert col in encoded_dataset.data.columns
        get_df_from_csv_.assert_called_once_with("fake/path0")
        assert len(sk_fit_transform_.call_args_list) == len(columns)
        pd.testing.assert_frame_equal(
            sk_fit_transform_.call_args_list[0][0][1], df[[columns[0]]]
        )

    def it_can_encode_with_template_call(self, request):
        _apply_ = method_mock(request, fop.OrdinalEncoder, "_apply")
        track_history_ = method_mock(request, Dataset, "track_history")
        df = DataFrameMock.df_generic(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset_in = Dataset(data_file="fake/path0")
        dataset_out = Dataset(data_file="fake/path0")
        _apply_.return_value = dataset_out
        ordinal_encoder = fop.OrdinalEncoder(
            columns=["exam_num_col_0"],
            derived_columns=["exam_str_col_0"],
        )

        replaced_dataset = ordinal_encoder(dataset_in)

        _apply_.assert_called_once_with(ordinal_encoder, dataset_in)
        track_history_.assert_called_once_with(replaced_dataset, ordinal_encoder)
        assert replaced_dataset is dataset_out

    @pytest.mark.parametrize(
        "other, expected_equal",
        [
            (
                fop.OrdinalEncoder(
                    columns=["exam_num_col_0"],
                    derived_columns=["encoded_exam_num_col_0"],
                ),
                True,
            ),
            (
                fop.OrdinalEncoder(
                    columns=["exam_num_col_1"],
                    derived_columns=["encoded_exam_num_col_0"],
                ),
                False,
            ),
            (
                fop.OrdinalEncoder(
                    columns=["exam_num_col_0"],
                    derived_columns=["encoded_exam_num_col_1"],
                ),
                False,
            ),
            (
                fop.OrdinalEncoder(
                    columns=["exam_num_col_1"],
                    derived_columns=["encoded_exam_num_col_1"],
                ),
                False,
            ),
            (dict(), False),
        ],
    )
    def it_knows_if_equal(self, other, expected_equal):
        feat_op = fop.OrdinalEncoder(
            columns=["exam_num_col_0"],
            derived_columns=["encoded_exam_num_col_0"],
        )

        equal = feat_op == other

        assert type(equal) == bool
        assert equal == expected_equal


class DescribeOneHotEncoder:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.OneHotEncoder)

        one_hot_encoder = fop.OneHotEncoder(
            columns=["col0"], derived_column_suffix="_enc", drop_option="first"
        )

        _init_.assert_called_once_with(
            ANY, columns=["col0"], derived_column_suffix="_enc", drop_option="first"
        )
        assert isinstance(one_hot_encoder, fop.OneHotEncoder)

    def and_it_validates_its_arguments(self, request):
        validate_columns_ = method_mock(
            request, fop.OneHotEncoder, "_validate_single_element_columns"
        )
        validate_drop_option_ = method_mock(
            request, fop.OneHotEncoder, "_validate_drop_option"
        )

        one_hot_encoder = fop.OneHotEncoder(
            columns=["col0"], derived_column_suffix="_enc", drop_option="first"
        )

        validate_columns_.assert_called_once_with(one_hot_encoder, ["col0"])
        validate_drop_option_.assert_called_once_with(one_hot_encoder, "first")

    def it_knows_its_encoder(self):
        one_hot_encoder = fop.OneHotEncoder(
            columns=["col0"], derived_column_suffix="_enc"
        )

        encoder_attr = one_hot_encoder.encoder

        assert isinstance(encoder_attr, sk_preproc.OneHotEncoder)

    @pytest.mark.parametrize(
        "drop_option",
        [("a"), ({}), ("")],
    )
    def it_knows_how_to_validate_drop_option(self, request, drop_option):
        initializer_mock(request, fop.OneHotEncoder)
        one_hot_encoder = fop.OneHotEncoder(
            columns=["col0"], derived_column_suffix="_enc", drop_option=drop_option
        )

        with pytest.raises(ValueError) as err:
            one_hot_encoder._validate_drop_option(drop_option)

        assert isinstance(err.value, ValueError)
        assert (
            f"drop_option '{drop_option}' not valid. Please use 'first' or 'if_binary'"
            == str(err.value)
        )

    @pytest.mark.parametrize(
        "column, derived_column_suffix, drop_option, num_categories, categories, expected_new_columns",
        [
            (
                "str_categorical_col",
                "_enc",
                "first",
                4,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                            "category_2",
                            "category_3",
                            "category_4",
                        ]
                    )
                ],
                [
                    "str_categorical_col_category_1_enc",
                    "str_categorical_col_category_2_enc",
                    "str_categorical_col_category_3_enc",
                    "str_categorical_col_category_4_enc",
                ],
            ),
            (
                "str_categorical_col",
                "_encoded",
                "first",
                4,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                            "category_2",
                            "category_3",
                            "category_4",
                        ]
                    )
                ],
                [
                    "str_categorical_col_category_1_encoded",
                    "str_categorical_col_category_2_encoded",
                    "str_categorical_col_category_3_encoded",
                    "str_categorical_col_category_4_encoded",
                ],
            ),
            (
                "str_categorical_col",
                "_enc",
                "if_binary",
                5,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                            "category_2",
                            "category_3",
                            "category_4",
                        ]
                    )
                ],
                [
                    "str_categorical_col_category_0_enc",
                    "str_categorical_col_category_1_enc",
                    "str_categorical_col_category_2_enc",
                    "str_categorical_col_category_3_enc",
                    "str_categorical_col_category_4_enc",
                ],
            ),
            (
                "str_categorical_col",
                "_enc",
                None,
                5,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                            "category_2",
                            "category_3",
                            "category_4",
                        ]
                    )
                ],
                [
                    "str_categorical_col_category_0_enc",
                    "str_categorical_col_category_1_enc",
                    "str_categorical_col_category_2_enc",
                    "str_categorical_col_category_3_enc",
                    "str_categorical_col_category_4_enc",
                ],
            ),
            (
                "bool_col",
                "_encoded",
                "first",
                1,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                        ]
                    )
                ],
                [
                    "bool_col_category_1_encoded",
                ],
            ),
            (
                "bool_col",
                "_encoded",
                "if_binary",
                1,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                        ]
                    )
                ],
                [
                    "bool_col_category_1_encoded",
                ],
            ),
            (
                "bool_col",
                "_encoded",
                None,
                2,
                [
                    np.array(
                        [
                            "category_0",
                            "category_1",
                        ]
                    )
                ],
                [
                    "bool_col_category_0_encoded",
                    "bool_col_category_1_encoded",
                ],
            ),
        ],
    )
    def it_can_apply_one_hot_encoder(
        self,
        request,
        column,
        derived_column_suffix,
        drop_option,
        num_categories,
        categories,
        expected_new_columns,
    ):
        df = DataFrameMock.df_multi_type(sample_size=100)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        sk_fit_transform_ = method_mock(
            request, sk_preproc.OneHotEncoder, "fit_transform"
        )
        sk_fit_transform_.return_value = np.zeros((100, num_categories))

        one_hot_encoder = fop.OneHotEncoder(
            columns=[column],
            derived_column_suffix=derived_column_suffix,
            drop_option=drop_option,
        )
        one_hot_encoder._encoder.categories_ = categories

        encoded_dataset = one_hot_encoder._apply(dataset)

        assert encoded_dataset is not None
        assert encoded_dataset is not dataset
        assert isinstance(encoded_dataset, Dataset)
        for col in expected_new_columns:
            assert col in encoded_dataset.data.columns
        assert one_hot_encoder.derived_columns == expected_new_columns
        get_df_from_csv_.assert_called_once_with("fake/path0")
        pd.testing.assert_frame_equal(
            sk_fit_transform_.call_args_list[0][0][1], df[[column]]
        )

    @pytest.mark.parametrize(
        "other, expected_equal",
        [
            (
                fop.OneHotEncoder(
                    columns=["str_categorical_col"],
                    derived_column_suffix="_enc",
                    drop_option="first",
                ),
                True,
            ),
            (
                fop.OneHotEncoder(
                    columns=["str_categorical_col1"],
                    derived_column_suffix="_enc",
                    drop_option="first",
                ),
                False,
            ),
            (
                fop.OneHotEncoder(
                    columns=["str_categorical_col"],
                    derived_column_suffix="_encoded",
                    drop_option="first",
                ),
                False,
            ),
            (
                fop.OneHotEncoder(
                    columns=["str_categorical_col"],
                    derived_column_suffix="_enc",
                    drop_option="if_binary",
                ),
                False,
            ),
            (dict(), False),
        ],
    )
    def it_knows_if_equal(self, other, expected_equal):
        feat_op = fop.OneHotEncoder(
            columns=["str_categorical_col"],
            derived_column_suffix="_enc",
            drop_option="first",
        )

        equal = feat_op == other

        assert type(equal) == bool
        assert equal == expected_equal
