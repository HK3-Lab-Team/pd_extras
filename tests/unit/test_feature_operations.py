import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock
from ..unitutil import ANY, function_mock, initializer_mock, method_mock


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
        # overriding __abstractmethods__ lets you instantiate an abstract class (PEP 3119)
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
        # overriding __abstractmethods__ lets you instantiate an abstract class (PEP 3119)
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
        # overriding __abstractmethods__ lets you instantiate an abstract class (PEP 3119)
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
        # overriding __abstractmethods__ lets you instantiate an abstract class (PEP 3119)
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
        "replacement_map",
        [([]), ({}), ({"a": 1}), ({1: "a"})],
    )
    def it_knows_how_to_validate_replacement_map(self, request, replacement_map):
        initializer_mock(request, fop.ReplaceStrings)
        replace_strings = fop.ReplaceStrings(
            columns=["col0"], derived_columns=["col1"], replacement_map=replacement_map
        )

        with pytest.raises(TypeError) as err:
            replace_strings._validate_replacement_map(replacement_map)

        assert isinstance(err.value, TypeError)
        assert (
            "replacement_map must be a non-empty dict mapping string keys to string values"
            == str(err.value)
        )

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
