from collections.abc import Iterable

import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..dataset_util import DataFrameMock
from ..unitutil import ANY, function_mock, initializer_mock, method_mock


class DescribeFillNa:
    def it_construct_from_args(self, request):
        _init_ = initializer_mock(request, fop.FillNA)

        fillna = fop.FillNA(columns=["nan"], derived_columns=["filled"], value=0)

        _init_.assert_called_once_with(
            ANY, columns=["nan"], derived_columns=["filled"], value=0
        )
        assert isinstance(fillna, fop.FillNA)

    @pytest.mark.parametrize(
        "columns, expected_length",
        [
            (["nan", "nan"], 2),
            ([], 0),
        ],
    )
    def but_it_raises_valueerror_with_columns_length_different_than_1(
        self, columns, expected_length, is_sequence_and_not_str_
    ):
        is_sequence_and_not_str_.return_value = True

        with pytest.raises(ValueError) as err:
            fop.FillNA(columns=columns, value=0)

        assert isinstance(err.value, ValueError)
        assert f"Length of columns must be 1, found {expected_length}" == str(err.value)

    @pytest.mark.parametrize(
        "derived_columns, expected_length",
        [
            (["nan", "nan"], 2),
            ([], 0),
        ],
    )
    def but_it_raises_valueerror_with_derived_columns_length_different_than_1(
        self, derived_columns, expected_length, is_sequence_and_not_str_
    ):
        is_sequence_and_not_str_.return_value = True

        with pytest.raises(ValueError) as err:
            fop.FillNA(columns=["nan"], derived_columns=derived_columns, value=0)

        assert isinstance(err.value, ValueError)
        assert f"Length of derived_columns must be 1, found {expected_length}" == str(
            err.value
        )

    @pytest.mark.parametrize(
        "columns, expected_type",
        [("nan", "str"), (dict(), "dict"), (set(), "set")],
    )
    def but_it_raises_typeerror_with_columns_not_list(
        self, columns, expected_type, is_sequence_and_not_str_
    ):
        is_sequence_and_not_str_.return_value = False

        with pytest.raises(TypeError) as err:
            fop.FillNA(columns=columns, value=0)

        assert isinstance(err.value, TypeError)
        assert f"columns parameter must be a list, found {expected_type}" == str(
            err.value
        )

    @pytest.mark.parametrize(
        "derived_columns, expected_type",
        [("nan", "str"), (dict(), "dict"), (set(), "set")],
    )
    def but_it_raises_typeerror_with_derived_columns_not_list(
        self, derived_columns, expected_type, is_sequence_and_not_str_
    ):
        is_sequence_and_not_str_.side_effect = [True, False]

        with pytest.raises(TypeError) as err:
            fop.FillNA(columns=["nan"], derived_columns=derived_columns, value=0)

        assert isinstance(err.value, TypeError)
        assert (
            f"derived_columns parameter must be a list, found {expected_type}"
            == str(err.value)
        )

    @pytest.mark.parametrize(
        "columns, derived_columns, expected_new_columns, expected_inplace",
        [
            (["nan_0"], ["filled_nan_0"], ["filled_nan_0"], False),
            (["nan_0"], None, [], True),
        ],
    )
    def it_can_fillna(
        self, request, columns, derived_columns, expected_new_columns, expected_inplace
    ):
        df = DataFrameMock.df_many_nans(nan_ratio=0.5, n_columns=3)
        get_df_from_csv_ = function_mock(request, "trousse.dataset.get_df_from_csv")
        get_df_from_csv_.return_value = df
        dataset = Dataset(data_file="fake/path0")
        pd_fillna_ = method_mock(request, pd.Series, "fillna")
        pd_fillna_.return_value = pd.Series([0] * 100)
        fillna = fop.FillNA(columns=columns, derived_columns=derived_columns, value=0)

        filled_dataset = fillna(dataset)

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

    # ====================
    #      FIXTURES
    # ====================

    @pytest.fixture
    def is_sequence_and_not_str_(self, request):
        return function_mock(
            request, "trousse.feature_operations.is_sequence_and_not_str"
        )


class DescribeOperationsList:
    def it_can_construct_itself(self, request):
        _init_ = initializer_mock(request, fop._OperationsList)

        operations_list = fop._OperationsList()

        _init_.assert_called_once_with(ANY)
        assert isinstance(operations_list, fop._OperationsList)

    def it_can_iadd_first_featop(self, request, fillna_col0_col1):
        op_list = fop._OperationsList()

        op_list += fillna_col0_col1

        assert op_list._operations_list == [fillna_col0_col1]
        for column in ["col0", "col1"]:
            assert op_list._operations_by_column[column] == [fillna_col0_col1]

    def it_can_iadd_next_featop(self, request, fillna_col0_col1, fillna_col4_none):
        op_list = fop._OperationsList()
        op_list._operations_list = [fillna_col0_col1]
        for column in ["col0", "col1"]:
            op_list._operations_by_column[column] = [fillna_col0_col1]

        op_list += fillna_col4_none

        assert op_list._operations_list == [fillna_col0_col1, fillna_col4_none]
        assert op_list._operations_by_column["col0"] == [fillna_col0_col1]
        assert op_list._operations_by_column["col1"] == [
            fillna_col0_col1,
        ]
        assert op_list._operations_by_column["col4"] == [fillna_col4_none]

    def it_can_getitem_from_int(self, fillna_col0_col1, fillna_col1_col4):
        op_list = fop._OperationsList()
        op_list._operations_list = [fillna_col0_col1, fillna_col1_col4]
        op_list._operations_by_column["col0"] = [fillna_col0_col1]
        op_list._operations_by_column["col1"] = [fillna_col0_col1, fillna_col1_col4]
        op_list._operations_by_column["col4"] = [fillna_col1_col4]

        feat_op0_ = op_list[0]
        feat_op1_ = op_list[1]

        assert isinstance(feat_op0_, fop.FillNA)
        assert isinstance(feat_op1_, fop.FillNA)
        assert feat_op0_ == fillna_col0_col1
        assert feat_op1_ == fillna_col1_col4

    def it_can_getitem_from_str(self, fillna_col0_col1, fillna_col1_col4):
        op_list = fop._OperationsList()
        op_list._operations_list = [fillna_col0_col1, fillna_col1_col4]
        op_list._operations_by_column["col0"] = [fillna_col0_col1]
        op_list._operations_by_column["col1"] = [fillna_col0_col1, fillna_col1_col4]
        op_list._operations_by_column["col4"] = [fillna_col1_col4]

        feat_op_col0 = op_list["col0"]
        feat_op_col1 = op_list["col1"]

        assert isinstance(feat_op_col0, list)
        assert isinstance(feat_op_col1, list)
        assert feat_op_col0 == [fillna_col0_col1]
        assert feat_op_col1 == [fillna_col0_col1, fillna_col1_col4]

    def but_it_raisestypeerror_with_wrong_type(self):
        op_list = fop._OperationsList()

        with pytest.raises(TypeError) as err:
            op_list[{"wrong"}]

        assert isinstance(err.value, TypeError)
        assert "Cannot get FeatureOperation with a label of type set" == str(err.value)

    @pytest.mark.parametrize(
        "column, getitem_return_value, expected_derived_columns",
        [
            (
                "col0",
                [fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)],
                ["col1"],
            ),
            (
                "col1",
                [
                    fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0),
                    fop.FillNA(columns=["col1"], derived_columns=["col4"], value=0),
                    fop.FillNA(columns=["col1"], derived_columns=["col2"], value=0),
                ],
                ["col4", "col2"],
            ),
            (
                "col4",
                [
                    fop.FillNA(columns=["col1"], derived_columns=["col1"], value=0),
                    fop.FillNA(columns=["col4"], derived_columns=None, value=0),
                ],
                [],
            ),
        ],
    )
    def it_can_get_derived_columns_from_col(
        self, request, column, getitem_return_value, expected_derived_columns
    ):
        op_list = fop._OperationsList()
        getitem_ = method_mock(request, fop._OperationsList, "__getitem__")
        getitem_.return_value = getitem_return_value

        derived_columns = op_list.derived_columns_from_col(column)

        assert type(derived_columns) == list
        assert derived_columns == expected_derived_columns

    def it_can_get_operations_from_derived_column(self, request):
        op_list = fop._OperationsList()
        getitem_ = method_mock(request, fop._OperationsList, "__getitem__")
        fop0 = fop.FillNA(columns=["col4"], derived_columns=["col1"], value=0)
        fop1 = fop.FillNA(columns=["col1"], derived_columns=["col4"], value=0)
        fop2 = fop.FillNA(columns=["col4"], derived_columns=None, value=0)
        getitem_.return_value = [fop0, fop1, fop2]

        operations = op_list._operations_from_derived_column("col4")

        assert type(operations) == list
        assert operations == [fop1]

    def it_can_get_original_columns_from_derived_column(self, request):
        op_list = fop._OperationsList()
        _operations_from_derived_column_ = method_mock(
            request, fop._OperationsList, "_operations_from_derived_column"
        )
        _operations_from_derived_column_.return_value = [
            fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)
        ]

        original_columns = op_list.original_columns_from_derived_column("col1")

        assert type(original_columns) == list
        assert original_columns == ["col0"]

    def but_it_raises_runtimeerror_with_multiple_operations_found(self, request):
        op_list = fop._OperationsList()
        _operations_from_derived_column_ = method_mock(
            request, fop._OperationsList, "_operations_from_derived_column"
        )
        _operations_from_derived_column_.return_value = [
            fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0),
            fop.FillNA(columns=["col2"], derived_columns=["col1"], value=0),
        ]

        with pytest.raises(RuntimeError) as err:
            op_list.original_columns_from_derived_column("col1")

        assert isinstance(err.value, RuntimeError)
        assert (
            "Multiple FeatureOperation found that generated column "
            "col1... the pipeline is compromised"
        ) == str(err.value)

    def but_it_raises_runtimeerror_with_zero_operations_found(self, request):
        op_list = fop._OperationsList()
        _operations_from_derived_column_ = method_mock(
            request, fop._OperationsList, "_operations_from_derived_column"
        )
        _operations_from_derived_column_.return_value = []

        with pytest.raises(RuntimeError) as err:
            op_list.original_columns_from_derived_column("col1")

        assert isinstance(err.value, RuntimeError)
        assert (
            "No FeatureOperation found that generated column "
            "col1... the pipeline is compromised"
        ) == str(err.value)

    def it_knows_its_len(self, fillna_col0_col1, fillna_col1_col4):
        op_list = fop._OperationsList()
        op_list._operations_list = [fillna_col0_col1, fillna_col1_col4]

        len_ = len(op_list)

        assert type(len_) == int
        assert len_ == 2

    def it_can_be_iterated_over(
        self, fillna_col0_col1, fillna_col1_col4, fillna_col1_col2
    ):
        op_list = fop._OperationsList()
        op_list._operations_list = [
            fillna_col0_col1,
            fillna_col1_col4,
            fillna_col1_col2,
        ]
        operations = []
        for operation in op_list:
            operations.append(operation)

        assert isinstance(op_list, Iterable)
        assert operations == [
            fillna_col0_col1,
            fillna_col1_col4,
            fillna_col1_col2,
        ]

    # ====================
    #      FIXTURES
    # ====================

    @pytest.fixture
    def fillna_col0_col1(self, request):
        return fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)

    @pytest.fixture
    def fillna_col1_col4(self, request):
        return fop.FillNA(columns=["col1"], derived_columns=["col4"], value=0)

    @pytest.fixture
    def fillna_col4_none(self, request):
        return fop.FillNA(columns=["col4"], derived_columns=None, value=0)

    @pytest.fixture
    def fillna_col1_col2(self, request):
        return fop.FillNA(columns=["col1"], derived_columns=["col2"], value=0)
