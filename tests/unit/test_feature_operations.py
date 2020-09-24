from unittest.mock import call

import pytest
from trousse import feature_operations as fop
from trousse.dataset import Dataset

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
        self, columns, expected_length
    ):
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
        self, derived_columns, expected_length
    ):
        with pytest.raises(ValueError) as err:
            fop.FillNA(columns=["nan"], derived_columns=derived_columns, value=0)

        assert isinstance(err.value, ValueError)
        assert f"Length of derived_columns must be 1, found {expected_length}" == str(
            err.value
        )

    def it_calls_fillna(self, request):
        initializer_mock(request, Dataset)
        fillna_ = method_mock(request, Dataset, "fillna")
        fillna_.return_value = Dataset("fake/path")
        dataset = Dataset("fake/path")
        columns = ["nan_0"]
        derived_columns = ["filled_0"]
        value = 0
        fillna_fop = fop.FillNA(
            columns=columns, derived_columns=derived_columns, value=value
        )

        filled_dataset = fillna_fop(dataset)

        fillna_.assert_called_once_with(
            dataset,
            columns=columns,
            derived_columns=derived_columns,
            value=value,
            inplace=False,
        )
        assert isinstance(filled_dataset, Dataset)


class DescribeOperationsList:
    def it_can_construct_itself(self, request):
        _init_ = initializer_mock(request, fop._OperationsList)

        operations_list = fop._OperationsList()

        _init_.assert_called_once_with(ANY)
        assert isinstance(operations_list, fop._OperationsList)

    def it_can_iadd_first_featop(self, request):
        tolist_ = function_mock(request, "trousse.feature_operations.tolist")
        tolist_.side_effect = [["col0", "col1"], ["col2", "col3"]]
        op_list = fop._OperationsList()
        feat_op = fop.FillNA(
            columns=["col0", "col1"], derived_columns=["col2", "col3"], value=0
        )

        op_list += feat_op

        assert op_list._operations_list == [feat_op]
        for column in ["col0", "col1", "col2", "col3"]:
            assert op_list._operations_by_column[column] == [feat_op]
        assert tolist_.call_args_list == [
            call(["col0", "col1"]),
            call(["col2", "col3"]),
        ]

    def it_can_iadd_next_featop(self, request):
        tolist_ = function_mock(request, "trousse.feature_operations.tolist")
        tolist_.side_effect = [["col1"], ["col4"]]
        op_list = fop._OperationsList()
        feat_op0 = fop.FillNA(
            columns=["col0", "col1"], derived_columns=["col2", "col3"], value=0
        )
        feat_op1 = fop.FillNA(columns="col1", derived_columns="col4", value=1)
        op_list._operations_list = [feat_op0]
        for column in ["col0", "col1", "col2", "col3"]:
            op_list._operations_by_column[column] = [feat_op0]

        op_list += feat_op1

        assert op_list._operations_list == [feat_op0, feat_op1]
        for column in ["col0", "col2", "col3"]:
            assert op_list._operations_by_column[column] == [feat_op0]
        assert op_list._operations_by_column["col1"] == [feat_op0, feat_op1]
        assert op_list._operations_by_column["col4"] == [feat_op1]
        assert tolist_.call_args_list == [
            call("col1"),
            call("col4"),
        ]

    def it_can_getitem_from_int(self):
        op_list = fop._OperationsList()
        feat_op0 = fop.FillNA(
            columns=["col0", "col1"], derived_columns=["col2", "col3"], value=0
        )
        feat_op1 = fop.FillNA(columns="col1", derived_columns="col4", value=1)
        op_list._operations_list = [feat_op0, feat_op1]
        for column in ["col0", "col2", "col3"]:
            op_list._operations_by_column[column] = [feat_op0]
        op_list._operations_by_column["col1"] = [feat_op0, feat_op1]
        op_list._operations_by_column["col4"] = [feat_op1]

        feat_op0_ = op_list[0]
        feat_op1_ = op_list[1]

        assert isinstance(feat_op0_, fop.FillNA)
        assert isinstance(feat_op1_, fop.FillNA)
        assert feat_op0_ == feat_op0
        assert feat_op1_ == feat_op1

    def it_can_getitem_from_str(self):
        op_list = fop._OperationsList()
        feat_op0 = fop.FillNA(
            columns=["col0", "col1"], derived_columns=["col2", "col3"], value=0
        )
        feat_op1 = fop.FillNA(columns="col1", derived_columns="col4", value=1)
        op_list._operations_list = [feat_op0, feat_op1]
        for column in ["col0", "col2", "col3"]:
            op_list._operations_by_column[column] = [feat_op0]
        op_list._operations_by_column["col1"] = [feat_op0, feat_op1]
        op_list._operations_by_column["col4"] = [feat_op1]

        feat_op_col0 = op_list["col0"]
        feat_op_col1 = op_list["col1"]

        assert isinstance(feat_op_col0, list)
        assert isinstance(feat_op_col1, list)
        assert feat_op_col0 == [feat_op0]
        assert feat_op_col1 == [feat_op0, feat_op1]
