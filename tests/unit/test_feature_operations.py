import pytest
from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..unitutil import ANY, initializer_mock, method_mock


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
