from trousse import feature_operations as fop
from trousse.dataset import Dataset

from ..unitutil import initializer_mock, method_mock


def it_calls_fillna(request):
    initializer_mock(request, Dataset)
    fillna_ = method_mock(request, Dataset, "fillna")
    fillna_.return_value = Dataset("fake/path")
    dataset = Dataset("fake/path")
    columns = ["nan_0", "nan_1"]
    derived_columns = ["filled_0", "filled_1"]
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
