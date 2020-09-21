try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from abc import abstractmethod
from typing import Any, List, Union

from .dataset import Dataset


@runtime_checkable
class FeatureOperation(Protocol):
    """Protocol definining how Operations should be applied on a Dataset."""

    columns: Union[List[str], str]
    derived_columns: Union[List[str], str] = None

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def is_similar(self, other: "FeatureOperation"):
        raise NotImplementedError


class FillNA(FeatureOperation):
    """Fill NaN values in ``columns`` columns with value ``value``.

    By default NaNs are filled in the original columns. To store the result of filling
    in other columns, ``derived_columns`` parameter has to be set with the name of
    the corresponding column names.

    Parameters
    ----------
    columns : Union[List[str], str]
        Names of the columns with NaNs to be filled
    value : Any
        Value used to fill the NaNs
    derived_columns : Union[List[str], str], optional
        Names of the columns where to store the filling result. Default is None,
        meaning that NaNs are filled in the original columns.

    Returns
    -------
    Dataset
        The new Dataset with NaNs filled.

    Raises
    ------
    ValueError
        If the number of columns to be filled is different from the number of the
        columns where to store the result (if ``derived_columns`` is not None)
    """

    def __init__(
        self,
        columns: Union[List[str], str],
        value: Any,
        derived_columns: Union[List[str], str] = None,
    ):
        self.columns = columns
        self.derived_columns = derived_columns
        self.value = value

    def __call__(self, dataset: Dataset) -> Dataset:
        return dataset.fillna(
            columns=self.columns,
            derived_columns=self.derived_columns,
            value=self.value,
            inplace=False,
        )

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError
