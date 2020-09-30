try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from abc import abstractmethod
from typing import Any

from .dataset import Dataset


@runtime_checkable
class FeatureOperation(Protocol):
    """Protocol definining how Operations should be applied on a Dataset."""

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def is_similar(self, other: "FeatureOperation"):
        raise NotImplementedError


class FillNA(FeatureOperation):
    """Fill NaN values in ``column`` column with value ``value``.

    By default NaNs are filled in the original column. To store the result of filling
    in other column, ``derived_column`` parameter has to be set with the name of
    the corresponding column name.

    Parameters
    ----------
    column : str
        Name of the column with NaNs to be filled
    value : Any
        Value used to fill the NaNs
    derived_column : str, optional
        Names of the column where to store the filling result. Default is None,
        meaning that NaNs are filled in the original column.

    Returns
    -------
    Dataset
        The new Dataset with NaNs filled.

    Raises
    ------
    TypeError
        If ``column`` or ``derived_column`` are not strings
    """

    def __init__(
        self,
        column: str,
        value: Any,
        derived_column: str = None,
    ):
        self.column = column
        self.derived_column = derived_column
        self.value = value

    def __call__(self, dataset: Dataset) -> Dataset:
        return dataset.fillna(
            column=self.column,
            derived_column=self.derived_column,
            value=self.value,
            inplace=False,
        )

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError
