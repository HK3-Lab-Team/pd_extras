try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from abc import abstractmethod
from typing import Any, List

from .dataset import Dataset


@runtime_checkable
class FeatureOperation(Protocol):
    """Protocol definining how Operations should be applied on a Dataset."""

    columns: List[str]
    derived_columns: List[str] = None

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def is_similar(self, other: "FeatureOperation"):
        raise NotImplementedError


class FillNA(FeatureOperation):
    """Fill NaN values ``columns`` (single-element list) column with value ``value``.

    By default NaNs are filled in the original columns. To store the result of filling
    in other columns, ``derived_columns`` parameter has to be set with the name of
    the corresponding column names.

    Parameters
    ----------
    columns : List[str]
        Name of the column with NaNs to be filled. It must be a single-element list.
    value : Any
        Value used to fill the NaNs
    derived_columns : List[str], optional
        Name of the column where to store the filling result. Default is None,
        meaning that NaNs are filled in the original column. If not None, it must be a
        single-element list.

    Returns
    -------
    Dataset
        The new Dataset with NaNs filled.

    Raises
    ------
    ValueError
        If ``columns`` or ``derived_columns`` are not a single-element list.
    """

    def __init__(
        self,
        columns: List[str],
        value: Any,
        derived_columns: List[str] = None,
    ):
        if len(columns) != 1:
            raise ValueError(f"Length of columns must be 1, found {len(columns)}")
        if derived_columns is not None and len(derived_columns) != 1:
            raise ValueError(
                f"Length of derived_columns must be 1, found {len(derived_columns)}"
            )

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
