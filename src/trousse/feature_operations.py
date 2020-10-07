try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from abc import abstractmethod
from typing import Any, List

from .dataset import Dataset
from .util import is_sequence_and_not_str


@runtime_checkable
class FeatureOperation(Protocol):
    """Protocol definining how Operations should be applied on a Dataset."""

    columns: List[str]
    derived_columns: List[str] = None

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
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
    TypeError
            If ``columns`` is not a list
    TypeError
        If ``derived_columns`` is not None and it is not a list
    """

    def __init__(
        self,
        columns: List[str],
        value: Any,
        derived_columns: List[str] = None,
    ):
        if not is_sequence_and_not_str(columns):
            raise TypeError(
                f"columns parameter must be a list, found {type(columns).__name__}"
            )
        if len(columns) != 1:
            raise ValueError(f"Length of columns must be 1, found {len(columns)}")
        if derived_columns is not None:
            if not is_sequence_and_not_str(derived_columns):
                raise TypeError(
                    f"derived_columns parameter must be a list, found {type(derived_columns).__name__}"
                )
            if len(derived_columns) != 1:
                raise ValueError(
                    f"Length of derived_columns must be 1, found {len(derived_columns)}"
                )

        self.columns = columns
        self.derived_columns = derived_columns
        self.value = value

    def __call__(self, dataset: Dataset) -> Dataset:
        dataset = copy.deepcopy(dataset)

        if self.derived_columns is not None:
            filled_col = dataset.data[self.columns[0]].fillna(self.value, inplace=False)
            dataset.data[self.derived_columns[0]] = filled_col
        else:
            dataset.data[self.columns[0]].fillna(self.value, inplace=True)

        return dataset

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a FillNA instance and it has the same fields value.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a FillNA instance and it has the same fields value,
            False otherwise
        """
        if not isinstance(other, FillNA):
            return False
        if (
            self.columns == other.columns
            and self.derived_columns == other.derived_columns
            and self.value == other.value
        ):
            return True

        return False

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError
