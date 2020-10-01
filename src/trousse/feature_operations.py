try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

import collections
from abc import abstractmethod
from typing import Any, List, Union

from .dataset import Dataset
from .util import is_sequence_and_not_str, tolist


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


class _OperationsList:
    def __init__(self):
        self._operations_list = []
        self._operations_by_column = collections.defaultdict(list)

    def __iadd__(self, feat_op: FeatureOperation):
        self._operations_list.append(feat_op)

        columns = tolist(feat_op.columns)
        derived_columns = (
            tolist(feat_op.derived_columns) if feat_op.derived_columns else None
        )

        for column in columns + derived_columns:
            self._operations_by_column[column].append(feat_op)

        return self

    def __getitem__(
        self, label: Union[int, str]
    ) -> Union[FeatureOperation, List[FeatureOperation]]:
        """Retrieve FeatureOperation element.

        Parameters
        ----------
        label : Union[int, str]
            Label used to retrieve the element. If int: get the label-th FeatureOperation.
            If str: ``label`` will be treated as the name of a column and it will get all
            the FeatureOperation associated with that column.

        Returns
        -------
        Union[FeatureOperation, List[FeatureOperation]]
            Requested FeatureOperation(s)

        Raises
        ------
        TypeError
            If ``label`` is not either an integer or a string.
        """
        if isinstance(label, int):
            return self._operations_list[label]
        elif isinstance(label, str):
            return self._operations_by_column[label]
        else:
            raise TypeError(
                f"Cannot get FeatureOperation with a label of type {type(label).__name__}"
            )


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
            if len(derived_columns) != 1:
                raise ValueError(
                    f"Length of derived_columns must be 1, found {len(derived_columns)}"
                )
            if not is_sequence_and_not_str(columns):
                raise TypeError(
                    f"derived_columns parameter must be a list, found {type(derived_columns).__name__}"
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
