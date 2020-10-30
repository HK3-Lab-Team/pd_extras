import collections
from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:  # pragma: no cover
    from .feature_operations import FeatureOperation


class OperationsList:
    def __init__(self):
        self._operations_list = []
        self._operations_by_column = collections.defaultdict(list)

    def derived_columns_from_col(self, column: str) -> List[str]:
        """Return name of the columns created by FeatureOperations applied on ``column``

        Parameters
        ----------
        column : str
            The column on which the FeatureOperation has been applied on.

        Returns
        -------
        List[str]
            Name of the columns created by FeatureOperations applied on ``column``.
        """
        derived_columns = []
        operations = self.operations_from_original_column(column)

        for operation in operations:
            if operation.derived_columns is not None:
                derived_columns.extend(operation.derived_columns)

        return derived_columns

    def operations_from_derived_column(
        self, derived_column: str
    ) -> List["FeatureOperation"]:
        """Return the FeatureOperations that generated ``derived_column``

        Parameters
        ----------
        derived_column : str
            The column that has been generated

        Returns
        -------
        List[FeatureOperation]
            FeatureOperations that generated ``derived_column``
        """
        return list(
            filter(
                lambda op: op.derived_columns is not None
                and derived_column in op.derived_columns,
                self[derived_column],
            )
        )

    def operations_from_original_column(
        self, original_column: str
    ) -> List["FeatureOperation"]:
        """Return the FeatureOperations applied on ``original_column``

        Parameters
        ----------
        original_column : str
            The column on which the FeatureOperation has been applied on.

        Returns
        -------
        List[FeatureOperation]
            FeatureOperations applied on ``original_column``
        """
        return list(
            filter(
                lambda op: original_column in op.columns,
                self[original_column],
            )
        )

    def original_columns_from_derived_column(self, derived_column: str) -> List[str]:
        """Return the name of the columns from which ``derived_column`` is generated from.

        Parameters
        ----------
        derived_column : str
            The column that has been generated

        Returns
        -------
        List[str]
            Name of the columns from which ``derived_column`` has been generated from.

        Raises
        ------
        RuntimeError
            If multiple FeatureOperation are found that generated ``derived_column``
        RuntimeError
            If no FeatureOperation are found that generated ``derived_column``
        """
        operations = self.operations_from_derived_column(derived_column)

        if len(operations) > 1:
            raise RuntimeError(
                "Multiple FeatureOperation found that generated column "
                f"{derived_column}... the pipeline is compromised"
            )
        if len(operations) == 0:
            raise RuntimeError(
                "No FeatureOperation found that generated column "
                f"{derived_column}... the pipeline is compromised"
            )

        return operations[0].columns

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a OperationsList and it contains the same operations.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a OperationsList and it contains the same operations,
            False otherwise.
        """
        if not isinstance(other, OperationsList):
            return False
        if not self._operations_list == other._operations_list:
            return False
        return True

    def __getitem__(
        self, label: Union[int, str]
    ) -> Union["FeatureOperation", List["FeatureOperation"]]:
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

    def __iadd__(self, feat_op: "FeatureOperation"):
        self._operations_list.append(feat_op)

        derived_columns = (
            feat_op.derived_columns if feat_op.derived_columns is not None else []
        )

        for column in feat_op.columns + derived_columns:
            self._operations_by_column[column].append(feat_op)

        return self

    def __iter__(self):
        for operation in self._operations_list:
            yield operation

    def __len__(self):
        return len(self._operations_list)
