# encoding: utf-8

import copy
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Tuple

import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_preproc

from .dataset import Dataset
from .util import is_sequence_and_not_str


class FeatureOperation(ABC):
    """Protocol definining how Operations should be applied on a Dataset."""

    columns: List[str]
    derived_columns: List[str] = None

    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the operation on a new instance of Dataset and track it in the history

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on and with the operation
            tracked in the history
        """

        dataset = self._apply(dataset)
        dataset.track_history(self)
        return dataset

    def _validate_single_element_columns(self, columns: Any) -> None:
        """Validate single-element list ``columns`` attribute

        Parameters
        ----------
        columns : Any
            Object to validate

        Raises
        ------
        TypeError
            If ``columns`` is not a list
        ValueError
            If ``columns`` is not a single-element list.
        """
        if not is_sequence_and_not_str(columns):
            raise TypeError(
                f"columns parameter must be a list, found {type(columns).__name__}"
            )
        if len(columns) != 1:
            raise ValueError(f"Length of columns must be 1, found {len(columns)}")

    def _validate_single_element_derived_columns(self, derived_columns: Any) -> None:
        """Validate single-element list ``derived_columns`` attribute

        Parameters
        ----------
        derived_columns : Any
            Object to validate

        Raises
        ------
        TypeError
            If ``derived_columns`` is not None and it is not a list
        ValueError
            If ``derived_columns`` is not a single-element list.
        """
        if derived_columns is not None:
            if not is_sequence_and_not_str(derived_columns):
                raise TypeError(
                    "derived_columns parameter must be a list, found "
                    f"{type(derived_columns).__name__}"
                )
            if len(derived_columns) != 1:
                raise ValueError(
                    f"Length of derived_columns must be 1, found {len(derived_columns)}"
                )

    @abstractmethod
    def _apply(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_similar(self, other: "FeatureOperation"):
        raise NotImplementedError


class Trousse:
    """Composes several FeatureOperations together.

    Parameters
    ----------
    *operations : Tuple[FeatureOperations]
        Tuple of FeatureOperations to compose
    """

    def __init__(self, *operations: FeatureOperation) -> None:
        self._operations = operations

    @property
    def operations(self) -> Tuple[FeatureOperation]:
        return self._operations

    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply the composed FeatureOperations to the ``dataset`` one after the other.

        Parameters
        ----------
        dataset : Dataset
            Dataset to apply the operations on

        Returns
        -------
        Dataset
            New Dataset instance with the operations applied on
        """
        for op in self._operations:
            dataset = op(dataset)
        return dataset

    def __repr__(self) -> str:
        return self.__class__.__name__ + f": {str(self._operations)}"

    def __str__(self) -> str:
        return self.__class__.__name__ + f": {str(self._operations)}"


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
        self._validate_single_element_columns(columns)
        self._validate_single_element_derived_columns(derived_columns)

        self.columns = columns
        self.derived_columns = derived_columns
        self.value = value

    def _apply(self, dataset: Dataset) -> Dataset:
        """Apply FillNA operation on a new Dataset instance and return it.

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on
        """
        dataset = copy.deepcopy(dataset)

        if self.derived_columns is not None:
            filled_col = dataset.data[self.columns[0]].fillna(self.value, inplace=False)
            dataset.data[self.derived_columns[0]] = filled_col
        else:
            dataset.data[self.columns[0]].fillna(self.value, inplace=True)

        return dataset

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a FillNA instance with the same fields value.

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

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\tvalue={self.value},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )

    def __str__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\tvalue={self.value},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError


class ReplaceSubstrings(FeatureOperation):
    """Replace substrings with strings in ``columns`` (single-element list)

    By default the substrings are replaced in the original columns. To store the result
    of the replacement in other columns, ``derived_columns`` parameter has to be set
    with the name of the corresponding column names.

    Parameters
    ----------
    columns : List[str]
        Name of the column with substrings to be replaced. It must be a single-element
        list.
    replacement_map : Mapping[str, str]
        Substrings replacement map. Must have string keys and string values.
    derived_columns : List[str], optional
        Name of the column where to store the replacement result. Default is None,
        meaning that the substrings are replaced in the original column. If not None, it
        must be a single-element list.

    Returns
    -------
    Dataset
        The new Dataset with substrings replaced.

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
        replacement_map: Mapping[str, str],
        derived_columns: List[str] = None,
    ):

        self._validate_single_element_columns(columns)
        self._validate_single_element_derived_columns(derived_columns)
        self._validate_replacement_map(replacement_map)

        self.columns = columns
        self.replacement_map = replacement_map
        self.derived_columns = derived_columns

    def _apply(self, dataset: Dataset) -> Dataset:
        """Apply ReplaceSubstrings operation on a new Dataset instance and return it.

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on
        """
        dataset = copy.deepcopy(dataset)

        for pattern, replacement in self.replacement_map.items():
            replaced_col = dataset.data[self.columns[0]].str.replace(
                pat=pattern, repl=replacement
            )
            if self.derived_columns is not None:
                dataset.data[self.derived_columns[0]] = replaced_col
            else:
                dataset.data[self.columns[0]] = replaced_col

        return dataset

    def _validate_replacement_map(self, replacement_map: Mapping[str, str]) -> None:
        """Validate ``replacement_map`` dict to map string keys to string values

        Parameters
        ----------
        replacement_map : Mapping[str, str]
            The dict to validate

        Raises
        ------
        TypeError
            If ``replacement_map`` is not a dict, or is an empty dict or if not all keys
            or not all values are strings
        """
        if (
            not isinstance(replacement_map, Mapping)
            or not replacement_map.keys()
            or not all(
                [
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in replacement_map.items()
                ]
            )
        ):
            raise TypeError(
                "replacement_map must be a non-empty dict mapping string keys to string"
                " values"
            )

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a ReplaceSubstrings instance with the same fields
        value.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a ReplaceSubstrings instance and it has the same fields
            value, False otherwise
        """
        if not isinstance(other, ReplaceSubstrings):
            return False
        if (
            self.columns == other.columns
            and self.derived_columns == other.derived_columns
            and self.replacement_map == other.replacement_map
        ):
            return True

        return False

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\treplacement_map={self.replacement_map},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )

    def __str__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\treplacement_map={self.replacement_map},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError


class ReplaceStrings(ReplaceSubstrings):
    """Replace strings with strings in ``columns`` (single-element list)

    By default the strings are replaced in the original columns. To store the result
    of the replacement in other columns, ``derived_columns`` parameter has to be set
    with the name of the corresponding column names.

    Parameters
    ----------
    columns : List[str]
        Name of the column with strings to be replaced. It must be a single-element
        list.
    replacement_map : Mapping[str, str]
        Strings replacement map. Must have string keys and string values.
    derived_columns : List[str], optional
        Name of the column where to store the replacement result. Default is None,
        meaning that the strings are replaced in the original column. If not None, it
        must be a single-element list.

    Returns
    -------
    Dataset
        The new Dataset with strings replaced.

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
        replacement_map: Mapping[str, str],
        derived_columns: List[str] = None,
    ):
        super().__init__(columns, replacement_map, derived_columns)

    def _apply(self, dataset: Dataset) -> Dataset:
        """Apply ReplaceStrings operation on a new Dataset instance and return it.

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on
        """
        dataset = copy.deepcopy(dataset)

        if self.derived_columns is not None:
            replaced_col = dataset.data[self.columns[0]].replace(
                to_replace=self.replacement_map, inplace=False
            )
            dataset.data[self.derived_columns[0]] = replaced_col
        else:
            dataset.data[self.columns[0]].replace(
                to_replace=self.replacement_map, inplace=True
            )

        return dataset

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a ReplaceStrings instance with the same fields
        value.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a ReplaceStrings instance and it has the same fields
            value, False otherwise
        """
        if not isinstance(other, ReplaceStrings):
            return False
        if (
            self.columns == other.columns
            and self.derived_columns == other.derived_columns
            and self.replacement_map == other.replacement_map
        ):
            return True

        return False

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\treplacement_map={self.replacement_map},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )

    def __str__(self) -> str:
        return (
            self.__class__.__name__ + f"(\n\tcolumns={self.columns},"
            f"\n\treplacement_map={self.replacement_map},"
            f"\n\tderived_columns={self.derived_columns},\n)"
        )


class OneHotEncoder(FeatureOperation):
    """Encode ``columns`` with a one-hot encoder.

    The result of the encoding is stored in new columns with names following the pattern
    `<column>_<category><derived_column_suffix>`.

    Parameters
    ----------
    columns : List[str]
        Name of the column to encode. It must be a single-element list.
    derived_column_suffix : str, optional
        Suffix to be added at the end of the new columns name. Default is "_enc"
    """

    _nan_value_placeholder = "NAN_VALUE"

    def __init__(
        self,
        columns: List[str],
        derived_column_suffix: str = "_enc",
    ) -> None:
        self._validate_single_element_columns(columns)

        self.columns = columns
        self.derived_column_suffix = derived_column_suffix
        self._encoder = sk_preproc.OneHotEncoder(sparse=False)

    @property
    def encoder(self) -> sk_preproc.OneHotEncoder:
        return self._encoder

    def _apply(self, dataset: Dataset) -> Dataset:
        """Apply OneHotEncoder operation on a new Dataset instance and return it.

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on
        """
        dataset = copy.deepcopy(dataset)
        data = dataset.data[[self.columns[0]]]

        data, nan_map = self._replace_nan_with_placeholder_value(data)

        columns_enc = pd.DataFrame(self._encoder.fit_transform(data).astype("bool"))
        encoded_categories = self._encoder.categories_[0].tolist()

        encoded_categories, columns_enc = self._remove_nan_category(
            encoded_categories, columns_enc
        )

        columns_enc = self._set_nan_via_mask(columns_enc, nan_map)

        derived_columns_names = [
            f"{self.columns[0]}_{col}{self.derived_column_suffix}"
            for col in encoded_categories
        ]

        self.derived_columns = derived_columns_names
        dataset.data[derived_columns_names] = columns_enc

        return dataset

    def _remove_nan_category(
        self, encoded_categories: List[str], columns_enc: pd.DataFrame
    ) -> Tuple[List[str], pd.DataFrame]:
        """Remove the NaN category from the encoded categories and corresponding column.

        If the NaN category is not present,``encoded_categories`` and
        ``columns_enc`` will be returned without modification.

        Parameters
        ----------
        encoded_categories : List[str]
            List of the encoded categories from which to remove the NaN category
        columns_enc : pd.DataFrame
            Encoded columns from which to remove the NaN category column

        Returns
        -------
        List[str]
            List of the encoded categories name without the placeholder NaN category
        pd.DataFrame
            Encoded columns without the NaN category column
        """
        try:
            nan_category_index = encoded_categories.index(self._nan_value_placeholder)
        except ValueError:
            pass
        else:
            del encoded_categories[nan_category_index]
            columns_enc = columns_enc.drop(nan_category_index, axis=1)
            columns_enc.columns = range(columns_enc.shape[1])

        return encoded_categories, columns_enc

    def _replace_nan_with_placeholder_value(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Replace NaNs in ``df`` with a placeholder value ("NAN_VALUE").

        Parameters
        ----------
        df : pd.DataFrame
            One-column DataFrame in which NaNs are replaced with "NAN_VALUE"

        Returns
        -------
        pd.DataFrame
            One-column DataFrame with NaNs replaced with "NAN_VALUE"
        np.ndarray
            Mask of bool values for each element in ``df`` that indicates whether an
            element was a NaN value and it has been replaced.
        """
        df = df.copy()
        nan_map = df.isna().values.ravel()

        df[self.columns[0]][nan_map] = self._nan_value_placeholder

        return df, nan_map

    def _set_nan_via_mask(self, df: pd.DataFrame, nan_mask: np.ndarray) -> pd.DataFrame:
        """Set encoded rows to NaN according to a boolean mask.

        The returned DataFrame will be of dtype "boolean", the Pandas nullable boolean
        data type.

        Parameters
        ----------
        df : pd.DataFrame
            Encoded rows
        nan_mask : np.ndarray
            Boolean mask, with True where to put the NaN.

        Returns
        -------
        pd.DataFrame
            Encoded rows with NaNs where ``nan_mask`` is True
        """
        df[nan_mask] = pd.NA
        return df.astype("boolean")

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a OneHotEncoder and it has the same fields value.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a OneHotEncoder instance and it has the same fields
            value, False otherwise
        """
        if not isinstance(other, OneHotEncoder):
            return False
        if (
            self.columns == other.columns
            and self.derived_column_suffix == other.derived_column_suffix
        ):
            return True

        return False

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError


class OrdinalEncoder(FeatureOperation):
    """Encode ``columns`` with a ordinal encoder.

    By default, the column is encoded in place. To store the result of the encoding in
    another column, ``derived_columns`` parameter has to be set with
    the name of the corresponding column name.

    Parameters
    ----------
    columns : List[str]
        Name of the column to encode. It must be a single-element list.
    derived_columns : List[str], optional
        Name of the column where to store the encoding result. Default is None,
        meaning that the original column is encoded in place. If not None, it
        must be a single-element list.

    Returns
    -------
    Dataset
        The new Dataset with the column encoded.

    Raises
    ------
    ValueError
        If ``columns`` or ``derived_columns`` are not a single-element list.
    TypeError
            If ``columns`` is not a list
    TypeError
        If ``derived_columns`` is not None and it is not a list
    """

    def __init__(self, columns: List[str], derived_columns: List[str] = None) -> None:
        self._validate_single_element_columns(columns)
        self._validate_single_element_derived_columns(derived_columns)

        self.columns = columns
        self.derived_columns = derived_columns
        self._encoder = sk_preproc.OrdinalEncoder()

    @property
    def encoder(self) -> sk_preproc.OrdinalEncoder:
        return self._encoder

    def _apply(self, dataset: Dataset) -> Dataset:
        """Apply OrdinalEncoder operation on a new Dataset instance and return it.

        Parameters
        ----------
        dataset : Dataset
            The dataset to apply the operation on

        Returns
        -------
        Dataset
            New Dataset instance with the operation applied on
        """
        dataset = copy.deepcopy(dataset)
        data = dataset.data[[self.columns[0]]]

        series_enc = self._encoder.fit_transform(data).astype("int64")

        if self.derived_columns is not None:
            dataset.data[self.derived_columns[0]] = series_enc
        else:
            dataset.data[self.columns[0]] = series_enc

        return dataset

    def __eq__(self, other: Any) -> bool:
        """Return True if ``other`` is a OrdinalEncoder instance with the same fields
        value.

        Parameters
        ----------
        other : Any
            The instance to compare

        Returns
        -------
        bool
            True if ``other`` is a OrdinalEncoder instance and it has the same fields
            value, False otherwise
        """
        if not isinstance(other, OrdinalEncoder):
            return False
        if (
            self.columns == other.columns
            and self.derived_columns == other.derived_columns
        ):
            return True

        return False

    def is_similar(self, other: FeatureOperation):
        raise NotImplementedError
