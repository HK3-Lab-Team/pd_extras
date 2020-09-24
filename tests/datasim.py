from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Sequence

import numpy as np
import pandas as pd

INVALID_NAN_BIAS = 0
INVALID_STRING_BIAS = 1
INVALID_SUBSTRING_BIAS = 2


def _sample_index_in_list(
    sample_count: int, step: int, bias: int, list_length: int
) -> List[int]:
    """
    Find the indexes of equally spaced elements in a list following the rule "id=A*x+b"

    In a list with ``list_length`` elements, this function will find the indexes
    of equally spaced elements. These element indexes will follow the rule
            "element_id = ``step`` * id_counter + ``bias``"
    with "id_counter" in [0, ``sample_count``].
    The function also takes into account that the ``list_length`` is finite, and when
    element_id > ``list_length`` the element_id will be calculated at the beginning
    of the list, as if the list was a circular list. This allows the function to always
    return ``sample_count`` elements.

    Parameters
    ----------
    sample_count : int
        Number of equally spaced element indexes that are returned.
    step : int
        Number of elements between two consecutive returned indexes
    bias : int
        Integer value that indicates the starting index from which the function
        starts to count to find the equally spaced indexes.
    list_length : int
        Length of the list from which the function will select few equally spaced
        element indexes.

    Returns
    -------
    List[int]
        List of indexes of equally spaced element from a list with a
        ``list_length`` number of elements. These element indexes will follow the rule:
                "element_id = ``step`` * id_counter + ``bias``"
        with "id_counter" in [0, ``sample_count``].
    """
    element_ids = []
    for id_counter in range(sample_count):
        element_id = step * id_counter + bias
        element_ids.append(element_id % list_length)

    return element_ids


@dataclass
class TestColumn:
    """
    DataClass used to initialize columns in TestDataSet class.

    Parameters
    ----------
    name : str
        Name of the column.
    original_values : Union[pd.Series, np.ndarray, List, Tuple]
        Values of the column.
    dtype : Union[type, str]
        Data type for the output Series. This must be a dtype supported
        by pandas/numpy.
    """
    name: str
    original_values: Union[pd.Series, np.ndarray, List, Tuple]
    dtype: Union[type, str, None]

    def __len__(self) -> int:
        return len(self.original_values)


class _TestColumn:
    def __init__(self, column: TestColumn, col_id: int = None):
        """
        Private Class handling column values to fix and the ones modified after fix.

        This class uses ``column`` to initialize the values of two pandas Series
        that will be contemporarily modified by the applied
        ReverseFeatureOperation:
        - ``_values_to_fix`` -> keeps track of the modified values
        - ``_values_after_fix`` -> keeps track of the values that are
            expected to be found when the user will apply the appropriate
            correction (e.g. result of a test)
        Infact, since the process of properly correcting ``values_to_fix`` cannot
        always be fully reverted (e.g. inserting NaN), each ReverseFeatureOperation
        takes care of providing the modification and the expected correction by
        modifying ``values_to_fix`` and ``values_after_fix``.

        Parameters
        ----------
        column : TestColumn
            TestColumn instance whose values are used to create the instance
        col_id : int
            Integer identifying the number of the column considered. This will be
            used by ReverseFeatureOperation instances in order to avoid replacing
            the values to be fixed in the same samples.
            Otherwise we may end up having samples with invalid values only.

        """
        self._name = column.name
        self._col_id = col_id
        self._dtype = column.dtype

        self._values_to_fix = pd.Series(column.original_values, dtype=column.dtype)
        self._values_after_fix = self._values_to_fix.copy()

    @property
    def name(self):
        return self._name

    @property
    def col_id(self):
        if self._col_id is None:
            raise AttributeError(
                "The column has not a column ID because it is not a column of"
                " a TestDataSet instance"
            )
        else:
            return self._col_id

    @col_id.setter
    def col_id(self, value):
        if self._col_id is None:
            self._col_id = value
        else:
            raise AttributeError(
                "The instance has already a specific ``col_id``."
                " Changing it is forbidden"
            )

    @property
    def dtype(self):
        return self._dtype

    def __len__(self) -> int:
        return len(self.values_to_fix)

    @property
    def values_to_fix(self) -> pd.Series:
        """
        Return the column values where ReverseFeatureOperation introduced values to fix.

        These are the column values on which the user (or a test)
        is supposed to apply the proper correction in order to go back to
        the original dataset (with clean values).
        Since this process cannot always be fully reversible, the expected values
        after the process will be ``values_after_fix`` property.

        Returns
        -------
        pd.Series
            Column values where possible ReverseFeatureOperation introduced
            values to fix.
        """
        return self._values_to_fix

    @property
    def values_after_fix(self) -> pd.Series:
        """
        Return the column values resulting after applying proper correction.

        These are modified along with ``values_to_fix`` to keep
        track of the values inserted.
        Since the process of properly correcting ``values_to_fix`` cannot always
        be fully reversible, these are the column values that the column
        will contain after that process.

        Returns
        -------
        pd.Series
            Column values resulting after applying proper correction to
            ``values_to_fix`` property of the instance.
        """
        return self._values_after_fix

    def update_sample(self, index: int, value_to_fix: Any, value_after_fix: Any):
        """
        Update a single column sample with new values

        Parameters
        ----------
        index : int
            Index of the column sample that is updated
        value_to_fix : Any
            Raw value that needs a fix (usually an error simulated by a
            ReverseFeatureOperation).
        value_to_fix : Any
            Value that is supposed to be found after that the proper correction
            is applied to ``value_to_fix`` argument.
        """
        self._values_to_fix[index] = value_to_fix
        self._values_after_fix[index] = value_after_fix

    def update_column_values(
        self, values_to_fix: Sequence[Any], values_after_fix: Sequence[Any]
    ):
        """
        Update a single column sample with new values

        Parameters
        ----------
        values_to_fix : Sequence[Any]
            Raw values that need a fix (usually values containing errors that have
            been simulated by a ReverseFeatureOperation).
        values_to_fix : Sequence[Any]
            Values that are supposed to be found after that the proper correction
            is applied to ``values_to_fix`` argument.
        """
        self._values_to_fix = values_to_fix
        self._values_after_fix = values_after_fix


class TestDataSet:
    def __init__(self, sample_size: int = None):
        # Creating two dicts with the same elements, so that
        # retrieving an element by name or index takes the same time
        self._columns_by_index = []
        self._sample_size = sample_size
        self._name_to_index_map = {}

    @property
    def sample_size(self) -> Union[int, None]:
        return self._sample_size

    def _validate_testcolumn_to_create(self, column: TestColumn):
        """
        Check the TestColumn instance ``column`` before inserting it in the dataset

        This method checks that:
        - ``column`` is an instance of TestColumn
        - no column with the same name is present in the dataset
        - the ``column`` has the same number of samples as the others

        Parameters
        ----------
        column : TestColumn
            Column that needs to be validated.

        Raises
        ------
        TypeError
            If the ``column`` argument is not a TestColumn instance.
        ValueError
            If another column with the same name is already present in the
            instance, or if the column has a different sample count from the
            other instance columns.
        """
        if not isinstance(column, TestColumn):
            raise TypeError(
                "The 'column' argument must be a TestColumn instance, instead"
                + f" its type is {type(column)}"
            )
        if column.name in self._name_to_index_map:
            raise ValueError(
                "The name and/or the column id is already present in the "
                "Dataset. If an update is required, use __setitem__ by indexing"
            )
        if len(column) != self.sample_size:
            raise ValueError(
                f'`column` argument named "{column.name}" has {len(column)}'
                f" values, while the other dataset columns have {self.sample_size}"
                " values."
            )

    def _validate_testcolumn_to_update(self, column: _TestColumn):
        """
        Check the TestColumn instance ``column`` before inserting it in the dataset

        This method checks that:
        - ``column`` is an instance of _TestColumn
        - another column with the same name is present in the dataset (the one
            to update)
        - the ``column`` has the same number of samples as the others

        Parameters
        ----------
        column : _TestColumn
            Column that needs to be validated.

        Raises
        ------
        TypeError
            If the ``column`` argument is not a _TestColumn instance.
        ValueError
            If no column with the same name is present in the instance,
            or if the column has a different sample count from the
            other instance columns.
        """
        if not isinstance(column, _TestColumn):
            raise TypeError(
                "The 'column' argument must be a _TestColumn instance, instead"
                + f" its type is {type(column)}"
            )
        if column.name not in self._name_to_index_map:
            raise ValueError(
                "The name and/or the column id is not present in the "
                "Dataset. If a new column must be added, use `add_column` method"
            )
        if len(column) != self.sample_size:
            raise ValueError(
                f'`column` argument named "{column.name}" has {len(column)}'
                f" values, while the other dataset columns have {self.sample_size}"
                " values."
            )

    def add_column(self, column: TestColumn):
        """
        Add single TestColumn instance ``column`` to the instance

        Parameters
        ----------
        columns : TestColumn
            Column (TestColumn instance) that is added to this
            TestDataSet instance.

        Raises
        ------
        TypeError
            If the ``column`` argument is not a TestColumn instance.
        ValueError
            If another column with the same name is already present in the
            instance, or if the column has a different sample count from the
            other instance columns.
        """
        # If the sample size is not initialized, set it to the length of the first
        # added column
        if self.sample_size is None:
            self._sample_size = len(column)

        self._validate_testcolumn_to_create(column)

        # Assign the new column ID (it will be used by ReverseFeatureOperation
        # instances in order to avoid replacing the values to be fixed
        # in the same samples)
        col_id = len(self._columns_by_index)
        self._columns_by_index.append(_TestColumn(column, col_id=col_id))
        # Add the corresponding name key to the map
        self._name_to_index_map[column.name] = col_id

    def add_columns(self, columns: List[TestColumn]):
        """
        Add multiple TestColumn instances ``columns`` to this instance

        Parameters
        ----------
        columns : List[TestColumn]
            List of the columns (TestColumn instances) that are added to this
            TestDataSet instance.
        """
        for col in columns:
            self.add_column(col)

    @property
    def dataframe_to_fix(self) -> pd.DataFrame:
        """
        Return the dataset where ReverseFeatureOperation introduced values to fix.

        The returned dataset is modified along with ``dataframe_after_fix`` to keep
        track of the values inserted by the ReverseFeatureOperation instances
        applied to columns.
        Particularly the dataset values are the ones on which the user (or a test)
        is supposed to apply the proper correction in order to go back to
        the original dataset (with clean values).
        Since this process cannot always be fully reversible, the expected values
        after the process are contained in ``dataframe_after_fix``.

        Returns
        -------
        pd.DataFrame
            Column values where possible ReverseFeatureOperation introduced
            values to fix.
        """
        data_dict = {}
        for col_name, column in zip(
            self._name_to_index_map.keys(), self._columns_by_index
        ):
            data_dict[col_name] = column.values_to_fix
        return pd.DataFrame(data_dict)

    @property
    def dataframe_after_fix(self) -> pd.DataFrame:
        """
        Return the dataset resulting after applying proper correction.

        The returned dataset is modified along with ``dataframe_to_fix`` to keep
        track of the values inserted by the ReverseFeatureOperation instances
        applied to columns.
        Since the process of properly correcting ``dataframe_to_fix`` cannot always
        be fully reversible, these are the values that the ``dataframe_to_fix`` dataset
        will contain after that process.

        Returns
        -------
        pd.DataFrame
            Column values resulting after applying proper correction to
            ``dataframe_to_fix`` property of the instance.
        """
        data_dict = {}

        for col_name, column in zip(
            self._name_to_index_map.keys(), self._columns_by_index
        ):
            data_dict[col_name] = column.values_after_fix
        return pd.DataFrame(data_dict)

    def __getitem__(self, item: Union[int, str]) -> _TestColumn:
        if isinstance(item, int):
            return self._columns_by_index[item]
        else:
            return self._columns_by_index[self._name_to_index_map[str(item)]]

    def __setitem__(self, column_name_id: Union[int, str], value: _TestColumn):
        """
        Set a new ``value`` to existing column identified by ``column_name_id``

        Parameters
        ----------
        column_name_id : Union[int, str]
            Name (str) or Id (int) of the column whose value is being set.
        value : _TestColumn
            New _TestColumn instance that replaces the column identified
            by ``column_name_id``.

        Raises
        ------
        TypeError
            If the ``column`` argument is not a _TestColumn instance.
        ValueError
            If no column with the same name is present in the instance,
            or if the column has a different sample count from the
            other instance columns.
        """
        self._validate_testcolumn_to_update(value)

        if isinstance(column_name_id, int):
            self._columns_by_index[column_name_id] = value
        else:
            self._columns_by_index[self._name_to_index_map[column_name_id]] = value

    def __contains__(self, column_name_id: Union[int, str]) -> bool:
        """
        Check if the column identified by ``column_name_id`` is present in TestDataSet

        Parameters
        ----------
        column_name_id : Union[int, str]
            Name or ID of the column whose presence needs to be checked.
        """
        if isinstance(column_name_id, str):
            return column_name_id in self._name_to_index_map
        elif isinstance(column_name_id, int):
            return column_name_id < len(self._columns_by_index)
        else:
            return False

    def __len__(self):
        """Return number of columns in this instance"""
        return len(self._columns_by_index)

    def shape(self) -> Tuple[Union[int, None], int]:
        """
        Return the Dataset shape

        Returns
        -------
        int
            Number of samples in this instance
        int
            Number of columns in this instance
        """
        return (self.sample_size, len(self._columns_by_index))


class ReverseFeatureOperation(ABC):
    def __init__(self, column_names: Union[Tuple[str], Tuple[int]]):
        """
        Abstract Class that revert preprocessing operations on TestDataSets

        Its subclasses apply operations to simulated synthetic data that revert
        the behaviour of the FeatureOperation classes.
        This is an abstract class so the abstract method "apply" needs to be reimplemented
        in subclasses in order to work.

        Parameters
        ----------
        columns : Union[List[str], List[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        """
        self._validate_column_names(column_names)
        self._column_names = column_names

    @staticmethod
    def _validate_column_names(column_names: Union[Tuple[str], Tuple[int]]):
        """
        Check if ``column_names`` does not contain a mix of strings and integers

        Parameters
        ----------
        column_names : Union[List[str], List[int]]
            List of names (string) and IDs (integer) of the columns on which the
            ReverseFeatureOperation will be applied, that will be validated.

        Raises
        ------
        ValueError
            If `column_names` attribute contains a mix of names (string) and
            IDs (integer) of the columns, which is not accepted.
        """
        if not all([isinstance(col, str) for col in column_names]) and not all(
            [isinstance(col, int) for col in column_names]
        ):
            raise ValueError(
                "`column_names` attribute contains a mix of names (string)"
                "and column IDs (integer), which is not accepted."
            )

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Apply the operation to ``column``

        This method performs an operation on ``column`` values that revert the
        behavior of a corresponding "pytrousse" FeatureOperation. Since this is meant
        to be used for checking the behavior of pytrousse FeatureOperation, it
        also keeps track of the related correction (because some Operations like
        inserting NaNs are not fully reversible). For this reason ``column`` keeps
        track of the original raw values (simulated by this ReverseFeatureOperation)
        and of the values that are supposed to be found after the corresponding
        FeatureOperation is applied.
        """
        raise NotImplementedError


class Compose(ReverseFeatureOperation):
    def __init__(self, reverse_feat_ops: List[ReverseFeatureOperation]):
        """
        Apply a sequence of multiple ReverseFeatureOperation instances

        Parameters
        ----------
        reverse_feat_ops: List[ReverseFeatureOperation]
            List of ReverseFeatureOperation instances that will be applied on
            a TestDataSet by using the __call__ method.
        """
        self._feat_ops = reverse_feat_ops

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Apply a sequence of multiple ReverseFeatureOperation instances on ``dataset``

        Parameters
        ----------
        dataset: TestDataSet
            TestDataSet instance on which the multiple ReverseFeatureOperation
            instances used to initialize this Compose instance will be applied.

        Returns
        -------
        dataset: TestDataSet
            TestDataSet instance on which the multiple ReverseFeatureOperation
            instances have been applied.
        """
        for op in self._feat_ops:
            dataset = op(dataset)

        return dataset


class InsertNaNs(ReverseFeatureOperation):
    def __init__(self, error_count: int, column_names: Union[Tuple[str], Tuple[int]]):
        """
        Insert NaN values into the columns ``column_names`` of a TestDataSet instance.

        The function inserts an ``error_count`` number of
        equally-spaced NaN values into a TestDataSet instance.

        Parameters
        ----------
        column_names : Union[List[str], List[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of values that will be replaced with NaNs in each column
        """
        super().__init__(column_names=column_names)
        self._error_count = error_count

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Insert NaN values into ``dataset`` keeping track of the modified samples.

        The function inserts an ``error_count`` number of equally-spaced NaN values
        into the ``dataset``.

        Parameters
        ----------
        dataset : TestDataSet
            TestDataSet instance containing the columns ``column_names`` used
            to initialize this instance and where some NaNs will be inserted.

        Returns
        -------
        TestDataSet
            TestDataSet instance where some NaNs were inserted.
        """
        for col_name in self._column_names:
            column = dataset[col_name]
            nan_sample_ids = _sample_index_in_list(
                sample_count=self._error_count,
                step=dataset.sample_size // self._error_count,
                # This BIAS prevents the function to insert NaNs for the same
                # samples/ids where it may insert other type of errors
                bias=column.col_id + INVALID_NAN_BIAS,
                list_length=dataset.sample_size,
            )

            for sample_id in nan_sample_ids:
                column.update_sample(
                    sample_id, value_to_fix=pd.NA, value_after_fix=pd.NA
                )

            dataset[col_name] = column

        return dataset


class InsertInvalidValues(ReverseFeatureOperation):
    def __init__(
        self,
        error_count: int,
        replacement_map: Dict[str, str],
        column_names: Union[Tuple[str], Tuple[int]],
    ):
        """
        Insert invalid values into the column and store the related correct values.

        The function inserts an ``error_count`` number of
        equally-spaced invalid values into the column. These values are create
        by using ``replacement_map`` dictionary that connects the original
        substring with the invalid one that is replaced.

        Parameters
        ----------
        column_names : Union[List[str], List[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of invalid values that are inserted in column.
        replacement_map : Dict[str, str]
            Dictionary where the keys are the invalid string that will replace
            some elements of ``values`` argument. The dictionary values are the
            values that will replace those invalid values in the `correct_column`
            property.
        """
        super().__init__(column_names=column_names)
        self._error_count = error_count
        self._replacement_map = replacement_map

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Insert invalid values into the column and store the related correct values.

        The function inserts an ``error_count`` number of
        equally-spaced invalid values into the column. These values are create
        by using ``replacement_map`` dictionary that connects the original
        substring with the invalid one that is replaced.

        Parameters
        ----------
        dataset : TestDataSet
            TestDataSet instance containing the columns ``column_names`` used to
            initialize this instance and where some invalid values will be inserted.

        Returns
        -------
        TestDataSet
            TestDataSet instance where some invalid values were inserted
        """
        for col_name in self._column_names:
            column = dataset[col_name]
            invalid_string_sample_ids = _sample_index_in_list(
                sample_count=self._error_count,
                step=dataset.sample_size // self._error_count,
                # This BIAS prevents the function to insert invalid values for
                # the same samples/ids where it may insert other type of errors
                bias=column.col_id + INVALID_STRING_BIAS,
                list_length=dataset.sample_size,
            )

            invalid_string_list = list(self._replacement_map.keys())

            for sample_id in invalid_string_sample_ids:
                error_to_insert = invalid_string_list[
                    sample_id % len(invalid_string_list)
                ]
                column.update_sample(
                    sample_id,
                    value_to_fix=error_to_insert,
                    value_after_fix=self._replacement_map[error_to_insert],
                )

            dataset[col_name] = column

        return dataset


class InsertInvalidSubStrings(ReverseFeatureOperation):
    def __init__(
        self,
        column_names: Union[Tuple[str], Tuple[int]],
        error_count: int,
        replacement_map: Dict[str, str],
    ):
        """
        Insert invalid substrings into the column and store the related correct values.

        The function inserts an ``error_count`` number of
        equally-spaced invalid strings into the column. These strings are created
        by using ``replacement_map`` dictionary that connects the original
        substring with the invalid one that is replaced, making some column values
        not coherent with the others. E.g. 08/03/2020 -> 08-03-2020

        Parameters
        ----------
        column_names : Union[List[str], List[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of errors that are inserted in column.
        replacement_map : Dict[str, str]
            Dictionary where the keys are the substrings that will be replaced
            in some elements of the column. The dictionary values are
            values that will replace those substrings, making the column values not
            coherent with the others.
        """
        super().__init__(column_names=column_names)
        self._error_count = error_count
        self._validate_replacement_map(replacement_map)
        self._replacement_map = replacement_map

    @staticmethod
    def _validate_replacement_map(replacement_map: Dict[str, str]):
        """
        Validate ``replacement_map`` argument used to initialize the instances

        Parameters
        ----------
        replacement_map : Dict[str, str]
            Dictionary that will be validated
        """
        if not (
            all([isinstance(k, str) for k in replacement_map.keys()])
            and all([isinstance(v, str) for v in replacement_map.values()])
        ):
            raise ValueError(
                "The `replacement_map` dictionary contains one (or more) key"
                " and/or value that is not a string"
            )

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Insert invalid substrings into the ``column`` and store the related correct values.

        If the correct substring is an empty string (i.e. the substring
        should not be present), the invalid substring is placed at
        the end of the value, otherwise it replaces the correct value.

        Parameters
        ----------
        column : TestDataSet
            TestDataSet instance containing the columns ``column_names`` used to
            initialize this instance and where some invalid substrings will be inserted.

        Returns
        -------
        TestDataSet
            TestDataSet instance where some invalid substrings were inserted
        """
        for col_name in self._column_names:
            column = dataset[col_name]
            # Create list of sample IDs that will be modified
            invalid_substring_sample_ids = _sample_index_in_list(
                sample_count=self._error_count,
                step=dataset.sample_size // self._error_count,
                # This BIAS prevents the function to insert invalid substrings for
                # the same samples/ids where it may insert other type of errors
                bias=column.col_id + INVALID_SUBSTRING_BIAS,
                list_length=dataset.sample_size,
            )

            substrings_to_insert = list(self._replacement_map.keys())

            for sample_id in invalid_substring_sample_ids:
                substring_to_insert = substrings_to_insert[
                    sample_id % len(substrings_to_insert)
                ]
                correct_substring = self._replacement_map[substring_to_insert]
                # Value that will be modified by inserting a substring
                old_value = column.values_to_fix[sample_id]
                # If the correct substring is an empty string (i.e. the substring
                # should not be present), the invalid substring is placed at
                # the end of the value, otherwise it replaces the correct value
                if correct_substring == "":
                    new_value = str(old_value) + substring_to_insert
                else:
                    new_value = str(column.values_to_fix[sample_id]).replace(
                        correct_substring, substring_to_insert
                    )

                # No need to change column.values_after_fix, because after the correction
                # the dataset should be fully corrected (as if this operation was
                # not performed)
                column.update_sample(
                    sample_id,
                    value_to_fix=new_value,
                    value_after_fix=column.values_after_fix[sample_id],
                )
            dataset[col_name] = column

        return dataset
