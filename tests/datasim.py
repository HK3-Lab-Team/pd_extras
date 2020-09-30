from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class ReverseFeatureOperation(ABC):
    def __init__(self, column_names: Union[Sequence[str], Sequence[int]]):
        """
        Abstract Class that revert preprocessing operations on TestDataSets

        Its subclasses apply operations to simulated synthetic data that revert
        the behaviour of the FeatureOperation classes.
        This is an abstract class so the abstract method "apply" needs to be reimplemented
        in subclasses in order to work.

        Parameters
        ----------
        columns : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        """
        self._validate_column_names(column_names)
        self._column_names = column_names

    @staticmethod
    def _validate_column_names(column_names: Union[Sequence[str], Sequence[int]]):
        """
        Check if ``column_names`` does not contain a mix of strings and integers

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
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

    def __call__(self, dataset: "TestDataSet") -> "TestDataSet":
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
    def __init__(self, column: TestColumn, col_id: int):
        """
        Private Class handling column values to fix and the ones modified after fix.

        This class uses ``column`` to initialize the values of two pandas Series
        that will be contemporarily modified by the applied
        ReverseFeatureOperation:
        - ``_values_to_fix`` -> keeps track of the modified values
        - ``_values_after_fix`` -> keeps track of the values that are
            expected to be found when the user will apply the appropriate
            correction (e.g. result of a test)
        In fact, since the process of properly correcting ``values_to_fix`` cannot
        always be fully reverted (e.g. inserting NaN), each ReverseFeatureOperation
        takes care of providing the modification and the expected correction by
        modifying ``values_to_fix`` and ``values_after_fix``.

        Parameters
        ----------
        column : TestColumn
            TestColumn instance whose values are used to create the instance
        col_id : int
            Integer identifying the index of the column considered. This will be
            used by ReverseFeatureOperation instances in order to avoid replacing
            the values to be fixed in the same samples.
            Otherwise we may end up having samples with rows full of invalid values.

        """
        self._name = column.name
        self._col_id = col_id

        self._values_to_fix = pd.Series(column.original_values, dtype=column.dtype)
        self._values_after_fix = self._values_to_fix.copy()

    @property
    def name(self):
        """
        Return the column name
        """
        return self._name

    @property
    def col_id(self):
        """
        Return the column index
        """
        return self._col_id

    def __len__(self) -> int:
        """
        Return the column sample size
        """
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
        self,
        values_to_fix: pd.Series,
        values_after_fix: pd.Series,
    ):
        """
        Update a single column sample with new values

        Parameters
        ----------
        values_to_fix : pandas.Series
            Raw values that need a fix (usually values containing errors that have
            been simulated by a ReverseFeatureOperation).
        values_to_fix : pandas.Series
            Values that are supposed to be found after that the proper correction
            is applied to ``values_to_fix`` argument.
        """
        self._values_to_fix = pd.Series(values_to_fix, dtype=self._dtype)
        self._values_after_fix = pd.Series(values_after_fix, dtype=self._dtype)


class TestDataSet:
    def __init__(self, sample_size: Optional[int] = None):
        """
        Class to store two datasets with simulated errors and with related corrections.

        This class creates a DataSet by adding columns to it and it is used to
        add simulated errors to the original values.
        This class also tracks the operations performed on the dataset and
        the values that are supposed to be created after the appropriate steps
        for the correction of the DataSet.

        Parameters
        ----------
        sample_size: Optional[int]
            Size of the samples that each dataset column will contain. If None,
            this value will be computed based on the sample size of the first
            added column. Default set to None
        """
        self._columns_by_index = []
        self._sample_size = sample_size
        self._name_to_index_map = {}
        self._operation_history = []

    @property
    def sample_size(self) -> Optional[int]:
        """
        Return the number of samples of the dataset
        """
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
        Add a new single TestColumn instance ``column`` to the instance

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

    def update_column(
        self,
        column_name_id: Union[int, str],
        new_value: _TestColumn,
        operation_used: ReverseFeatureOperation,
    ) -> None:
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
        self._validate_testcolumn_to_update(new_value)

        # Record the new operation used
        self._operation_history.append(operation_used)

        if isinstance(column_name_id, int):
            self._columns_by_index[column_name_id] = new_value
        else:
            self._columns_by_index[self._name_to_index_map[column_name_id]] = new_value

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

    @property
    def last_operation_index(self) -> int:
        """
        Return the index of the last operation performed on the instance.

        This property may be called from "ReplaceSamples" instances
        that uses this as a bias for computing the indexes of the samples that will
        be replaced/modified. This allows every different operation
        to modify a different set of samples (to prevent having only few
        samples full of errors).

        Returns
        -------
        int
            Index of the last operation performed on the instance.
        """
        return len(self._operation_history)

    def __getitem__(self, item: Union[int, str]) -> _TestColumn:
        if isinstance(item, int):
            return self._columns_by_index[item]
        else:
            return self._columns_by_index[self._name_to_index_map[str(item)]]

    def __contains__(self, column_name_id: Union[int, str]) -> bool:
        """
        Check if the column identified by ``column_name_id`` is present in TestDataSet

        Parameters
        ----------
        column_name_id : Union[int, str]
            Name or ID of the column whose presence needs to be checked.

        Returns
        -------
        bool
            Returns True if the column identified by ``column_name_id`` is present
            in TestDataSet, False otherwise.
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

    def shape(self) -> Tuple[Optional[int], int]:
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


def from_pandas(df: pd.DataFrame) -> TestDataSet:
    test_dataset = TestDataSet()
    for col_name in df.columns:
        df_column = df[col_name]
        test_dataset.add_column(
            TestColumn(
                name=col_name, original_values=df_column.values, dtype=df_column.dtype
            )
        )
    return test_dataset


def from_tuples(tuple_list: List[Tuple[str, List, Optional[str]]]) -> TestDataSet:
    """
    Create TestDataSet starting from a list of tuples

    Each tuple of the list corresponds to a TestColumn that is created and added
    to TestDataSet. Therefore, it must contain essential infos for the column:
    1. Name
    2. Original Values
    3. Dtype of the column (this is optional)
    """
    test_dataset = TestDataSet()
    for col in tuple_list:
        test_dataset.add_column(
            TestColumn(
                name=col[0],
                original_values=col[1],
                dtype=(None if len(col) < 3 else col[2]),
            )
        )

    return test_dataset
