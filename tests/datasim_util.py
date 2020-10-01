from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Union

import pandas as pd

from .datasim import ReverseFeatureOperation, TestDataSet, _TestColumn


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


def _insert_substring_by_index(
    original_string: str,
    substring_to_insert: str,
    substr_position_id: int,
) -> str:
    """
    Insert substring into a string in a specific position.

    Parameters
    ----------
    original_string : str
    substring_to_insert : str
    substr_position_id : int
    """
    return (
        original_string[:substr_position_id]
        + substring_to_insert
        + original_string[substr_position_id:]
    )


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


class ChangeColumnDType(ReverseFeatureOperation):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        new_dtype: Union[str, type],
        dtype_after_fix: Union[str, type],
    ):
        """
        Callable class for changing dtype of the columns in a TestDataSet

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns whose dtype will be
            changed. A mix of names and column IDs is not accepted.
        new_dtype : Union[str, type]
            New type that will be set as column dtype for each column in
            dataset named ``column_names``.
        dtype_after_fix : Union[str, type]
            Type of the column that will be found after appropriate correction
            is applied to the column
        """
        super().__init__(column_names)
        self._new_dtype = new_dtype
        self._dtype_after_fix = dtype_after_fix

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Change the dtype of some ``dataset`` columns.

        This method will convert the columns of ``dataset`` to new dtypes,
        according to the parameters which this instance was initialized with.

        Parameters
        ----------
        dataset : TestDataSet
            TestDataSet instance containing the columns ``column_names`` used to
            initialize this instance, whose columns will converted to a new dtype.

        Returns
        -------
        TestDataSet
            TestDataSet instance where the columns have been converted to
            the new dtype.
        """
        for col_name in self._column_names:
            column = dataset[col_name]

            column.update_column_values(
                column.values_to_fix.astype(self._new_dtype),
                column.values_after_fix.astype(self._dtype_after_fix),
            )

            dataset.update_column(col_name, column, self)

        return dataset


class ReplaceSamples(ReverseFeatureOperation, ABC):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        error_count: int,
    ):
        """
        Abstract Class for inserting invalid substrings into column values.

        This class is the abstract class for callables that insert an
        ``error_count`` of invalid substrings into equally-spaced samples of
        the columns named ``column_names``.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of samples that are being modified/replaced.
        """
        super().__init__(column_names)
        self._error_count = error_count

    @abstractmethod
    def replace_samples(
        self, column: _TestColumn, invalid_sample_ids: List[int]
    ) -> _TestColumn:
        """
        Abstract method that replaces some samples in ``column`` object

        This method replaces/modify the sample values with index in
        ``invalid_sample_ids`` list and returns the ``column`` with new sample values.
        This needs to be implemented in subclasses.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being replaced/modified
        invalid_sample_ids : List[int]
            List of indexes of the samples that are being replaced

        Returns
        -------
        _TestColumn
            _TestColumn instance where some values have been replaced/modified
        """
        pass

    def __call__(self, dataset: TestDataSet) -> TestDataSet:
        """
        Insert invalid substrings into the ``column`` and store the related correct values.

        If the correct substring is an empty string (i.e. the substring
        should not be present), the invalid substring is placed at
        the end of the value, otherwise it replaces the correct value.

        Parameters
        ----------
        dataset : TestDataSet
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
            invalid_sample_ids = _sample_index_in_list(
                sample_count=self._error_count,
                step=dataset.sample_size // self._error_count,
                # This BIAS prevents the function to insert invalid substrings for
                # the same samples/ids where it may insert other type of errors
                bias=column.col_id + dataset.last_operation_index,
                list_length=dataset.sample_size,
            )

            dataset.update_column(
                col_name, self.replace_samples(column, invalid_sample_ids), self
            )

        return dataset


class InsertNaNs(ReplaceSamples):
    def __init__(
        self,
        error_count: int,
        column_names: Union[Sequence[str], Sequence[int]],
        nan_value_to_insert: Any,
        nan_value_after_fix: Any,
    ):
        """
        Insert NaN values into the columns ``column_names`` of a TestDataSet instance.

        The function inserts an ``error_count`` number of
        equally-spaced NaN values into a TestDataSet instance.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of values that will be replaced with NaNs in each column
        nan_value : Any
            Value that will be inserted as NaN value
        """
        super().__init__(column_names=column_names, error_count=error_count)
        self._nan_value_to_insert = nan_value_to_insert
        self._nan_value_after_fix = nan_value_after_fix

    def replace_samples(
        self, column: _TestColumn, invalid_sample_ids: List[int]
    ) -> _TestColumn:
        """
        Insert NaN values into ``dataset`` keeping track of the modified samples.

        The function inserts an ``error_count`` number of equally-spaced NaN values
        into the ``column``.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being replaced/modified
        invalid_sample_ids : List[int]
            List of indexes of the samples that are being replaced

        Returns
        -------
        _TestColumn
            _TestColumn instance where some values have been replaced/modified
        """
        for sample_id in invalid_sample_ids:
            column.update_sample(
                sample_id,
                value_to_fix=self._nan_value_to_insert,
                value_after_fix=self._nan_value_after_fix,
            )
        return column


class InsertNewValues(ReplaceSamples):
    def __init__(
        self,
        error_count: int,
        replacement_map: Dict[str, str],
        column_names: Union[Sequence[str], Sequence[int]],
    ):
        """
        Insert invalid values into the column and store the related correct values.

        The function inserts an ``error_count`` number of
        equally-spaced invalid values into the column. These values are create
        by using ``replacement_map`` dictionary that connects the original
        substring with the invalid one that is replaced.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of values that will be replaced with NaNs in each column
        nan_value : Any
            Value that will be inserted as NaN value
        """
        super().__init__(column_names, error_count)
        self._replacement_map = replacement_map

    def replace_samples(
        self, column: _TestColumn, invalid_sample_ids: List[int]
    ) -> _TestColumn:
        """
        Insert invalid values into the ``column`` and store the related correct values.

        The function inserts an ``error_count`` number of
        equally-spaced invalid values into the ``column``. These values are created
        by using ``replacement_map`` dictionary that connects the original
        substring with the invalid one that is replaced.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being replaced/modified
        invalid_sample_ids : List[int]
            List of indexes of the samples that are being replaced

        Returns
        -------
        _TestColumn
            _TestColumn instance where some values have been replaced/modified
        """
        invalid_string_list = list(self._replacement_map.keys())

        for sample_id in invalid_sample_ids:
            error_to_insert = invalid_string_list[sample_id % len(invalid_string_list)]
            column.update_sample(
                sample_id,
                value_to_fix=error_to_insert,
                value_after_fix=self._replacement_map[error_to_insert],
            )

        return column


class InsertOutOfScaleValues(ReplaceSamples):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        error_count: int,
        upperbound_increase: float,
        lowerbound_increase: float,
    ):
        """
        Insert out-of-scale values (with '<' or '>') into the ``column``.

        The class is a callable that inserts an ``error_count`` number of
        equally-spaced out-of-scale values into the _TestColumn instance that
        is passed. The class also stores in the _TestColumn instance the related
        values that will result after appropriate correction.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of invalid values that are inserted in column.
        upperbound_increase : float
            This number indicates the increase ratio of the upper bound
            out-of-scale value, when it will be corrected.
            Particularly, this represents the value U such that:
             '> 80' -> will be corrected as: '> 80 + U * 80'
        lowerbound_increase : float
            This number indicates the increase ratio of the lower bound
            out-of-scale value, when it will be corrected.
            Particularly, this represents the value L such that:
             '< 10' -> will be corrected as: '< 10 + L * 10'
        """
        super().__init__(column_names, error_count)
        self._upperbound_increase = upperbound_increase
        self._lowerbound_increase = lowerbound_increase

    @staticmethod
    def _compute_column_min_max(column: _TestColumn) -> Tuple[float, float]:
        """
        Return the minimum and maximum values in a ``column``.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being evaluated

        Returns
        -------
        float
            Minimum value in ``column``
        float
            Maximum value in ``column``

        Raises
        ------
        TypeError
            If no numeric values are found in the ``column``
        """
        column_series = pd.to_numeric(column.values_to_fix, errors="coerce")
        if column_series.count() == 0:
            raise TypeError(f"No numeric values were found in column {column.name}")
        else:
            min_col_value = column_series.min()
            max_col_value = column_series.max()

        return min_col_value, max_col_value

    def replace_samples(
        self, column: _TestColumn, invalid_sample_ids: List[int]
    ) -> _TestColumn:
        """
        Insert out-of-scale values (with '<' or '>') into the ``column``.

        The method inserts an ``error_count`` number of equally-spaced
        out-of-scale values into the _TestColumn instance that is passed.
        The method also stores in the _TestColumn instance the related
        values that will result after appropriate correction.
        Particularly half of ``error_count`` values will contain
        '< ("minimum value")' and the other half
        '> ("maximum value")', and the related corrected
        values will be "minimum value" - "lowerbound increase" * "minimum value"
        and "maximum value" - "upperbound increase" * "maximum value",
        respectively.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being replaced/modified
        invalid_sample_ids : List[int]
            List of indexes of the samples that are being replaced

        Returns
        -------
        _TestColumn
            _TestColumn instance where some values have been replaced/modified
        """
        min_col_value, max_col_value = self._compute_column_min_max(column)

        for sample_id in invalid_sample_ids:
            if sample_id % 2:
                new_value = "<" + str(min_col_value)
                value_after_fix = min_col_value * (1 - self._lowerbound_increase)
            else:
                new_value = ">" + str(max_col_value)
                value_after_fix = max_col_value * (1 + self._upperbound_increase)
            column.update_sample(
                sample_id,
                value_to_fix=new_value,
                value_after_fix=value_after_fix,
            )

        return column


class SubstringReplacementMap(ABC):
    def __init__(
        self, new_substring: str, value_after_fix: Any, fix_with_substring: bool
    ) -> None:
        """
        Abstract Class describing how a substring must be replaced in generic values.

        This class is a map that can describe where and how a substring must be
        replaced in string values, meanly the ``new_substring`` to insert/replace
        and the one that is supposed to be found after appropriate correction
        of the inserted new substring.

        Parameters
        ----------
        new_substring : str
            The substring that will be used to modify the original sample value.
        value_after_fix : Any
            The value that is supposed to be the correction of the
            ``new_substring`` value. Depending on the operation using this object,
            it can be a substring so that the correct sample value will be obtained
            by replacing the ``new_substring`` with this substring.
            It may also be a generic value and the correct sample value will be
            equal to this value.
        fix_with_substring : bool
            It describes if the value, that replaces the one where ``new_substring``
            was inserted, corresponds to ``value_after_fix`` only (when   set to
            True), or to the new value where ``value_after_fix`` replaces
            the newly inserted ``new_substring`` (when set to False).
        """
        self._superclass_validate(new_substring, fix_with_substring)

        self.new_substring = new_substring
        self.value_after_fix = value_after_fix
        self.fix_with_substring = fix_with_substring

    @staticmethod
    def _superclass_validate(new_substring: str, fix_with_substring: bool):
        """
        Validate the arguments used to initialize the SubstringReplacementMap class

        This method checks that ``new_substring`` is a string and that
        ``fix_with_substring`` is a boolean.

        Parameters
        ----------
        new_substring : str
            The substring that will be used to modify the original sample value.
        fix_with_substring : bool
            It describes how the value where the substring is replaced should
            result after appropriate correction.

        Raises
        ------
        ValueError
            If ``new_substring`` argument is not a string value or if
            ``fix_with_substring`` argument is not boolean
        """
        if not isinstance(new_substring, str):
            raise ValueError(
                f"The `new_substring` argument is {new_substring}"
                f" and it is not a string, but it is {type(new_substring)}"
            )
        if not isinstance(fix_with_substring, bool):
            raise ValueError(
                f"The `fix_with_substring` argument is {fix_with_substring}"
                f" and it is not boolean, but it is {type(fix_with_substring)}"
            )


class SubstringReplaceMapByValue(SubstringReplacementMap):
    def __init__(
        self,
        substr_to_replace: str,
        new_substring: str,
        value_after_fix: Any,
        fix_with_substring: bool,
    ) -> None:
        """
        Class describing how a substring must be replaced in string values.

        This class is a map that connects where and how a substring must be
        replaced in string values. Particularly it relates which substring must
        be replaced, the ``new_substring`` to replace and the one that is
        supposed to be found after appropriate correction of the newly replaced
        substring.

        Parameters
        ----------
        substr_to_replace : str
            Substring that specifies which part of the sample value must be replaced
            by the ``new_substring``. If this substring is not present in the sample
            value or it is an empty string (i.e. ''), ``new_substring`` will be
            added at the end of the sample value.
        new_substring : str
            Substring that will be used to modify the original sample value.
        value_after_fix : Any
            The value that is supposed to be the correction of the
            ``new_substring`` value. Depending on the operation using this object,
            it can be a substring so that the correct sample value will be obtained
            by replacing the ``new_substring`` with this substring.
            It may also be a generic value and the correct sample value will be
            equal to this value.
        fix_with_substring : bool
            If True the sample value after correction must be the
            original value where the inserted substring is replaced by the correct
            substring (specified by the third element of each tuple in
            ``replacement_map``). If False, the sample value after the correction
            must be a generic value (also specified by the third element of
            each tuple in ``replacement_map``).

        Raises
        ------
        ValueError
            If ``substr_to_replace`` argument is not a string value or if
            ``new_substring`` argument is not a string value
        """
        self._subclass_validate(substr_to_replace)

        super().__init__(new_substring, value_after_fix, fix_with_substring)
        self.substr_to_replace = substr_to_replace

    @staticmethod
    def _subclass_validate(substr_to_replace: str):
        """
        Validate the ``substr_to_replace`` argument used to initialize the instance

        This method checks that ``substr_to_replace`` is string

        Parameters
        ----------
        substr_to_replace : str
            Argument that needs to be validated

        Raises
        ------
        ValueError
            If the ``substr_to_replace`` argument is not a string value
        """
        if not isinstance(substr_to_replace, str):
            raise ValueError(
                f"The `substr_to_replace` argument is {substr_to_replace}"
                f" and it is not a string, but it is {type(substr_to_replace)}"
            )


class SubstringReplaceMapByIndex(SubstringReplacementMap):
    def __init__(
        self,
        substr_position_id: int,
        new_substring: str,
        value_after_fix: Any,
        fix_with_substring: bool,
    ) -> None:
        """
        Class describing how a substring must be inserted in string values.

        This class is a map that connects where and how a substring must be
        replaced in string values. Particularly it connects the ``new_substring``
        to insert, the position where it should be inserted in the original
        string and the substring that is supposed to be found after appropriate
        correction of the newly inserted substring.

        Parameters
        ----------
        substr_position_id : str
            Int that specifies where the ``new_substring`` must be inserted in
            the sample value.
        new_substring : str
            Substring that will be used to modify the original sample value.
        value_after_fix : Any
            The value that is supposed to be the correction of the
            ``new_substring`` value. Depending on the operation using this object,
            it can be a substring so that the correct sample value will be obtained
            by replacing the ``new_substring`` with this substring.
            It may also be a generic value and the correct sample value will be
            equal to this value.
        fix_with_substring : bool
            If True the sample value after correction must be the
            original value where the inserted substring is replaced by the correct
            substring (specified by the third element of each tuple in
            ``replacement_map``). If False, the sample value after the correction
            must be a generic value (also specified by the third element of
            each tuple in ``replacement_map``).

        Raises
        ------
        ValueError
            If the ``substr_position_id`` argument is not an integer value or if
            ``new_substring`` argument is not a string value
        """
        self._subclass_validate(substr_position_id)

        super().__init__(new_substring, value_after_fix, fix_with_substring)
        self.substr_position_id = substr_position_id

    @staticmethod
    def _subclass_validate(substr_position_id: int):
        """
        Validate the ``substr_position_id`` argument used to initialize the instance

        This method checks that ``substr_position_id`` is integer

        Parameters
        ----------
        substr_position_id : int
            Argument that needs to be validated

        Raises
        ------
        ValueError
            If the ``substr_position_id`` argument is not an integer value
        """
        if not isinstance(substr_position_id, int):
            raise ValueError(
                f"The `where_insert_substring` argument is {substr_position_id}"
                f" and it is not a int, but it is {type(substr_position_id)}"
            )


class TransformSubstrings(ReplaceSamples, ABC):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        error_count: int,
        replacement_map_list: List[SubstringReplacementMap],
    ):
        """
        Abstract Class for inserting invalid substrings into column values.

        This class is the abstract class for callables that insert an
        ``error_count`` of invalid substrings into equally-spaced samples of
        the columns named ``column_names``.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of errors that are inserted in column.
        replacement_map_list : List[SubstringReplaceMapByIndex]
            SubstringReplaceMapByIndex instance that specifies the substrings to
            insert, the positions where they should be inserted and the
            corresponding corrections
        """
        super().__init__(column_names, error_count)
        self._replacement_map_list = replacement_map_list

    @abstractmethod
    def replace_value(
        self, previous_value: str, replacement_map: SubstringReplacementMap
    ) -> Tuple:
        """
        Abstract method that replaces a substring in the sample value ``previous_value``

        This needs to be implemented in subclasses.

        Parameters
        ----------
        previous_value : str
            Sample value that needs to be replaced
        replacement_map : SubstringReplacementMap
            SubstringReplacementMap instance that describes where the substring must
            be inserted/replaced, the new substring and the value after fix.

        Returns
        -------
        str
            New value that replaces the sample ``previous_value``.
        Any
            Value that will replace the new value after it being fixed by proper
            correction.
        """
        pass

    def replace_samples(
        self, column: _TestColumn, invalid_sample_ids: List[int]
    ) -> _TestColumn:
        """
        Method that replaces some samples in ``column`` object

        This method replaces/modify the sample values with index in
        ``invalid_sample_ids`` list and returns the ``column`` with new sample values.
        This needs to be implemented in subclasses.

        Parameters
        ----------
        column : _TestColumn
            _TestColumn instance whose values are being replaced/modified
        invalid_sample_ids : List[int]
            List of indexes of the samples that are being replaced

        Returns
        -------
        _TestColumn
            _TestColumn instance where some values have been replaced/modified
        """
        for sample_id in invalid_sample_ids:
            replacement_map = self._replacement_map_list[
                sample_id % len(self._replacement_map_list)
            ]

            new_value, value_after_fix = self.replace_value(
                previous_value=str(column.values_to_fix[sample_id]),
                replacement_map=replacement_map,
            )

            column.update_sample(
                sample_id,
                value_to_fix=new_value,
                value_after_fix=value_after_fix,
            )

        return column


class InsertSubstringsByIndex(TransformSubstrings):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        error_count: int,
        replacement_map_list: List[SubstringReplaceMapByIndex],
    ):
        """
        Insert invalid substrings into the column and store the related correct values.

        The function inserts an ``error_count`` of invalid substrings into
        equally-spaced samples of the columns named ``column_names``.
        The function introduces a new substring inside the sample values in a
        specific position without overwriting any other part of the value.
        These substrings, the positions where they should be inserted and the
        corresponding corrections, are specified through the
        SubstringReplaceMapByIndex instance, i.e. ``replacement_map_list`` argument.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of errors that are inserted in column.
        replacement_map_list : List[SubstringReplaceMapByIndex]
            SubstringReplaceMapByIndex instance that specifies the substrings to
            insert, the positions where they should be inserted and the
            corresponding corrections
        """
        super().__init__(column_names, error_count, replacement_map_list)

    def replace_value(
        self, previous_value: str, replacement_map: SubstringReplaceMapByIndex
    ) -> Tuple:
        """
        Replace substring in ``previous_value`` according to ``replacement_map``

        Implementation of the method of the abstract class, called when the instance
        is called to transform some samples of a TestDataSet.
        The method replaces a substring into sample ``previous_value`` according
        to index specified in ``replacement_map`` argument.

        Parameters
        ----------
        previous_value : str
            Sample value that needs to be replaced
        replacement_map : SubstringReplacementMap
            SubstringReplacementMap instance that describes where the substring must
            be inserted, the new substring and the value after fix.

        Returns
        -------
        str
            New value that replaces the sample ``previous_value``.
        Any
            Value that will replace the new value after it being fixed by proper
            correction.
        """
        # Value that will be modified by inserting a substring
        substr_position_id = replacement_map.substr_position_id
        # Replacing the new substring in the related position
        new_value = _insert_substring_by_index(
            previous_value,
            replacement_map.new_substring,
            substr_position_id,
        )
        if replacement_map.fix_with_substring:
            value_after_fix = _insert_substring_by_index(
                previous_value,
                replacement_map.value_after_fix,
                substr_position_id,
            )
        else:
            value_after_fix = replacement_map.value_after_fix

        return new_value, value_after_fix


class ReplaceSubstringsByValue(TransformSubstrings):
    def __init__(
        self,
        column_names: Union[Sequence[str], Sequence[int]],
        error_count: int,
        replacement_map_list: List[SubstringReplaceMapByValue],
    ):
        """
        Replace substrings in ``column_names`` with invalid ones.

        This class is a callable that inserts invalid substrings into
        ``error_count`` equally-spaced samples of the columns named ``column_names``.
        Particularly it replaces a new substring inside the sample values with
        another one. These substrings, the ones that are replaced and the
        corresponding corrections, are specified through the
        SubstringReplaceMapByIndex instances, i.e. ``replacement_map_list`` argument
        Since each instance corresponds to a different substring, if
        ``error_count`` exceeds the ``replacement_map_list``, the
        ``replacement_map_list`` will be treated as a circular list, in order to
        repeat different errors in the same column.

        Parameters
        ----------
        column_names : Union[Sequence[str], Sequence[int]]
            List of the names/column IDs of the columns on which the
            ReverseFeatureOperation is applied. A mix of names and column IDs
            is not accepted.
        error_count : int
            Number of errors that are inserted in column.
        replacement_map_list : List[SubstringReplaceMapByIndex]
            SubstringReplaceMapByIndex instance that specifies the new substrings to
            insert, the ones that must be replaced and the expected value after
            an appropriate correction.
        """
        super().__init__(column_names, error_count, replacement_map_list)

    def replace_value(
        self, previous_value: str, replacement_map: SubstringReplaceMapByValue
    ) -> Tuple[str, Any]:
        """
        Replace substring in ``previous_value`` according to ``replacement_map``

        Implementation of the method of the abstract class that describes
        the behavior of the class when it replaces values (by calling it).
        The method replaces the sample ``previous_value`` with the new substring
        present after the appropriate correction.

        Parameters
        ----------
        previous_value : str
            Sample value that needs to be replaced
        replacement_map : SubstringReplacementMap
            SubstringReplacementMap instance that describes where the substring must
            be inserted, the new substring and the value after fix.

        Returns
        -------
        str
            New value that replaces the sample ``previous_value``.
        Any
            Value that will replace the new value after it being fixed by proper
            correction.
        """
        # Value that will be modified by inserting a substring
        substr_to_replace = replacement_map.substr_to_replace
        # Replacing the new substring in the related position
        new_value = previous_value.replace(
            substr_to_replace, replacement_map.new_substring
        )
        if replacement_map.fix_with_substring:
            value_after_fix = previous_value.replace(
                substr_to_replace, replacement_map.value_after_fix
            )
        else:
            value_after_fix = replacement_map.value_after_fix

        return new_value, value_after_fix


#         Examples
#         --------
#         # TODO: Do two examples with the new replacementmap structure (one with fix_with_substring=True and one = False)
#         Replacing substrings with substrings:
#         >>> replacement_map = {'.': ',', '%':'', '&':''}  "value_to_replace (like '.')": "replaced value (invalid string, ',')": "correct string after fix ('.')"
#         "value_to_replace (like '')": "replaced value (invalid string, '%')": "correct string after fix ('pd.NA')"
#         "value_to_replace (like '/')": "replaced value (invalid string, '-')": "correct string after fix ('/')"
#         "value_to_replace (like '')": "replaced value (invalid string, '&')": "correct string after fix ('pd.NA')"
#         >>> df = pd.DataFrame({'col_0':[1.1, 1.2], 'col_1':[1-1-03, 3-12-97]})
#         >>> td = TestDataSet()
#         >>> TransformSubstrings(column_names=['col_0', 'col_1'], error_count=1, replacement_map = []
#         and we want to apply this operation to a column containing a value like:
#         >>> original_value = 8.2
#         Then the invalid value created, and its correction will be respectively:
#         >>> invalid_value = 8.2%
#         >>> value_after_fix = pd.NA
