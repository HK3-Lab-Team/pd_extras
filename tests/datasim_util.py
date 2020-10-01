from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .datasim import TestDataSet, _TestColumn

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
            invalid_sample_ids = _sample_index_in_list(
                sample_count=self._error_count,
                step=dataset.sample_size // self._error_count,
                # This BIAS prevents the function to insert invalid substrings for
                # the same samples/ids where it may insert other type of errors
                bias=column.col_id + INVALID_SUBSTRING_BIAS,
                list_length=dataset.sample_size,
            )

            dataset[col_name] = self.replace_samples(column, invalid_sample_ids)

        return dataset


class InsertNaNs(ReplaceSamples):
    def __init__(
        self, error_count: int, column_names: Union[Sequence[str], Sequence[int]]
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
        """
        super().__init__(column_names=column_names, error_count=error_count)

    @staticmethod
    def replace_samples(
        column: _TestColumn, invalid_sample_ids: List[int]
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
            column.update_sample(sample_id, value_to_fix=pd.NA, value_after_fix=pd.NA)
        return column


class InsertInvalidValues(ReplaceSamples):
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
            Number of invalid values that are inserted in column.
        replacement_map : Dict[str, str]
            Dictionary where the keys are the invalid string that will replace
            some elements of ``values`` argument. The dictionary values are the
            values that will replace those invalid values in the `correct_column`
            property.
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

             , If True, after appro the sample value after correction must be the
            original value where the inserted substring is replaced by the correct
            substring (specified by the third element of each tuple in
            ``replacement_map``). If False, the sample value after the correction
            must be a generic value (also specified by the third element of
            each tuple in ``replacement_map``).
        """
        self._validate(new_substring, fix_with_substring)
        self.new_substring = new_substring
        self.value_after_fix = value_after_fix
        self.fix_with_substring = fix_with_substring

    @staticmethod
    def _validate(new_substring: str, fix_with_substring: bool):
        """
        Validate the arguments used to initialize the instance

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
        super().__init__(new_substring, value_after_fix, fix_with_substring)
        self._validate(substr_to_replace)
        self.substr_to_replace = substr_to_replace

    @staticmethod
    def _validate(substr_to_replace: str):
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
        super().__init__(new_substring, value_after_fix, fix_with_substring)
        self._validate(substr_position_id)
        self.substr_position_id = substr_position_id

    @staticmethod
    def _validate(substr_position_id: int):
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
        from ``replacement_map`` argument and it stores the new value that will be
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
