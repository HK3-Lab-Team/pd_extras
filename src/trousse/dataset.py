import collections
import copy
import dbm
import logging
import os
import shelve
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple, Union

import pandas as pd
import sklearn
from joblib import Parallel, delayed

from .exceptions import (
    MultipleObjectsInFileError,
    MultipleOperationsFoundError,
    NotShelveFileError,
)
from .feature_enum import OperationTypeEnum
from .feature_operation import FeatureOperation
from .settings import CATEG_COL_THRESHOLD

logger = logging.getLogger(__name__)


def get_df_from_csv(df_filename: str) -> pd.DataFrame:
    """
    Read csv file ``df_filename`` and return pandas DataFrame

    Parameters
    ----------
    df_filename: str
        Path to csv file that contains data for pandas DataFrame

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing data from csv file ``df_filename``

    """
    try:
        df = pd.read_csv(df_filename)
        logger.info("Data imported from file successfully")
        return df
    except FileNotFoundError as e:
        logger.error(e)
        return None


_COL_NAME_COLUMN = "col_name"
_COL_TYPE_COLUMN = "col_type"


def _find_samples_by_type(df: pd.DataFrame, samples_type: str) -> Set[str]:
    """
    Return names of the sample that share the same value ``samples_type``

    This function accepts a pandas DataFrame ``df`` with two columns: "col_name",
    "col_type". Then it finds the rows with "col_type" value == ``samples_type`` and
    returns a set with the "col_name" values of those rows.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame with 2 columns: "col_name" and "col_type"
    samples_type: str
        Value of "col_type" column of ``df``. The "col_name" of the samples with
        this value will be included in the returned set

    Returns
    -------
    set
        Set with "col_name" value of the rows that have "col_type"
        value == ``samples_type``
    """
    return set(df[df[_COL_TYPE_COLUMN] == samples_type][_COL_NAME_COLUMN].values)


def _find_single_column_type(df_col: pd.Series) -> Dict[str, str]:
    """
    Analyze the ``df_col`` to find the type of its values.

    After computing the type of each value, the function compares them to check if they
    are different (in this case the column type is "mixed_type_col"). If not, the
    based on the type of the column first element, the function returns a column type
    as follows:
    - float/int -> "numerical_col"
    - bool -> "bool_col"
    - str -> "string_col"
    - other types -> "other_col"

    Parameters
    ----------
    df_col: pd.Series
        Pandas Series, i.e. column, that the function will analyze and assign a type to

    Returns
    -------
    Dict
        Dictionary with:
        - "col_name": Name of the column analyzed
        - "col_type": Type identified for the column
    """
    col = df_col.name
    # Select not-NaN only
    notna_column = df_col[df_col.notna()]
    # Compare the first_row type with every other row of the same column
    col_types = notna_column.apply(lambda r: str(type(r))).values
    has_same_types = all(col_types == col_types[0])
    if has_same_types:
        # Check the type of the first element
        col_type = col_types[0]
        if "bool" in col_type or set(notna_column) == {0, 1}:
            return {_COL_NAME_COLUMN: col, _COL_TYPE_COLUMN: "bool_col"}
        elif "str" in col_type:
            # String columns
            return {_COL_NAME_COLUMN: col, _COL_TYPE_COLUMN: "string_col"}
        elif "float" in col_type or "int" in col_type:
            # look if the col_type contains 'int' or 'float' keywords
            return {_COL_NAME_COLUMN: col, _COL_TYPE_COLUMN: "numerical_col"}
        else:
            return {_COL_NAME_COLUMN: col, _COL_TYPE_COLUMN: "other_col"}
    else:
        return {_COL_NAME_COLUMN: col, _COL_TYPE_COLUMN: "mixed_type_col"}


def _split_columns_by_type_parallel(
    df: pd.DataFrame, col_list: List[str]
) -> Tuple[Set[str]]:
    """
    Find column types of DataFrame ``df``

    For performance reasons, joblib is employed to analyze columns in parallel
    processes.
    For each column, after computing each value type, the function compares
    them to check if they are different (in this case the column type is
    "mixed_type_col"). If not, the based on the type of the column first element,
    the function returns a column type as follows:
    - float/int -> "numerical_col"
    - bool -> "bool_col"
    - str -> "string_col"
    - other types -> "other_col"

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing columns to be analyzed
    col_list: List[str]
        List of the column names that will be analyzed

    Returns
    -------
    set:
        Set of columns containing values of different types
    set:
        Set of columns containing numerical values
    set:
        Set of columns containing string values
    set:
        Set of columns containing boolean values
    set:
        Set of the remaining columns that are in ``col_list`` but do not belong to any
        other set
    """
    column_type_dict = Parallel(n_jobs=-1)(
        delayed(_find_single_column_type)(df_col=df[col]) for col in col_list
    )
    column_type_df = pd.DataFrame(column_type_dict)

    mixed_type_cols = _find_samples_by_type(column_type_df, "mixed_type_col")
    numerical_cols = _find_samples_by_type(column_type_df, "numerical_col")
    str_cols = _find_samples_by_type(column_type_df, "string_col")
    other_cols = _find_samples_by_type(column_type_df, "other_col")
    bool_cols = _find_samples_by_type(column_type_df, "bool_col")

    return mixed_type_cols, numerical_cols, str_cols, bool_cols, other_cols


@dataclass
class ColumnListByType:
    """
    This dataclass is to gather the different column types inside a pd.DataFrame.
    The columns are split according to the type of their values.
    """

    constant_cols: Set
    mixed_type_cols: Set
    numerical_cols: Set
    med_exam_col_list: Set
    str_cols: Set
    str_categorical_cols: Set
    num_categorical_cols: Set
    other_cols: Set
    bool_cols: Set

    def __str__(self):
        return (
            f"Columns with:"
            f"\n\t1.\tMixed types: \t\t{len(self.mixed_type_cols)}"
            f"\n\t2.\tNumerical types (float/int): \t{len(self.numerical_cols)}"
            f"\n\t3.\tString types: \t\t{len(self.str_cols)}"
            f"\n\t4.\tBool types: \t\t{len(self.bool_cols)}"
            f"\n\t5.\tOther types: \t\t{len(self.other_cols)}"
            f"\nAmong these categories:"
            f"\n\t1.\tString categorical columns: {len(self.str_categorical_cols)}"
            f"\n\t2.\tNumeric categorical columns: {len(self.num_categorical_cols)}"
            f"\n\t3.\tMedical Exam columns (numerical, no metadata): {len(self.med_exam_col_list)}"
            f"\n\t4.\tOne repeated value: {len(self.constant_cols)}"
        )


class Dataset:
    def __init__(
        self,
        metadata_cols: Tuple = (),
        feature_cols: Tuple = None,
        data_file: str = None,
        df_object: pd.DataFrame = None,
        nan_percentage_threshold: float = 0.999,
        new_columns_encoding_maps: Union[
            DefaultDict[str, List[FeatureOperation]], None
        ] = None,
    ):
        """
        Class containing useful methods and attributes related to the Dataset.

        It also keeps track of the operations performed on DataFrame, and returns
        subgroups of columns split by type.

        Parameters
        ----------
        metadata_cols: Tuple[str], optional
            Tuple with the name of the columns that have metadata information related to
            the sample.
            Default set to ()
        feature_cols: Tuple[str], optional
            Tuple with the name of the columns that contains sample features.
            Default is None, meaning that all the columns but the ``metadata_cols`` will be
            considered as features.
        data_file: str, optional
            Path to the csv file containing data. Either this or ``df_object`` must be
            provided. In case ``df_object`` is provided, this will not be considered.
            Default set to None.
        df_object: pd.DataFrame, optional
            Pandas DataFrame instance containing the data. Either this or data_file
            must be provided. In case ``data_file`` is provided, only this will
            be considered as data. Default set to None.
        nan_percentage_threshold: float, optional
            Float value in the range [0,1] describing the threshold of
            NaN values count to decide if the column is relevant or not.
        new_columns_encoding_maps: Union[
            DefaultDict[str, List[FeatureOperation]], None
        ], optional
            Dict where the keys are the column name and the values are the related
            operations that created the column or that were performed on them.
            This is to keep track of the operations performed on dataframe features.
        """
        if df_object is None:
            if data_file is None:
                logging.error("Provide either data_file or df_object as argument")
            else:
                self.df = get_df_from_csv(data_file)
        else:
            self.df = df_object

        self._metadata_cols = set(metadata_cols)
        if feature_cols is None:
            self._feature_cols = set(self.df.columns) - self.metadata_cols
        else:
            self._feature_cols = set(feature_cols)
        self.nan_percentage_threshold = nan_percentage_threshold

        # Dict of Lists ->
        #         key: column_name,
        #         value: List of FeatureOperation instances
        if new_columns_encoding_maps is None:
            # Dict already initialized to lists for every "column_name"
            new_columns_encoding_maps = collections.defaultdict(list)
        self.feature_elaborations = new_columns_encoding_maps

        # Columns generated by Feature Refactoring (e.g.: encoding, bin_splitting)
        self.derived_columns = set()

    # =====================
    # =    PROPERTIES     =
    # =====================

    @property
    def metadata_cols(self) -> Set[str]:
        """Return columns representing metadata

        Returns
        -------
        Set[str]
            Metadata columns
        """
        return self._metadata_cols

    @property
    def feature_cols(self) -> Set[str]:
        """Return columns representing features

        Returns
        -------
        Set[str]
            Feature columns
        """
        return self._feature_cols

    @property
    def many_nan_columns(self) -> Set[str]:
        """
        Return name of the columns containing many NaN.

        This property selects the columns where the ratio of NaN values over the
        sample count is higher than ``nan_percentage_threshold`` attribute
        (in range [0,1]).

        Returns
        -------
        Set[str]
            Set of column names with NaN ratio higher than
            ``nan_percentage_threshold`` attribute.
        """
        many_nan_columns = set()
        for c in self.feature_cols:
            # Check number of NaN
            if (
                sum(self.df[c].isna())
                > self.nan_percentage_threshold * self.df.shape[0]
            ):
                many_nan_columns.add(c)

        return many_nan_columns

    @property
    def constant_cols(self) -> Set[str]:
        """Return name of the columns containing only one repeated value.

        Returns
        -------
        Set[str]
            Set of column names with only one repeated value
        """
        df_nunique = self.df[self.feature_cols].nunique(dropna=False)
        constant_cols = df_nunique[df_nunique == 1].index
        return set(constant_cols)

    @property
    def trivial_columns(self) -> Set[str]:
        """
        Return name of the columns containing many NaN or only one repeated value.

        This function return the name of the column that were included in
        ``constant_cols`` or ``many_nan_cols`` attributes

        Returns
        -------
        Set[str]
            Set containing the name of the columns with many NaNs or with only
            one repeated value
        """
        return self.many_nan_columns.union(self.constant_cols)

    @property
    def column_list_by_type(self) -> ColumnListByType:
        """
        Analyze the instance and return an object with the column list split by type.

        NOTE: This gathers many properties/column_types together and
        returns an object containing them because calculating them together is much
        more efficient when we need two (or more) of them (and we do not waste much
        time if we only need one column type).

        Returns
        -------
        ColumnListByType
            ColumnListByType instance containing the column list split by type
        """
        constant_cols = self.constant_cols

        # TODO: Exclude NaN columns (self.nan_cols) from `col_list` too (so they will
        #  not be included in num_categorical_cols just for one not-Nan value)

        col_list = self.feature_cols - constant_cols

        (
            mixed_type_cols,
            numerical_cols,
            str_cols,
            bool_cols,
            other_cols,
        ) = _split_columns_by_type_parallel(self.df, col_list)

        str_categorical_cols = self._get_categorical_cols(str_cols)
        num_categorical_cols = self._get_categorical_cols(numerical_cols)
        med_exam_col_list = (
            numerical_cols | bool_cols - constant_cols - self.metadata_cols
        )

        return ColumnListByType(
            mixed_type_cols=mixed_type_cols,
            constant_cols=constant_cols,
            numerical_cols=numerical_cols | bool_cols,  # TODO: Remove bool_cols
            med_exam_col_list=med_exam_col_list,
            str_cols=str_cols,
            str_categorical_cols=str_categorical_cols,
            num_categorical_cols=num_categorical_cols,
            bool_cols=bool_cols,
            other_cols=other_cols,
        )

    def _get_categorical_cols(self, col_list: Tuple[str]) -> Set[str]:
        """
        Identify every categorical column in dataset.

        It will also set those column's types to "category".
        To avoid considering every string column as categorical, it selects the
        columns with few unique values. Therefore:
            1. If ``df`` attribute contains few samples (less than 50), it is
                reasonable to expect less than 7 values repeated for the column to
                be considered as categorical.
            2. If ``df`` attribute contains many samples, it is
                reasonable to expect more than 7 possible values in a categorical
                column (variability increases). So the method will recognize the
                column as categorical if the unique values are less than
                `number of values` (excluding NaNs) // ``CATEG_COL_THRESHOLD``.
                ``CATEG_COL_THRESHOLD`` is a parameter defined in `settings.py` that
                corresponds to the minimum number of expected samples with the same
                repeated value on average
                (E.g. CATEG_COL_THRESHOLD = 300 -> We expect more than 300 samples
                with the same value on average)


        Parameters
        ----------
        col_list: Tuple[str]
            Tuple of the name of the columns that will be analyzed

        Returns
        -------
        Set[str]
            Set of categorical columns
        """
        categorical_cols = set()

        for col in col_list:
            unique_val_nb = len(self.df[col].unique())
            if unique_val_nb < 7 or (
                unique_val_nb < self.df[col].count() // CATEG_COL_THRESHOLD
            ):
                self.df[col] = self.df[col].astype("category")
                categorical_cols.add(col)

        return categorical_cols

    @property
    def to_be_fixed_cols(self) -> Set[str]:
        """
        Return name of the columns containing values of mixed types.

        Returns
        -------
        Set[str]
            Set of columns with values of different types
        """
        return self.column_list_by_type.mixed_type_cols

    @property
    def to_be_encoded_cat_cols(self):
        """
        Find categorical columns that needs encoding.

        It also checks if they are already encoded.

        Returns
        -------
        Set[str]
            Set of categorical column names that need encoding

        """
        to_be_encoded_categorical_cols = set()
        cols_by_type = self.column_list_by_type
        # TODO: Check this because maybe categorical columns that are numerical, do
        #  not need encoding probably!
        categorical_cols = (
            cols_by_type.str_categorical_cols | cols_by_type.num_categorical_cols
        )
        for categ_col in categorical_cols:
            if self.get_enc_column_from_original(categ_col) is None:
                to_be_encoded_categorical_cols.add(categ_col)

        return to_be_encoded_categorical_cols

    @property
    def med_exam_col_list(self) -> Set:
        """
        Get the name of the columns containing numerical values (metadata excluded).

        The method will exclude from numerical columns the ones that have the same
        repeated value, and the ones that contain metadata, but it will include columns
        with many NaN

        Returns
        -------
        Set
            Set containing ``numerical_cols`` without ``metadata_cols`` and
            ``constant_cols``
        """
        return self.column_list_by_type.med_exam_col_list

    # =====================
    # =    METHODS        =
    # =====================

    def least_nan_cols(self, threshold: int) -> Set:
        """
        Get the features with a count of NaN values lower than the ``threshold`` argument.

        Parameters
        ----------
        threshold: int
            Number of samples that have a NaN value. If the count of NaN values
            from a column is lower than ``threshold`` value, the name of the column
            will be in the returned set

        Returns
        -------
        Set
            Set of column names with a NaN count lower than the ``threshold`` argument
        """
        best_feature_list = set()
        # TODO: Add argument col_list to select the columns to analyze
        for c in self.med_exam_col_list:
            if sum(self.df[c].isna()) < threshold:
                best_feature_list.add(c)

        return best_feature_list

    def get_encoded_string_values_map(
        self, column_name: str
    ) -> Union[Dict[int, str], None]:
        """
        Return the encoded values map of the column named ``column_name``.

        Selecting the first operation of column_name because it will be the operation
        that created it (whether it is the encoded of one or multiple columns)

        Parameters
        ----------
        column_name: str
            Name of the derived column which we are looking the encoded_values_map of

        Returns
        -------
        Dict[int, str]
            Dict where the keys are the integer values of the ``column_name``, and the
                                   values are the values of the encoded column
        """
        try:
            encoded_map = self.feature_elaborations[column_name][
                0
            ].encoded_string_values_map
            return encoded_map
        except (KeyError, IndexError):
            logging.info(f"The column {column_name} was not among the operations.")
            return None

    def convert_column_id_to_name(self, col_id_list: Tuple[int]) -> Set:
        """
        Convert the column IDs to column names

        Parameters
        ----------
        col_id_list: List of column IDs to be converted to actual names

        Returns
        -------
        Set[str]
            Set of column names corresponding to ``col_id_list``

        """
        col_names = set()
        for c in col_id_list:
            col_names.add(self.df.columns[c])
        return col_names

    def check_duplicated_features(self) -> bool:
        """
        Check if there are columns with the same name (presumably duplicated).

        Returns
        -------
        bool
            Boolean that indicates if there are columns with the same name
        """
        # TODO: Rename to "contains_duplicated_features"
        # TODO: In case there are columns with the same name, check if the
        #  values are the same too and inform the user appropriately
        logger.info("Checking duplicated columns")
        # Check if there are duplicates in the df columns
        if len(self.df.columns) != len(set(self.df.columns)):
            logger.error("There are duplicated columns")
            return True
        else:
            return False

    def show_columns_type(self, col_list: Tuple[str] = None) -> None:
        """
        Print the type of the ``col_list`` columns.

        The possible identified types are:
        - float/int -> "numerical_col"
        - bool -> "bool_col"
        - str -> "string_col"
        - other types -> "other_col"

        Parameters
        ----------
        col_list: Tuple[str], optional
            Tuple of the name of columns that should be considered.
            If set to None, only the columns in ``self.feature_cols`` property.
        """
        col_list = self.feature_cols if col_list is None else col_list
        column_type_dict_list = Parallel(n_jobs=-1)(
            delayed(_find_single_column_type)(df_col=self.df[col]) for col in col_list
        )
        for i, col_type_dict in enumerate(column_type_dict_list):
            print(
                f"{i}: {col_type_dict[_COL_NAME_COLUMN]} -> {col_type_dict[_COL_TYPE_COLUMN]}"
            )

    def add_operation(self, feature_operation: FeatureOperation) -> None:
        """
        Add a new operation to the instance attribute ``feature_operations``

        For each column contained in ``original_columns`` or ``derived_columns``,
        attributes of ``feature_operation``, this method adds ``feature_operation``
        to the corresponding key of this instance attribute ``feature_operations``
        dictionary.
        This method also checks if every column in ``original_columns`` is
        in the list of ``metadata_cols`` attribute. In that case the derived column(s)
        contains metadata information and it (they) will be added to
        ``metadata_cols`` attribute.

        Parameters
        ----------
        feature_operation: FeatureOperation
            FeatureOperation instance that will be added to the keys of
            ``self.feature_operation`` attribute corresponding to ``original_columns``
            or ``derived_columns`` feature_operation attributes
        """
        # TODO: Think about the case where the same column is among "original_columns"
        #  and "derived_columns". Does this make sense? Should we raise an error?
        #  Would we add the same FeatureOperation twice to the column? That could
        #  create problems when looking for that operation because two are found!
        # This is used to identify the type of columns produced (it will be tested
        # and changed in the loop)
        is_metadata_cols = True
        # Loop for every original column name, so we append this operation to every
        # column_name
        for o in feature_operation.original_columns:
            self.feature_elaborations[o].append(feature_operation)
            # Check if at least one of original_columns is not in the list of
            # metadata_cols (in that case it does not contain only metadata information)
            if o not in self.metadata_cols:
                is_metadata_cols = False

        # TODO: Next line should now raise an Error because None means
        #  "Not Specified", and if user wants to leave it blank, he should use "()",
        #  (which is the new default value)
        if feature_operation.derived_columns is not None:
            # Add the same operation for each derived column
            for d in feature_operation.derived_columns:
                self.feature_elaborations[d].append(feature_operation)
            # If every original_column is in the list of metadata_cols, the
            # derived_columns is also derived by metadata_cols only and therefore
            # must be inserted in metadata_cols set, too
            if is_metadata_cols:
                self._metadata_cols = self.metadata_cols.union(
                    set(feature_operation.derived_columns)
                )
            # Add the derived columns to the list of the instance
            self.derived_columns = self.derived_columns.union(
                set(feature_operation.derived_columns)
            )

    def find_operation_in_column(
        self, feat_operation: FeatureOperation
    ) -> Union[FeatureOperation, None]:
        """
        Search an operation previously executed on this object.

        This method checks in ``feature_operations`` attribute if an operation similar
        to ``feat_operation`` has already been performed on this instance.
        Therefore, the unknown attributes of the operation can be set to None
        (corresponding to "Not Specified") and, if the operation is found,
        it will contain full information and it will be returned.

        Parameters
        ----------
        feat_operation: FeatureOperation
            FeatureOperation instance containing some information about the operation
            the user is looking for. It must contain either ``original_columns`` or
            ``derived_columns`` attribute, otherwise no operation is returned.
            If some attributes are unknown and the user wants to leave them unspecified,
            they can be set to None.

        Returns
        -------
        FeatureOperation
            FeatureOperation instance that has been performed on the Dataset
            instance, and that has the same specified attributes of ``feat_operation``.
            If no operation similar to ``feat_operation`` is found, None is returned.
            If more than one operation similar to ``feat_operation`` is found,
            MultipleOperationsFoundError is raised.

        Raises
        ------
        MultipleOperationsFoundError
            Exception raised when more than one operation similar to ``feat_operation``
            is found.

        """
        # Select only the first element of the original_columns (since each of the
        # columns is linked to an operation) and check if the 'feat_operation'
        # argument is among the operations linked to that column.
        if feat_operation.original_columns is not None:
            selected_column_operations = self.feature_elaborations[
                feat_operation.original_columns[0]
            ]
        elif feat_operation.derived_columns is not None:
            selected_column_operations = self.feature_elaborations[
                feat_operation.derived_columns[0]
            ]
        else:
            logging.warning(
                "It is not possible to look for an operation if neither "
                "original columns nor derived columns attributes are provided"
            )
            return None

        similar_operations = []
        for f in selected_column_operations:
            if f == feat_operation:
                similar_operations.append(f)

        if len(similar_operations) == 0:
            return None
        elif len(similar_operations) == 1:
            return similar_operations[0]
        else:
            nl = "\n"
            raise MultipleOperationsFoundError(
                "Multiple operations were found. Please provide additional information."
                "\nOperations found: "
                + str(
                    [
                        f"{nl * 2}{i}. {sim_op}"
                        for (i, sim_op) in enumerate(similar_operations)
                    ]
                )
            )

    def get_enc_column_from_original(
        self,
        column_name: str,
        encoder: sklearn.preprocessing._encoders._BaseEncoder = None,
    ) -> Union[Tuple[str, ...], None]:
        """
        Return the name of the column with encoded values, derived from ``column_name``.

        This method checks if an operation of type ``OperationTypeEnum.CATEGORICAL_ENCODING``
        has been performed on the ``column_name`` column. In case it is, it will return
        the name of the column with encoded values, otherwise None.

        Parameters
        ----------
        column_name: str
            Name of the original column that has been encoded
        encoder: sklearn.preprocessing._encoders._BaseEncoder, optional
            Type of encoder used to encode ``column_name`` column

        Returns
        -------
        Tuple[str]
            Tuple of names of one (or more) columns with encoded values. None
            if the column has not been encoded

        See Also
        --------
        If the user wants to check the columns generated by operations of different
        type, 'find_operation_in_column' method should be employed.
        """
        feat_operation = FeatureOperation(
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            original_columns=column_name,
            encoder=encoder,
            derived_columns=None,
        )
        found_operat = self.find_operation_in_column(feat_operation)
        # If no operation is found, or the column is the derived column
        # (i.e. the input of encoding function), we return None
        # TODO: Is the check "column_name in found_operat.derived_columns" necessary?
        if found_operat is None or column_name in found_operat.derived_columns:
            return None
        else:
            return found_operat.derived_columns

    def get_original_from_enc_column(
        self,
        column_name: str,
        encoder: sklearn.preprocessing._encoders._BaseEncoder = None,
    ) -> Union[Tuple[str, ...], None]:
        """
        Return the name of the column with original values, used to generate ``column_name``.

        This method checks if an operation of type OperationTypeEnum.CATEGORICAL_ENCODING
        has been performed and generated ``column_name`` column. In case it exists, it
        will return the name of the column with original values, otherwise None.

        Parameters
        ----------
        column_name: str
            Name of the column that has been encoded.
        encoder: sklearn.preprocessing._encoders._BaseEncoder, optional
            Type of encoder used to generate ``column_name`` column.

        Returns
        -------
        Tuple[str]
            Tuple of names of one (or more) column name(s) with original values. None
            if the column is not generated by a OperationTypeEnum.CATEGORICAL_ENCODING
            operation.

        See Also
        --------
        If the user wants to check the columns generated by operations of different
        type, 'find_operation_in_column' method should be employed.
        """
        feat_operation = FeatureOperation(
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            original_columns=None,
            encoder=encoder,
            derived_columns=column_name,
        )
        found_operat = self.find_operation_in_column(feat_operation)
        # If no operation is found, or the column is the derived column
        # (i.e. the input of encoding function), we return None
        # TODO: Is the check "column_name in found_operat.original_columns" necessary?
        if found_operat is None or column_name in found_operat.original_columns:
            return None
        else:
            return found_operat.original_columns

    def to_file(self, filename: Union[Path, str], overwrite: bool = False) -> None:
        """
        Export Dataset instance to ``filename``

        This function uses "shelve" module that creates 3 files containing only
        the Dataset object.

        Parameters
        ----------
        filename: Union[Path, str]
            Name/Path of the file where the data dump will be exported
        overwrite: bool, optional
            Option to overwrite the file if it already exists as ``filename``.
            Default set to False

        Raises
        ------
        FileExistsError
            If a file in ``filename`` path is already present and ``overwrite`` is set
            to False. In case overwriting is not a problem, ``overwrite`` should be set
            to True.
        """
        filename = str(filename)
        if not overwrite:
            if os.path.exists(filename):
                raise FileExistsError(
                    f"File {filename} already exists. If overwriting is not a problem, "
                    f"set the 'overwrite' argument to True"
                )

        with shelve.open(filename, "n") as my_shelf:  # 'n' for new
            try:
                my_shelf["dataset"] = self
            except TypeError as e:
                logging.error(f"ERROR shelving: \n{e}")
            except KeyError as e:
                logging.error(f"Exporting data unsuccessful: \n{e}")

    def __str__(self) -> str:
        """
        Return text with the number of columns for every variable type

        Returns
        -------
        str
            String that describes the info and types of the columns of the
            ``df`` attribute.
        """
        return (
            f"{self.column_list_by_type}"
            f"\nColumns with many NaN: {len(self.many_nan_columns)}"
        )

    def __call__(self) -> pd.DataFrame:
        """
        Return pandas DataFrame ``df`` attribute

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame ``df`` attribute of this instance
        """
        return self.df


def copy_dataset_with_new_df(dataset: Dataset, new_pandas_df: pd.DataFrame) -> Dataset:
    """
    Copy a Dataset instance using "shallow_copy"

    Every attribute of the Dataset instance will be kept, except for ``df``
    attribute that is replaced by ``new_pandas_df``.
    Use this carefully to avoid keeping information of previous operation
    associated with columns that are no longer present.

    Parameters
    ----------
    dataset: Dataset
        Dataset instance that will be copied
    new_pandas_df: pd.DataFrame
        Pandas DataFrame instance that contains the new values of ``df`` attribute
        of the new Dataset instance

    Returns
    -------
    Dataset
        Dataset instance with same attribute values as ``dataset`` argument,
        but with ``new_pandas_df`` used as ``df`` attribute value.
    """
    if not set(dataset.df.columns).issubset(new_pandas_df.columns):
        logging.warning(
            "Some columns of the previous Dataset instance "
            "are being lost, but information about operation on them "
            "is still present"
        )
    new_dataset = copy.copy(dataset)
    new_dataset.df = new_pandas_df
    return new_dataset


def read_file(filename: Union[Path, str]) -> Dataset:
    """
    Import a Dataset instance stored inside ``filename`` file.

    This function uses 'shelve' module and it expects to find 3 files with
    suffixes ".dat", ".bak", ".dir" that contain only one Dataset
    instance.

    Parameters
    ----------
    filename: Union[Path, str]
        Name/Path of the file where the data dump may be found.

    Returns
    -------
    Dataset
        Dataset instance that was saved in ``filename`` path.

    Raises
    ------
    TypeError
        If no Dataset instances were found inside the ``filename`` file.
    MultipleObjectsInFileError
        If multiple objects were found inside the ``filename`` file.
    """
    try:
        my_shelf = shelve.open(str(filename))
    except dbm.error:
        # We leave the FileNotFoundError management to the function
        raise NotShelveFileError(
            f"The file {filename} was not created by 'shelve' module or no "
            f"db type could be determined"
        )
    else:
        # Check how many objects have been stored
        if len(my_shelf.keys()) != 1:
            raise MultipleObjectsInFileError(
                f"There are {len(my_shelf.keys())} objects in file {filename}. Expected 1."
            )
        # Retrieve the single object
        dataset = list(my_shelf.values())[0]

        # Check if the object is a Dataset instance
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"The object is not a Dataset "
                f"instance, but it is {dataset.__class__}"
            )
        my_shelf.close()

    return dataset