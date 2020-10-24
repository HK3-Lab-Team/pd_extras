import collections
import copy
import dbm
import logging
import os
import shelve
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed

from .convert_to_mixed_type import _ConvertDfToMixedType
from .exceptions import MultipleObjectsInFileError, NotShelveFileError
from .feature_operations import FeatureOperation
from .operations_list import OperationsList
from .settings import CATEG_COL_THRESHOLD
from .util import lazy_property

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


@dataclass
class _ColumnListByType:
    """
    This dataclass is to gather the different column types inside a pd.DataFrame.
    The columns are split according to the type of their values.
    """

    constant_cols: Set = field(default_factory=set)
    mixed_type_cols: Set = field(default_factory=set)
    numerical_cols: Set = field(default_factory=set)
    med_exam_col_list: Set = field(default_factory=set)
    str_cols: Set = field(default_factory=set)
    str_categorical_cols: Set = field(default_factory=set)
    num_categorical_cols: Set = field(default_factory=set)
    other_cols: Set = field(default_factory=set)
    bool_cols: Set = field(default_factory=set)

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
        new_columns_encoding_maps: Union[
            DefaultDict[str, List["FeatureOperation"]], None
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
                data = get_df_from_csv(data_file)
        else:
            data = df_object

        self._metadata_cols = set(metadata_cols)
        if feature_cols is None:
            self._feature_cols = set(data.columns) - self.metadata_cols
        else:
            self._feature_cols = set(feature_cols)

        # Dict of Lists ->
        #         key: column_name,
        #         value: List of FeatureOperation instances
        if new_columns_encoding_maps is None:
            # Dict already initialized to lists for every "column_name"
            new_columns_encoding_maps = collections.defaultdict(list)
        self.feature_elaborations = new_columns_encoding_maps
        self._operations_history = OperationsList()

        # Columns generated by Feature Refactoring (e.g.: encoding, bin_splitting)
        self.derived_columns = set()

        self._data = data

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

    def nan_columns(self, nan_ratio: float = 1) -> Set[str]:
        """Return name of the columns containing at least a ``nan_ratio`` ratio of NaNs.

        Select the columns where the nan_ratio of NaN values over the
        sample count is higher than ``nan_ratio`` (in range [0,1]).

        Parameters
        ----------
        nan_ratio : float, optional
            Minimum ratio “nan samples”/”total samples” for the column to be considered
            a “nan column”. Default is 1, meaning that only the columns entirely composed
            by NaNs will be returned.

        Returns
        -------
        Set[str]
            Set of column names with NaN ratio higher than ``nan_ratio`` parameter.
        """
        nan_columns = set()
        for c in self.feature_cols:
            # Check number of NaN
            if sum(self._data[c].isna()) > nan_ratio * self._data.shape[0]:
                nan_columns.add(c)

        return nan_columns

    @property
    def constant_cols(self) -> Set[str]:
        """Return name of the columns containing only one repeated value.

        Returns
        -------
        Set[str]
            Set of column names with only one repeated value
        """
        df_nunique = self._data[self.feature_cols].nunique(dropna=False)
        constant_cols = df_nunique[df_nunique == 1].index
        return set(constant_cols)

    @property
    def trivial_columns(self) -> Set[str]:
        """
        Return name of the columns containing many NaN or only one repeated value.

        This function return the name of the column that were returned by
        ``constant_cols`` property or ``nan_columns`` method.

        Returns
        -------
        Set[str]
            Set containing the name of the columns with many NaNs or with only
            one repeated value
        """
        return self.nan_columns(nan_ratio=0.999).union(self.constant_cols)

    @lazy_property
    def _columns_type(self) -> _ColumnListByType:
        """
        Analyze the instance and return an object with the column list split by type.

        NOTE: This gathers many properties/column_types together and
        returns an object containing them because calculating them together is much
        more efficient when we need two (or more) of them (and we do not waste much
        time if we only need one column type).

        Returns
        -------
        _ColumnListByType
            _ColumnListByType instance containing the column list split by type
        """
        constant_cols = self.constant_cols

        # TODO: Exclude NaN columns (self.nan_cols) from `col_list` too (so they will
        #  not be included in num_categorical_cols just for one not-Nan value)

        col_list = self.feature_cols - constant_cols

        mixed_type_cols = set()
        numerical_cols = set()
        str_cols = set()
        bool_cols = set()
        other_cols = set()
        categorical_cols = set()

        PD_INFER_TYPE_MAP = {
            "string": str_cols,
            "bytes": other_cols,
            "floating": numerical_cols,
            "integer": numerical_cols,
            "mixed-integer": mixed_type_cols,
            "mixed-integer-float": numerical_cols,
            "decimal": numerical_cols,
            "complex": numerical_cols,
            "boolean": bool_cols,
            "datetime64": other_cols,
            "datetime": other_cols,
            "date": other_cols,
            "timedelta64": other_cols,
            "timedelta": other_cols,
            "time": other_cols,
            "period": other_cols,
            "mixed": mixed_type_cols,
            "interval": str_cols,
            "category": categorical_cols,
            "categorical": categorical_cols,
        }

        mixed_data = self._data_to_mixed_types(self._data)

        for col in col_list:
            col_type = pd.api.types.infer_dtype(mixed_data[col], skipna=True)
            PD_INFER_TYPE_MAP[col_type].add(col)

        str_categorical_cols = self._get_categorical_cols(str_cols)
        num_categorical_cols = self._get_categorical_cols(numerical_cols)

        for categorical_col in categorical_cols:
            if mixed_data[categorical_col].dtype.categories.inferred_type == "integer":
                num_categorical_cols.add(categorical_col)
                numerical_cols.add(categorical_col)
            elif mixed_data[categorical_col].dtype.categories.inferred_type == "string":
                str_categorical_cols.add(categorical_col)
                str_cols.add(categorical_col)
            else:
                raise RuntimeError("there is something wrong with the type guessing...")

        # `num_categorical_cols` is already included in `numerical_cols`,
        # so no need to add it here
        med_exam_col_list = (
            numerical_cols | bool_cols - constant_cols - self.metadata_cols
        )

        return _ColumnListByType(
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

    @property
    def mixed_type_columns(self) -> Set[str]:
        """Return the name of the columns with mixed type.

        Returns
        -------
        Set[str]
            The names of the columns with mixed type
        """
        return self._columns_type.mixed_type_cols

    @property
    def numerical_columns(self) -> Set[str]:
        """Return the name of the columns with numerical type.

        Returns
        -------
        Set[str]
            The names of the columns with numerical type
        """
        return self._columns_type.numerical_cols

    @property
    def med_exam_col_list(self) -> Set[str]:
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
        return self._columns_type.med_exam_col_list

    @property
    def str_columns(self) -> Set[str]:
        """Return the name of the columns with string type.

        Returns
        -------
        Set[str]
            The names of the columns with string type
        """
        return self._columns_type.str_cols

    @property
    def str_categorical_columns(self) -> Set[str]:
        """Return the name of the columns with string categorical type.

        Returns
        -------
        Set[str]
            The names of the columns with string categorical type
        """
        return self._columns_type.str_categorical_cols

    @property
    def num_categorical_columns(self) -> Set[str]:
        """Return the name of the columns with numerical categorical type.

        Returns
        -------
        Set[str]
            The names of the columns with numerical categorical type
        """
        return self._columns_type.num_categorical_cols

    @property
    def bool_columns(self) -> Set[str]:
        """Return the name of the columns with boolean type.

        Returns
        -------
        Set[str]
            The names of the columns with boolean type
        """
        return self._columns_type.bool_cols

    @property
    def other_type_columns(self) -> Set[str]:
        """Return the name of the columns with non-conventional type.

        Types that are included in this category are: bytes, datetime64, datetime, date,
        timedelta64, timedelta, time, period.

        Returns
        -------
        Set[str]
            The names of the columns with non-conventional type
        """
        return self._columns_type.other_cols

    @property
    def data(self) -> pd.DataFrame:
        """Return data as a pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Data
        """
        return self._data

    @property
    def operations_history(self) -> OperationsList:
        """Return the history of the operations performed on the dataset.

        Returns
        -------
        OperationsList
            History of the operations
        """
        return self._operations_history

    @property
    def to_be_fixed_cols(self) -> Set[str]:
        """
        Return name of the columns containing values of mixed types.

        Returns
        -------
        Set[str]
            Set of columns with values of different types
        """
        return self._columns_type.mixed_type_cols

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
        # TODO: Check this because maybe categorical columns that are numerical, do
        #  not need encoding probably!
        categorical_cols = self.str_categorical_cols | self.num_categorical_cols
        for categ_col in categorical_cols:
            if self.get_enc_column_from_original(categ_col) is None:
                to_be_encoded_categorical_cols.add(categ_col)

        return to_be_encoded_categorical_cols

    # =====================
    # =    METHODS        =
    # =====================
    @staticmethod
    def _data_to_mixed_types(df: pd.DataFrame):
        """
        Transform 'object'-typed column values to appropriate types.

        This static method analyzes the DataFrame ``df`` columns that have
        dtype='object'. The columns that have numeric and string values are interpreted
        and casted by Pandas as columns with dtype='object' and all the numeric,
        boolean or datetime values are transformed into string typed values
        (e.g. 2 -> '2').
        This would disturb the Dataset column type inference, so this method restore
        numeric, boolean and datetime values to the appropriate types for columns
        with dtype='object'.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the columns that will be analyzed. It will not be
            modified inplace.

        Returns
        -------
        pd.DataFrame
            New DataFrame with numeric and boolean values restored to the appropriate
            types for columns with dtype='object'.
        """
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            mixedtype_converter = _ConvertDfToMixedType(column=col)
            df = mixedtype_converter(df)

        return df

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
            unique_val_nb = len(self._data[col].unique())
            if unique_val_nb < 7 or (
                unique_val_nb < self._data[col].count() // CATEG_COL_THRESHOLD
            ):
                self._data[col] = self._data[col].astype("category")
                categorical_cols.add(col)

        return categorical_cols

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
            col_names.add(self._data.columns[c])
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
        if len(self._data.columns) != len(set(self._data.columns)):
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
            delayed(_find_single_column_type)(df_col=self._data[col])
            for col in col_list
        )
        for i, col_type_dict in enumerate(column_type_dict_list):
            print(
                f"{i}: {col_type_dict[_COL_NAME_COLUMN]} -> {col_type_dict[_COL_TYPE_COLUMN]}"
            )

    def track_history(self, feature_operation: "FeatureOperation") -> None:
        """
        Add a new operation to the history.

        If every original column is in the ``metadata_cols`` attribute, then all the
        derived columns will be added to ``metadata_cols``.

        Parameters
        ----------
        feature_operation: FeatureOperation
            FeatureOperation to be added to the history
        """
        self._operations_history += feature_operation

        if feature_operation.derived_columns is not None:
            # If every original column is in the list of metadata_cols, the
            # derived_columns is also derived by metadata_cols only and therefore
            # must be inserted in metadata_cols set, too

            if all(
                [column in self.metadata_cols for column in feature_operation.columns]
            ):
                self._metadata_cols = self.metadata_cols.union(
                    set(feature_operation.derived_columns)
                )

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
            f"{self._columns_type}"
            f"\nColumns with many NaN: {len(self.nan_columns(0.999))}"
        )


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
    if not set(dataset._data.columns).issubset(new_pandas_df.columns):
        logging.warning(
            "Some columns of the previous Dataset instance "
            "are being lost, but information about operation on them "
            "is still present"
        )
    new_dataset = copy.copy(dataset)
    new_dataset._data = new_pandas_df
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
