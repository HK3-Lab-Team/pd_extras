import collections
import copy
import logging
import os
import shelve
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple, Union

import pandas as pd

from .feature_enum import EncodingFunctions, OperationTypeEnum
from .settings import CATEG_COL_THRESHOLD

logger = logging.getLogger(__name__)


def get_df_from_csv(df_filename):
    try:
        df = pd.read_csv(df_filename)
        logger.info("Data imported from file successfully")
        return df
    except FileNotFoundError as e:
        logger.error(e)
        return None


def _to_tuple(x: Union[str, Iterable]) -> Tuple:
    if x is None:
        return None
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, list) or isinstance(x, set):
        return tuple(x)
    else:
        return tuple([x])


class FeatureOperation:
    """
    This is a Class to store the operations executed on df.
    """

    def __init__(
        self,
        operation_type: OperationTypeEnum,
        original_columns: Union[Tuple[str], str, None] = None,
        derived_columns: Union[Tuple[str], str, None] = None,
        encoded_values_map: Union[Dict[int, Any], None] = None,
        encoder=None,
        details: Union[Dict, None] = None,
    ):
        """
        This is a Model to store the operations executed on df.
        @param details: It contains details about the operation, like the map between encoded
            value and original value. It may be set to None
        @param derived_columns: if it is equal to original_columns, it will be reassigned to None
        """
        # This is to avoid that a single column (i.e. a string) is interpreted as a tuple of single chars
        original_columns = _to_tuple(original_columns)
        derived_columns = _to_tuple(derived_columns)

        self.original_columns = original_columns
        self.operation_type = operation_type
        self.details = details
        self.encoder = encoder
        self.encoded_values_map = encoded_values_map
        self.encoding_function = encoder

        if derived_columns == original_columns:
            self.derived_columns = None
        else:
            self.derived_columns = derived_columns

    @property
    def encoded_string_values_map(self) -> Union[Dict[int, str], None]:
        """
        This is a modified version of the attribute 'self.encoded_values_map'
        where we want string values instead of tuples
        """
        if self.encoded_values_map is None:
            return None
        else:
            encoded_string_values_map = {}
            for key, value in self.encoded_values_map.items():
                if not isinstance(value, str):
                    encoded_string_values_map[key] = "-".join(str(x) for x in value)
                else:
                    encoded_string_values_map[key] = value
            return encoded_string_values_map

    def __eq__(self, other):
        """
        This is useful when we want to compare two FeatureOperation instances (used in method
        'find_operation_in_column'). The conditions to identify an equality are:
        1. Same operation_type (one of the values of OperationTypeEnum)
        2. Original columns OR derived columns are the same (not both required so that I can input one list
        of columns and find the other
        3. If the encoder is provided for both instances, it must be the same (it could
        be OneHotEncoder/OrdinalEncoder,...)

        Parameters
        ----------
        other: FeatureOperation
            The instance that has to be compared with self

        Returns
        -------
        bool: True if all the conditions above are fulfilled and False otherwise
        """
        if isinstance(other, self.__class__):
            if (
                self.operation_type == other.operation_type
                and (
                    self.original_columns is None
                    or other.original_columns is None
                    or set(self.original_columns) == set(other.original_columns)
                )
                and (
                    self.derived_columns is None
                    or other.derived_columns is None
                    or set(self.derived_columns) == set(other.derived_columns)
                )
                and (
                    self.encoder is None
                    or other.encoder is None
                    or self.encoder == other.encoder
                )
            ):
                return True
        return False

    def __str__(self):
        return (
            f"Columns that have been used to produce the result: {self.original_columns}"
            f"\nThe type of the operation that has been applied is: {self.operation_type}"
            f"\nThe columns that have been created after the operation are: {self.derived_columns}"
            f"\nThe map between the original values and the encoded ones is: \n{self.encoded_values_map}"
            f"\nThe encoding function that has been used is: {self.encoding_function}"
        )


@dataclass
class ColumnListByType:
    """
    This dataclass is to gather the different column types inside a pd.DataFrame.
    The columns are split according to the type of their values.
    """

    same_value_cols: Set
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
            f"\n\t4.\tOne repeated value: {len(self.same_value_cols)}"
        )


class DataFrameWithInfo:
    def __init__(
        self,
        metadata_cols: Tuple = (),
        data_file: str = None,
        df_object: pd.DataFrame = None,
        nan_percentage_threshold: float = 0.999,
        metadata_as_features: bool = False,
        new_columns_encoding_maps: Union[
            DefaultDict[str, List[FeatureOperation]], None
        ] = None,
    ):
        """
        This class contains some useful methods and attributes mostly related to features/columns.
        It helps in keeping track of the operations on DataFrame, and returns subgroups of columns split by type

        @param new_columns_encoding_maps: Union[DefaultDict[str, List[FeatureOperation]], None]
        @param metadata_cols: Tuple of the names of the columns that have infos related to the patient
            and that are not from medical exams
        @param data_file:
        @param nan_percentage_threshold: Float value for the threshold of NaN values count to decide
        if the column is relevant or not
        """
        if df_object is None:
            if data_file is None:
                logging.error("Provide either data_file or df_object as argument")
            else:
                self.df = get_df_from_csv(data_file)
        else:
            self.df = df_object

        self.metadata_cols = set(metadata_cols)
        self.nan_percentage_threshold = nan_percentage_threshold

        # Dict of Lists ->
        #         key: column_name,
        #         value: List of FeatureOperation instances
        if new_columns_encoding_maps is None:
            new_columns_encoding_maps = collections.defaultdict(list)
        self.feature_elaborations = new_columns_encoding_maps

        # Columns generated by Feature Refactoring (e.g.: encoding, bin_splitting)
        self.derived_columns = set()

        # Option to choose whether to include "metadata_cols" in
        # later analysis (treating them as normal features or not)
        self.metadata_as_features = metadata_as_features

    # =====================
    # =    PROPERTIES     =
    # =====================

    @property
    def many_nan_columns(self) -> Set:
        """
        :return: Set[str] -> List of column names with 99.9% (or nan_percentage_threshold) of NaN
        """
        many_nan_columns = set()
        for c in self.df.columns:
            # Check number of NaN
            if (
                sum(self.df[c].isna())
                > self.nan_percentage_threshold * self.df.shape[0]
            ):
                many_nan_columns.add(c)

        return many_nan_columns

    @property
    def same_value_cols(self) -> Set:
        """
        :return: Set[str] -> List of column names with the same repeated value
        """
        same_value_columns = set()
        for c in self.df.columns:
            # Check number of unique values
            if len(self.df[c].unique()) == 1:
                same_value_columns.add(c)

        return same_value_columns

    @property
    def trivial_columns(self):
        """
        Returns
        -------
            Set -> Combination of the columns with many NaNs and with the same repeated value
        """
        return self.many_nan_columns.union(self.same_value_cols)

    @property
    def column_list_by_type(self) -> ColumnListByType:
        """
        This returns an object containing the list of DF columns split according to their types.
        IMPLEMENTATION NOTE: This gathers many properties/column_types together and returns
        an object containing them because calculating them together is much more efficient
        when we need two (or more) of them (and we do not waste much time if we only need one).
        :return:  ColumnListByType: NamedTuple with the column list split by type
        """
        mixed_type_cols = set()
        numerical_cols = set()
        str_cols = set()
        other_cols = set()
        bool_cols = set()
        same_value_cols = self.same_value_cols

        if self.metadata_as_features:
            col_list = set(self.df.columns) - same_value_cols
        else:
            col_list = set(self.df.columns) - same_value_cols - self.metadata_cols

        # TODO: Change to .apply(axis=0)  on columns!
        for col in col_list:
            # Select not-NaN only
            notna_col_df = self.df[self.df[col].notna()]
            # Compare the first_row type with every other row of the same column
            col_types = notna_col_df[col].apply(lambda r: str(type(r))).values
            has_same_values = all(col_types == col_types[0])
            if has_same_values:
                # Check the type of the first element
                col_type = col_types[0]
                unique_values = notna_col_df[col].unique()
                if "bool" in col_type or (
                    len(unique_values) == 2
                    and unique_values[0] in [0, 1]
                    and unique_values[1] in [0, 1]
                ):
                    # True/False are considered as [0,1]
                    bool_cols.add(col)
                elif "str" in col_type:
                    # String columns
                    str_cols.add(col)
                elif "float" in col_type or "int" in col_type:
                    # look if the col_type contains 'int' or 'float' keywords
                    numerical_cols.add(col)
                else:
                    other_cols.add(col)
            else:
                mixed_type_cols.add(col)

        str_categorical_cols = self._get_categorical_cols(str_cols)
        num_categorical_cols = self._get_categorical_cols(numerical_cols)
        med_exam_col_list = (
            numerical_cols | bool_cols - same_value_cols - self.metadata_cols
        )

        return ColumnListByType(
            mixed_type_cols=mixed_type_cols,
            same_value_cols=same_value_cols,
            numerical_cols=numerical_cols | bool_cols,
            med_exam_col_list=med_exam_col_list,
            str_cols=str_cols,
            str_categorical_cols=str_categorical_cols,
            num_categorical_cols=num_categorical_cols,
            bool_cols=bool_cols,
            other_cols=other_cols,
        )

    def _get_categorical_cols(self, str_cols):
        """
        This method is to identify every categorical column in df_info.
        It will also set those column's types to "category"
        Parameters
        ----------
        str_cols

        Returns
        -------

        """
        categorical_cols = set()
        # OLD_WAY:
        # Select the columns with dtype != 'bool','int','unsigned int', 'float', 'complex'
        # non_numeric_cols = set(filter(lambda c: self.df[c].dtype.kind not in 'biufc', col_list))
        for col in str_cols:
            unique_val_nb = len(self.df[col].unique())
            # To avoid considering features like 'PatientID' (or every string column) as categorical,
            # we only select the columns with few unique values
            if unique_val_nb < 7 or (
                unique_val_nb < self.df[col].count() // CATEG_COL_THRESHOLD
            ):
                categorical_cols.add(col)

        return categorical_cols

    @property
    def to_be_fixed_cols(self) -> Set:
        return self.column_list_by_type.mixed_type_cols

    @property
    def to_be_encoded_cat_cols(self):
        """
        This property is to find categorical columns that needs encoding, so it also checks
        if they are already encoded.
        """
        to_be_encoded_categorical_cols = set()
        cols_by_type = self.column_list_by_type
        categorical_cols = (
            cols_by_type.str_categorical_cols | cols_by_type.num_categorical_cols
        )
        for categ_col in categorical_cols:
            if self.get_enc_column_from_original(categ_col) is None:
                to_be_encoded_categorical_cols.add(categ_col)

        return to_be_encoded_categorical_cols

    @property
    def med_exam_col_list(self) -> Set:
        """ This returns numerical_cols without metadata_cols and same_value_cols"""
        return self.column_list_by_type.med_exam_col_list

    # =====================
    # =    METHODS        =
    # =====================

    def get_encoded_string_values_map(
        self, column_name: str
    ) -> Union[Dict[int, str], None]:
        """
        This method returns the encoded values map of the column named 'column_name'.
        Selecting the first operation of column_name because it will be the operation that created it (whether
        it is the encoded of one or multiple columns)
        @param column_name: str -> Name of the derived column which we are looking the encoded_values_map of
        @return: Dict[int, str] -> Dict where the keys are the integer values of the 'column_name', and the
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

    def convert_column_id_to_name(self, col_id_list) -> Set:
        col_names = set()
        for c in col_id_list:
            col_names.add(self.df.columns[c])
        return col_names

    def check_duplicated_features(self) -> bool:

        logger.info("Checking duplicated columns")
        # Check if there are duplicates in the df columns
        if len(self.df.columns) != len(set(self.df.columns)):
            logger.error("There are duplicated columns")
            return False
        else:
            return True

    def show_columns_type(self, col_list=None):
        """ It prints the type of the first element of the columns """
        col_list = self.df.columns if col_list is None else col_list

        for col in col_list:
            print(type(self.df[col].iloc[0]))

    def add_operation(self, feature_operation: FeatureOperation):
        """
        This method adds a new operation corresponding to the key:'original_columns' (and eventually
        to 'derived_columns' if generated by the operation).
        It will also update the df with the "df_new" argument and it will automatically update every other argument
        accordingly.
        It also checks if at least one of original_columns is not in the list of metadata_cols
        (in that case it does not contain only metadata information)
        """
        # This is used to identify the type of columns produced (it will be tested and changed in the loop)
        is_metadata_cols = True
        # Loop for every original column name, so we append this operation to every column_name
        for o in feature_operation.original_columns:
            self.feature_elaborations[o].append(feature_operation)
            # Check if at least one of original_columns is not in the list of metadata_cols
            # (in that case it does not contain only metadata information)
            if o not in self.metadata_cols:
                is_metadata_cols = False
        if feature_operation.derived_columns is not None:
            # Add the same operation for each derived column
            for d in feature_operation.derived_columns:
                self.feature_elaborations[d].append(feature_operation)
            # If every original_column is in the list of metadata_cols, the derived_columns is also
            # derived by metadata_cols only and therefore must be inserted in metadata_cols set, too
            if is_metadata_cols:
                self.metadata_cols = self.metadata_cols.union(
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
        This method retrieves a specific "feat_operation", instance of FeatureOperation, using its attributes.
        It may be used to check if an operation has already been performed
        """
        # Select only the first element of the original_columns (since each of the columns is linked
        # to an operation) and check if the 'feat_operation' argument is among the operations linked to
        # that column.
        if feat_operation.original_columns is not None:
            selected_column_operations = self.feature_elaborations[
                feat_operation.original_columns[0]
            ]
        else:
            if feat_operation.derived_columns is not None:
                selected_column_operations = self.feature_elaborations[
                    feat_operation.derived_columns[0]
                ]
            else:
                logging.warning(
                    "It is not possible to look for an operation if neither "
                    "original columns nor derived columns attributes are provided"
                )
                return None

        for f in selected_column_operations:
            if f == feat_operation:
                return f

        return None

    def get_enc_column_from_original(
        self, column_name, encoder: EncodingFunctions = None,
    ) -> Union[Tuple[str, ...], None]:
        """
        This checks if the column is already encoded. In case it is, it will return the name of the
        column with encoded values, otherwise None.
        WARNING: If you also want to check the operation used for Encoding, you should use
        'find_operation_in_column' method

        Parameters
        ----------
        column_name: Column to be checked
        encoder: EncodingFunctions
            Select the type of encoder used

        Returns
        -------
        Tuple[str, ...] -> Returns name of the column with encoded values. Returns None if the column
                           has not been encoded
        """
        feat_operation = FeatureOperation(
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            original_columns=column_name,
            encoder=encoder,
        )
        found_operat = self.find_operation_in_column(feat_operation)
        # If no operation is found, or the column is the derived column (i.e. the input of encoding function),
        # we return None
        if found_operat is None or column_name in found_operat.derived_columns:
            return None
        else:
            return found_operat.derived_columns

    def get_original_from_enc_column(
        self, column_name, encoder: EncodingFunctions = None,
    ) -> Union[Tuple[str, ...], None]:
        """
        This checks if the column you provide is the encoded version of a original one.
        In case it is, it will return the name of the column with original values, otherwise None.
        WARNING: This is to check among OperationTypeEnum.CATEGORICAL_ENCODING operations.
        If you also want to check other operation types used for Encoding, you should use
        'find_operation_in_column' method

        Parameters
        ----------
        column_name: Column to be checked
        encoder: EncodingFunctions
            Select the type of encoder used

        Returns
        -------
        Tuple[str, ...] -> Returns name of the column with encoded values. Returns None if the column
                           has not been encoded
        """
        feat_operation = FeatureOperation(
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            original_columns=column_name,
            encoder=encoder,
        )
        found_operat = self.find_operation_in_column(feat_operation)
        # If no operation is found, or the column is the derived column (i.e. the input of encoding function),
        # we return None
        if found_operat is None or column_name in found_operat.original_columns:
            return None
        else:
            return found_operat.original_columns

    def least_nan_cols(self, threshold: int) -> Set:
        """
        This is to get the features with a NaN count lower than the "threshold" argument
        """
        best_feature_list = set()
        for c in self.med_exam_col_list:
            if sum(self.df[c].isna()) < threshold:
                best_feature_list.add(c)

        return best_feature_list

    def __str__(self):
        """ Returns text of the number of features for every variable type """
        return (
            f"{self.column_list_by_type}"
            f"\nColumns with many NaN: {len(self.many_nan_columns)}"
        )

    def __call__(self):
        return self.df


def copy_df_info_with_new_df(
    df_info: DataFrameWithInfo, new_pandas_df: pd.DataFrame
) -> DataFrameWithInfo:
    """
    This function is to copy a DataFrameWithInfo instance as "shallow_copy"
    @param df_info: DataFrameWithInfo instance that will be copied
    @param new_pandas_df: pd.DataFrame instance that contains the new values of 'df' attribute
        of the new DataFrameWithInfo instance
    @return: DataFrameWithInfo instance with same attributes as 'df_info' argument, except for 'df'
        attribute that is replaced by new_pandas_df
    """
    new_df_info = copy.copy(df_info)
    new_df_info.df = new_pandas_df
    return new_df_info


def import_df_with_info_from_file(filename) -> DataFrameWithInfo:
    """
    This uses 'shelve' module to import the data of a previous DataFrameWithInfo instance
    @param filename: Name/Path of the file where the data dump may be found
    @return: No return value
    """
    # We leave the error management to the function (FileNotFoundError)
    my_shelf = shelve.open(str(filename))
    # Check how many objects have been stored
    assert (
        len(my_shelf.keys()) == 1
    ), f"There are {len(my_shelf.keys())} objects in file {filename}. Expected 1."
    # Retrieve the single object
    df_info = list(my_shelf.values())[0]
    # Check if the object is a DataFrameWithInfo instance
    assert "DataFrameWithInfo" in str(df_info.__class__), (
        f"The object is not a DataFrameWithInfo "
        f"instance, but it is {df_info.__class__}"
    )
    my_shelf.close()
    return df_info


def export_df_with_info_to_file(
    df_info: DataFrameWithInfo, filename: str, overwrite: bool = False
):
    """
    This uses 'shelve' module to export the data of a previous DataFrameWithInfo instance
    to 'filename'
    @param df_info: DataFrameWithInfo instance that needs to be exported
    @param filename: Name/Path of the file where the data dump will be exported
    """
    if not overwrite:
        if os.path.exists(filename):
            raise FileExistsError(
                f"File {filename} already exists. If overwriting is not a problem, "
                f"set the 'overwrite' argument to True"
            )
    my_shelf = shelve.open(filename, "n")  # 'n' for new
    try:
        my_shelf["df_info"] = df_info
    except TypeError as e:
        logging.error(f"ERROR shelving: \n{e}")
    except KeyError as e:
        logging.error(f"Exporting data unsuccessful: \n{e}")

    my_shelf.close()


if __name__ == "__main__":
    df_sani_dir = os.path.join(
        "/home/lorenzo-hk3lab/WorkspaceHK3Lab/",
        "Partitioning",
        "data",
        "Sani_15300_anonym.csv",
    )

    metadata_cols = (
        "GROUPS	TAG	DATA_SCHEDA	NOME	ID_SCHEDA	COMUNE	PROV	MONTH	YEAR	BREED	SEX	AGE	"
        "SEXUAL STATUS	BODYWEIGHT	PULSE RATE	RESPIRATORY RATE	TEMP	BLOOD PRESS MAX	BLOOD "
        "PRESS MIN	BLOOD PRESS MEAN	BODY CONDITION SCORE	HT	H	DEATH	TIME OF DEATH	"
        "PROFILO_PAZIENTE	ANAMNESI_AMBIENTALE	ANAMNESI_ALIMENTARE	VACCINAZIONI	FILARIOSI	GC_SEQ"
    )
    metadata_cols = tuple(metadata_cols.replace("\t", ",").split(","))

    df_sani = DataFrameWithInfo(metadata_cols=metadata_cols, data_file=df_sani_dir)

    whole_word_replace_dict = {
        "---": None,
        ".": None,
        "ASSENTI": "0",
        "non disponibile": None,
        "NV": None,
        "-": None,
        "Error": None,
        #     '0%': '0'
    }

    char_replace_dict = {"°": "", ",": "."}

    # df_feat_analysis = DfFeatureAnalysis(df_sani, metadata_cols=metadata_cols)
    # # mixed_type_cols, numerical_col_list, str_col_list, other_col_list = df_feat_analysis.get_column_list_by_type()
    # df_feat_analysis.show_column_type_infos()
