import itertools
import logging
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from pd_extras.utils.dataframe_with_info import (
    DataFrameWithInfo,
    FeatureOperation,
    copy_df_info_with_new_df,
)
from pd_extras.utils.refactoring.feature_enum import (
    ENCODED_COLUMN_SUFFIX,
    EncodingFunctions,
    OperationTypeEnum,
)

logger = logging.getLogger(__name__)

NAN_CATEGORY = "Nan"
BIN_SPLIT_COL_SUFFIX = "_bin_id"


def convert_maps_from_tuple_to_str(group_id_to_tuple_map):
    """
    It gets a dictionary (with tuple values) and it converts the tuple
    values into strings and returns it as a dictionary
    """
    gr_id_to_string_map = {}
    for gr_id in group_id_to_tuple_map.keys():
        # Turn the group tuple into a string
        gr_id_to_string_map[gr_id] = "-".join(
            str(el) for el in group_id_to_tuple_map[gr_id]
        )
    return gr_id_to_string_map


def split_continuous_column_into_bins(
    df_info: DataFrameWithInfo, col_name, bin_threshold
):
    """
    This function adds a column to DataFrame df_info called "[col_name]_bin_id" where we split the "col_name" into bins
    :param df_info: DataFrameWithInfo -> DataFrameWithInfo instance containing the 'col_name' column to split
    :param col_name: String -> Name of the column to be split into discrete intervals
    :param bin_threshold: List -> It contains the thresholds used to separate different groups
                                  (the threshold will be included in the bin with higher values)
    :return: pd.DataFrame -> Same "df_info" passed with a new column with the bin_indices
                             which the column value belongs to
             Dict[List] -> Dictionary with the bin_indices as keys and bin_ranges as values
    """
    new_col_name = f"{col_name}{BIN_SPLIT_COL_SUFFIX}"
    # Initialize the bin <--> id_range map  with the min and max value
    bin_id_range_map = {}
    # For the BIN 0 choose the column minimum as the bin "lower_value",
    # in the other case the "upper_value" of the previous loops is set as "lower_value"
    lower_value = min(df_info.df[col_name].unique()) - 1
    # Loop over the bins (we need to increase by 1 because they are only the separating values)
    for i in range(len(bin_threshold) + 1):

        bin_id_range_map[i] = []
        # Append the bin upper and lower value to the "bin_id_range_map"
        # For the first and last bin, we set some special values
        bin_id_range_map[i].append(lower_value)

        # Assign the bin upper value:
        # 1. Either to the higher threshold
        # 2. Or to the column maximum value (if there is not a higher threshold in list)
        try:
            upper_value = bin_threshold[i]
        except IndexError:
            upper_value = max(df_info.df[col_name].unique())

        # Append the bin upper value to the "bin_id_range_map"
        bin_id_range_map[i].append(upper_value)

        # Identify the values in the range [lower_value, upper_value] in every row,
        # and assign them "i" as the value of the new column "_bin_id"
        df_info.df.loc[
            (df_info.df[col_name] >= lower_value)
            & (df_info.df[col_name] <= upper_value),
            new_col_name,
        ] = i

        # Set the upper_value as the lower_value for the next higher bin
        lower_value = upper_value

    # Cast the new column to int8
    df_info.df.loc[:, new_col_name] = df_info.df[new_col_name].astype("Int16")

    df_info.add_operation(
        FeatureOperation(
            original_columns=col_name,
            operation_type=OperationTypeEnum.BIN_SPLITTING,
            encoded_values_map=bin_id_range_map,
            derived_columns=new_col_name,
        )
    )

    return df_info


def combine_categorical_columns_to_one(
    df_info: DataFrameWithInfo, columns_list: Tuple[str], include_nan: bool = False
) -> Tuple[DataFrameWithInfo, str]:
    """
    This function generates and indexes the possible permutations of the unique values
    of the column list "col_names".
    Then it insert a new column into the df calculating for every row the ID corresponding
    to the combination of those columns_list (i.e. which combination of values the row belongs to).
    The map between the ID and the combination of values will be
    stored in df_info as detail of the FeatureOperation.

    Parameters
    ----------
    df_info: DataFrameWithInfo
    columns_list: Tuple[str]
    include_nan: bool

    Returns
    -------
    df_info: DataFrameWithInfo
        Same "df" passed with a new column that is the combination
        of "col_names" (separated by "-" and with suffix BIN_ID_COL_SUFFIX)
    new_column_name: str
        Name of the new column
    """
    # Define the name of the new column containing the combination of 'column_list' values
    new_column_name = f"{'-'.join([c for c in columns_list])}{ENCODED_COLUMN_SUFFIX}"

    # If the column has already been created, return the df_info
    if new_column_name in df_info.df.columns:
        logging.warning(
            f"The column {new_column_name} is already present in df_info argument. Maybe "
            f"a similar operation has already been performed. No new column has been "
            f"created to avoid overwriting."
        )
        return df_info, new_column_name

    # Get the unique values for every column in "col_names"
    col_unique_values = []
    for c in columns_list:
        if include_nan:
            unique_values_in_column = list(df_info.df[c].unique())
        else:
            # Remove NaN
            unique_values_in_column = [
                i for i in list(df_info.df[c].unique()) if str(i) != "nan"
            ]
        unique_values_in_column.sort()
        col_unique_values.append(unique_values_in_column)

    # Create the possible combinations (vector product) between the columns' values
    new_columns_encoding_maps = {}
    # Set the new column to NaN (then we fill in the appropriate values)
    df_info.df.loc[:, new_column_name] = np.nan
    for partit_id, combo in enumerate(itertools.product(*col_unique_values)):
        # Fill the encoding map to keep track of the link between the combination and the encoded value
        new_columns_encoding_maps[partit_id] = combo
        # Combine the boolean arrays to describe whether the row has the same values as the combination "combo"
        is_row_in_group_combo = np.logical_and.reduce(
            (
                [
                    df_info.df[columns_list[i]] == combo[i]
                    for i in range(len(columns_list))
                ]
            )
        )
        # Assign "i" to every row that has that specific combination of values in columns "col_names"
        df_info.df.loc[is_row_in_group_combo, new_column_name] = partit_id

    # Cast the ids from float64 to Int16 (capital 'I' to include NaN values)
    df_info.df.loc[:, new_column_name] = df_info.df[new_column_name].astype("Int16")
    # Track this operation in df_info
    df_info.add_operation(
        FeatureOperation(
            original_columns=columns_list,
            operation_type=OperationTypeEnum.FEAT_COMBOS_ENCODING,
            encoded_values_map=new_columns_encoding_maps,
            derived_columns=new_column_name,
        )
    )
    return df_info, new_column_name


def _one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    drop_one_new_column: bool = True,
    drop_old_column: bool = False,
):
    """
    OneHotEncoding of 'column' in df

    Parameters
    ----------
    df
    column
    drop_one_new_column
    drop_old_column

    Returns
    -------

    """
    df_new = df.copy()
    series = df_new[column].values
    series = series.reshape(-1, 1)
    # We choose to drop the first category of the feature (it can be deduced by the others ->
    #   it is just a combination of 0 of the other categories)
    if drop_one_new_column:
        encoder = OneHotEncoder(drop="first")
    else:
        encoder = OneHotEncoder()
    encoder_fitted = encoder.fit(series)
    transformed_cols = encoder_fitted.transform(series).toarray()
    encoded_categories = encoder_fitted.categories_[0].tolist()
    try:
        encoded_categories.remove(NAN_CATEGORY.title())
    except ValueError:
        logger.debug(f"No NaN values were found in column {column}")
    # Name the new columns after the categories (adding a suffix). Exclude the first which was dropped
    new_column_names = [
        f"{column}_{col}{ENCODED_COLUMN_SUFFIX}" for col in encoded_categories[1:]
    ]
    # Add the new encoded columns to the df_new
    for i, col in enumerate(new_column_names):
        df_new[col] = transformed_cols[:, i]
    # Drop the columns that has been encoded
    if drop_old_column:
        df_new = df_new.drop(column, axis=1)

    # Convert the encoded columns to boolean type (this pandas Dtype is for handling NaN values)
    df_new[new_column_names] = df_new[new_column_names].astype(pd.BooleanDtype())
    return df_new, encoder_fitted, new_column_names


def _ordinal_encode_column(df, column, drop_old_column: bool = False):
    """

    Parameters
    ----------
    df
    column
    drop_old_column

    Returns
    -------

    """
    df_new = df.copy()
    series = df_new[column].values
    # Adding a new dimension for use with OrdinalEncoder (1D array to column vector shape=(5,1))
    series = series[..., np.newaxis]
    encoder = OrdinalEncoder()
    encoder_fitted = encoder.fit(series)  # Encoder object (for reverse transformation)
    series_enc = encoder_fitted.transform(series)
    new_column = f"{column}{ENCODED_COLUMN_SUFFIX}"
    # Convert the encoded columns to Integer type (this pandas Dtype is for handling NaN values)
    df_new[new_column] = series_enc
    df_new[new_column] = df_new[new_column].astype(pd.Int16Dtype())

    if drop_old_column:
        df_new = df_new.drop(column, axis=1)
    return df_new, encoder_fitted, [new_column]


def encode_single_categorical_column(
    df_info: DataFrameWithInfo,
    col_name: str,
    encoding: EncodingFunctions = EncodingFunctions.ORDINAL,
    drop_one_new_column: bool = True,
    drop_old_column: bool = False,
    force: bool = False,
    case_sensitive: bool = False,
):
    """
    This function will encode the categorical column with the specified 'encoding' technique.
    If the column has already been encoded or it contains numerical values already,
    no operations will be performed and the input 'df_info' is returned (see 'force' argument).

    Notes
    -----
    The NAN_CATEGORY is a generic value to identify NaN values. These will be encoded as a
    category but the column (in OneHotEncoding) is automatically dropped inside the encoding function.
    The NaN values are restored as NaN after encoding for each values that was NaN originally.

    Parameters
    ----------
    df_info: DataFrameWithInfo
    col_name
    encoding
    drop_one_new_column
    drop_old_column
    force: bool
        This is to choose whether to force the encoding operation even if the column is numerical
        or it has already been encoded.
    case_sensitive

    Returns
    -------

    """
    # If the column has already been encoded and the new column has already been created, return df_info
    enc_column = df_info.get_enc_column_from_original(column_name=col_name)

    # Check if encoding operation is required
    if not force:
        if enc_column is not None:
            logging.warning(
                f"The column {col_name} has already been encoded "
                f'as "{enc_column}". No further operations are performed '
            )
            return df_info
        elif df_info.df[col_name].dtype.kind in "biufc":
            logging.warning(
                f"The column {col_name} is already numeric. No further operations are performed "
            )
            return df_info

    df_to_encode = df_info.df.copy()
    # Find index of rows with NaN and convert it to a fixed value so the corresponding encoded col will be dropped
    nan_serie_map = df_to_encode[col_name].isna()
    nan_serie_map = nan_serie_map.index[nan_serie_map].tolist()
    df_to_encode.loc[nan_serie_map][col_name] = NAN_CATEGORY.title()
    # Set to 'title' case so str with different capitalization are interpreted as equal
    if not case_sensitive:
        df_to_encode.loc[:, col_name] = df_to_encode[col_name].astype(str).str.title()

    # Encoding using the selected function
    if encoding == EncodingFunctions.ORDINAL:
        df_encoded, encoder, new_columns = _ordinal_encode_column(
            df_to_encode, column=col_name, drop_old_column=drop_old_column
        )
    elif encoding == EncodingFunctions.ONEHOT:
        df_encoded, encoder, new_columns = _one_hot_encode_column(
            df_to_encode,
            column=col_name,
            drop_one_new_column=drop_one_new_column,
            drop_old_column=drop_old_column,
        )
    else:
        logging.error(
            f"No valid encoding_func argument. Possible "
            f"values are: {[e.name for e in EncodingFunctions]}"
        )
        return None

    # Set the rows with missing values originally to NaN
    df_encoded.loc[nan_serie_map, col_name] = pd.NA
    df_encoded.loc[nan_serie_map, new_columns] = np.nan

    # Generate encoded values map
    encoded_values_map = {}
    for val_id, val in enumerate(encoder.categories_[0]):
        encoded_values_map[val_id] = val

    df_info_encoded = copy_df_info_with_new_df(df_info, df_encoded)

    df_info_encoded.add_operation(
        FeatureOperation(
            original_columns=col_name,
            operation_type=OperationTypeEnum.CATEGORICAL_ENCODING,
            encoder=encoder,
            encoded_values_map=encoded_values_map,
            derived_columns=tuple(new_columns),
        )
    )

    return df_info_encoded


def encode_multi_categorical_columns(
    df_info: DataFrameWithInfo,
    columns: Tuple = None,
    encoding: EncodingFunctions = EncodingFunctions.ORDINAL,
    drop_one_new_column: bool = True,
    drop_old_column: bool = False,
):
    """
    Encoding every categorical column in 'columns' argument into separate features by
    using 'encode_single_categorical_column'.

    Parameters
    ----------
    df_info
    columns
    encoding
    drop_one_new_column
    drop_old_column

    Returns
    -------

    """
    if columns is None:
        columns = df_info.column_list_by_type.str_categorical_cols
    else:
        # Check if the col_names are all bool cols
        columns = set(columns)
        df_categ_cols = df_info.column_list_by_type.categorical_cols
        if columns.intersection(df_categ_cols) != columns:
            logging.error(
                f'The columns from "col_names" argument are not all categorical. '
                f"Non-categorical columns are: {columns - df_categ_cols}"
            )

    # Converting categorical cols
    for col in columns:
        df_info = encode_single_categorical_column(
            df_info=df_info,
            encoding=encoding,
            col_name=col,
            drop_old_column=drop_old_column,
            drop_one_new_column=drop_one_new_column,
        )

    return df_info


def convert_features_from_bool_to_binary(
    df_info: DataFrameWithInfo, col_names: Tuple = None
):
    """
    Converting the boolean features from col_names argument
    @param df_info:
    @param col_names:
    @return:
    """

    if col_names is None:
        col_names = df_info.column_list_by_type.bool_cols
    else:
        # Check if the col_names are all bool cols
        col_names = set(col_names)
        df_bool_cols = df_info.column_list_by_type.bool_cols
        if col_names.intersection(df_bool_cols) != col_names:
            logging.error(
                f'The columns from "col_names" argument are not all bool. Non-bool columns are:'
                f"{col_names - df_bool_cols}"
            )
    # Converting from bool to binary
    for col in col_names:
        df_info.df[col] = df_info.df[col] * 1
    return df_info


def make_categorical_columns_multiple_combinations(
    df_info: DataFrameWithInfo, col_names
):
    """
    This function selects a number N of column from 1 to len(col_names).
    Then it combines the unique values of the first N columns from col_names in order to
    index the possible permutations of the unique values of those columns.
    - First element/column of partition cols is Level 1 (SEX -> M/F -> 0/1)
    - Second element/column combines its unique values with the ones from the first column to generate
        more possible combinations (e.g. SEXUAL STATUS -> I/NI * M/F -> (I,M)(NI,M)(I,F)(NI,F) ->
        values in new_column: 0,1,2,3  )
    - ....
    So each level will define many different groups (defined by a different combination of the
    possible values of one or more partition cols)

    @param df_input: DataFrameWithInfo containing the df
    @param col_names: List of columns that will be combined to each other
    :return: pd.DataFrame -> DataFrame with new columns with group IDs for different partitioning levels
             Dict[Dict[Tuple]] -> This contains:
                 - 1st level keys: name of col_names used to partition data
                 - 2nd level keys: ID of the combination
                 - tuple: combination of values of those columns (1st level keys)
    """
    combination_columns = []
    for i in range(len(col_names)):
        df_info, new_column = combine_categorical_columns_to_one(
            df_info, col_names[: i + 1]
        )
        combination_columns.append(new_column)

    return df_info, combination_columns


# def get_column_enc_categories():
#     pass


if __name__ == "__main__":
    sys.path.append("../..")
    import os

    CWD = os.path.abspath(os.path.dirname("__file__"))
    # DB_SMVET = os.path.join('/home/lorenzo-hk3lab/WorkspaceHK3Lab', 'smvet','data', 'Sani_15300_anonym.csv')
    # SEGMENTATION_DATA = os.path.join(CWD, '..', 'segmentation', 'resources', 'dense_areas_percentage.csv')
    DB_CORRECT = os.path.join(CWD, "..", "..", "data", "Sani_15300_anonym.csv")
    df_info = DataFrameWithInfo(metadata_cols=(), data_file=DB_CORRECT)
    print(df_info.df.columns)
    col = "SEX"
    df_info = encode_single_categorical_column(
        df_info, col_name=col, encoding=EncodingFunctions.ONEHOT
    )
    print("end")
