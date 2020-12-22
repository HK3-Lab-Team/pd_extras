# encoding: utf-8

import hashlib
import os
import random
import string
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


def add_nonce_func(
    string_array: Union[str, int, float, np.array]
) -> Union[str, int, float, np.array]:
    """
    Add random prefix and suffix to an array of strings ``string_array``

    This function takes an array of strings passed as ``string_array`` and
    attaches nonces (random prefix and suffix) to each string.
    It can also be used in a vectorized way
    Prefix and suffix will contain 12 random characters each.

    Parameters
    ----------
    string_array: Union[str, int, float, np.array]
        This can be a number, a string or a numpy array of values
        (e.g. a DataFrame column)

    Returns
    -------
    np.array
        Array of strings with nonces
    """
    return (
        "".join(random.choice(string.hexdigits) for i in range(12))
        + string_array
        + ("".join(random.choice(string.hexdigits) for i in range(12)))
    )


def add_id_owner_col(
    private_df: pd.DataFrame, cols_to_hash: Tuple[str]
) -> pd.DataFrame:
    """
    This function uses the columns of the "private_df" database to generate an hash
    value and it creates an "ID_OWNER" column with those values.
    To generate hash values, the function adds nonces (random prefix and suffix)
    to the column values and then we use "sha256". See
    https://medium.com/luckspark/hashing-pandas-dataframe-column-with-nonce-763a8c23a833
    for more info.

    Parameters
    ----------
    private_df: pd.DataFrame
        DataFrame with the owner's private data
    cols_to_hash: Tuple[str]
        This is a list of column names with the infos we want to hash

    Returns
    -------
    pd.DataFrame
        DataFrame similar to ``private_df`` with a new "ID_OWNER" column
    """
    # Turn rows into strings to be used
    rows_into_strings = np.sum(
        np.array([private_df[c].values for c in cols_to_hash]), axis=0
    )
    # Create a string with nonces --> Vectorization with Numpy Arrays
    private_df["HASH_NONCES"] = add_nonce_func(rows_into_strings)

    # Use "sha256" to hash the "HASH_NONCES" column
    def hash_lambda(owner_name):
        return hashlib.sha256(str.encode(str(owner_name["HASH_NONCES"]))).hexdigest()

    private_df["ID_OWNER"] = private_df.apply(hash_lambda, axis=1)

    # Delete "HASH_NONCES" column
    private_df = private_df.drop("HASH_NONCES", 1)

    return private_df


def create_private_info_db(
    df: pd.DataFrame, private_cols_to_map: Tuple[str]
) -> pd.DataFrame:
    """
    Create a DataFrame with private data and a unique ID.

    This function will store in a DataFrame all the owner's private data
    contained in the columns ``private_cols_to_map`` needed to identify them.
    The function will also add a unique owner ID (in the column "OWNER_ID") that
    is hashed based on ``private_cols_to_map``.
    In case there are multiple rows with the same private info
    (e.g.: multiple data from the same customer), only one of those rows
    is included in the returned DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that we will anonymize
    private_cols_to_map: Tuple[str]
        List of the columns that will be stored in the private_db
        that will be returned, along with the new "ID_OWNER"

    Returns
    -------
    pd.DataFrame
        DataFrame with the values of the ``private_cols_to_map`` and
        their hashed value in the column "ID_OWNER"
    """
    # Create the private_db with the columns with private info only
    private_df = df[private_cols_to_map]

    # In case there are multiple rows with the same private info
    # (e.g.: multiple data from the same customer), only one of these rows
    # should be included in ``private_df``
    private_df.drop_duplicates(inplace=True)

    # Add the ID_OWNER column with the hash value of the row
    private_df = add_id_owner_col(private_df, private_cols_to_map)

    return private_df


def anonymize_data(
    df: pd.DataFrame,
    file_name: str,
    private_cols_to_remove: Tuple[str],
    private_cols_to_map: Tuple[str],
    dest_path: Union[Path, str],
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate generic from private data leaving a unique ID as map between them.

    This function will take the Pandas DataFrame ``df`` and it will return two
    files written inside the ``dest_path`` directory:
    1. One file (called "[file_name]_anonym") will contain the database ``df`` where
    we replaced the columns ``private_cols_to_remove`` with the column "ID_OWNER"
    2. Another file (called "[file_name]_private_info") will contain only the
    owner infos ``private_cols_to_map``, which we associated an ID_OWNER to.
    To generate hash values for the "ID_OWNER" column values, the algorithm
    adds nonces (random prefix and suffix) to the column values and then
    it uses "SHA256" algorithm.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that we will anonymize
    file_name: str
        Name of the database we are working on (no ".csv" suffix). Used as
        prefix when saving csv output files.
    private_cols_to_remove: Tuple[str]
        Columns that will be removed from "_anonym" file
    private_cols_to_map: Tuple[str]
        Columns of the "_private_info" files
    dest_path: Union[Path, str]
        The directory where we will save the two files
    random_seed: int
        Integer value used as "seed" for the generation of random prefixes and
        suffixes in "nonces".

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the private info ``private_cols_to_map``,
        along with another column "ID_OWNER" that allows to map these private
        informations to the data in the other DataFrame. This file is
        also saved to "[``dest_path``] / [``file_name``]_private_info.csv" file.
    pd.DataFrame
        DataFrame containing the same infos as the DataFrame ``df``, but
        the columns "private_cols_to_remove" have been replaced by "ID_OWNER"
        column.
        This file is also saved to "[``dest_path``] / [``file_name``]_anonym.csv"
        file.
    """
    # Fix the random seed for the generation of random prefixes and
    # suffixes in "nonces", used for creating "ID_OWNER" column.
    random.seed(random_seed)
    # Create the "_anonym" DataFrame which will contain the anonymized database
    anonym_df = df.copy()
    # Fill NaN values in the columns we will map, to make DataFrame merge easier
    df[private_cols_to_map] = df[private_cols_to_map].fillna("----")
    # Create the "_private_info" db which will contain the map to owner's private data
    private_df = create_private_info_db(df, private_cols_to_map)

    # Merge to insert the new ID_OWNER column corresponding to the
    # private column value combinations
    anonym_df = anonym_df.merge(private_df)

    # Delete the columns with private owner's data
    anonym_df = anonym_df.drop(private_cols_to_remove, axis=1)

    # Write the two DataFrames to CSV files
    private_df.to_csv(
        os.path.join(dest_path, f"{file_name}_private_info.csv"),
        mode="w+",
        index=False,
    )
    anonym_df.to_csv(
        os.path.join(dest_path, f"{file_name}_anonym.csv"), mode="w+", index=False
    )

    return anonym_df, private_df
