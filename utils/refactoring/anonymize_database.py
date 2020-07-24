import pandas as pd
import numpy as np
import os
import random
import string
import hashlib

def add_nonce_func(string_array):
    """
    This function takes an array of strings passed as "string_array" and
    attaches them nonces (random prefix and suffix), using Vectorization.
    
    :param cols_values: This is a list of numpy arrays, i.e. the columns we add nonce to
    :return: np.array of strings with nonces
    """
    return ''.join(random.choice(string.hexdigits) for i in range(12))\
            + string_array \
            + (''.join(random.choice(string.hexdigits) for i in range(12)))


def add_id_owner_col(private_df, cols_to_hash):
    """
    This function uses the columns of the "private_df" database to generate an hash value 
    and it creates an "ID_OWNER" column with those values.
    To generate hash values, we add nonces (random prefix and suffix) to the column values and then we use "sha256".
    See https://medium.com/luckspark/hashing-pandas-dataframe-column-with-nonce-763a8c23a833 for more info.
    
    :param private_df: Pandas.DataFrame with the owner's private data
    :param cols_to_hash: This is a list of column names with the infos we want to hash
    
    :return: Pandas.DataFrame similar to "private_df" with a new "ID_OWNER" column
    """
    # Turn rows into strings to be used 
    rows_into_strings = np.sum(np.array([private_df[c].values for c in cols_to_hash]), axis = 0)
    # Create a string with nonces --> Vectorization with Numpy Arrays
    private_df['HASH_NONCES'] = add_nonce_func(rows_into_strings)
    
    # Use "sha256" to hash the "HASH_NONCES" column
    hash_lambda = lambda x: hashlib.sha256(str.encode(str(x['HASH_NONCES']))).hexdigest()
    private_df['ID_OWNER'] = private_df.apply(hash_lambda, axis = 1)
    
    # Delete "HASH_NONCES" column
    private_df = private_df.drop('HASH_NONCES', 1)
    
    return private_df

def create_private_info_db(df, private_cols_to_map):
    """
    This function creates a Pandas.DataFrame where you will store all the owner's 
    private data needed to identify them. 
    These informations are listed in "private_cols_to_map" argument.
    
    :param df: Pandas.DataFrame that we will anonymize
    :param private_cols_to_map: This is a list of the columns that will be stored in the 
    private_db that will be returned, along with the new "ID_OWNER"
    :return: Pandas.DataFrame with the values of the "private_cols_to_map" and their hashed value in the column "ID_OWNER"
    """
    # Create the private_db with the columns with private infos only
    private_df = df[private_cols_to_map]
    
    # Get unique combinations of the columns you chose
    private_df = private_df.groupby(private_cols_to_map, as_index=False, group_keys=False).size().reset_index()
    
    # Eliminate size column
    private_df = private_df.drop(columns=[0])
    
    # Add the ID_OWNER column with the hash value of the row 
    private_df = add_id_owner_col(private_df, private_cols_to_map)
    
    return private_df

def anonymize_data(df, file_name, private_cols_to_remove, private_cols_to_map, dest_path):
    """
    This function will take the Pandas DataFrame "df" and it will return two files written inside the "dest_path":
    1. One file (called "[file_name]_anonym") will contain the database "df" where 
    we replaced the columns "private_cols_to_remove" with the column "ID_OWNER"
    2. Another file (called "[file_name]_private_info") will contain only the 
    owner infos "private_cols_to_map", which we associated an ID_OWNER to. 
    The ID_OWNER will be hashed using SHA256.
    
    :param df: Pandas.DataFrame that we will anonymize
    :param file_name: Name of the database we are working on (no ".csv" suffix). Used as prefix when saving csv output files.
    :param private_cols_to_remove: Columns that will be removed from "_anonym" file
    :param private_cols_to_map: Columns of the "_private_info" files 
    :param dest_path: The directory where we will save the two files
    
    :return: [file_name]_anonym : pd.DataFrame
             [file_name]_private_info : pd.DataFrame
    """
    # Fill NaN values in the columns we will map, to make DataFrame merge easier
    df[private_cols_to_map] = df[private_cols_to_map].fillna('----')
    # Create the "_private_info" db which will contain the map to owner's private data
    private_df = create_private_info_db(df, private_cols_to_map)

    # Create the "_anonym" DataFrame which will contain the anonymized database
    anonym_df = pd.DataFrame(df_sani)
    
    # Merge to insert the new ID_OWNER column
    anonym_df = anonym_df.merge(private_df)

    # Delete the columns with private owner's data
    anonym_df = anonym_df.drop(private_cols_to_remove, axis=1)
    
    # Write the two DataFrames to CSV files
    try:
        private_df.to_csv(os.path.join(dest_path,f'{file_name}_private_info.csv'), mode = 'w+', index=False)
        anonym_df.to_csv(os.path.join(dest_path,f'{file_name}_anonym.csv'), mode = 'w+', index=False)
    except FileNotFoundError:
        print("FileNotFoundError: The destination path was not found")
    
    return anonym_df, private_df


if __name__ == "__main__":
    
    private_cols_to_map = ['CLIENTE', 'INDIRIZZO', 'CIVICO', 'COMUNE', 'PROV']
    private_cols_to_remove = ['CLIENTE', 'INDIRIZZO', 'CIVICO']
    
    df_sani_dir = os.path.join(os.getcwd(), 'data', 'Sani 15300.csv')
    output_data_dir = os.path.join(os.getcwd(), 'data')
    df_sani = pd.read_csv(df_sani_dir)
    
    anonymize_data(df_sani, "Sani_15300", private_cols_to_remove, private_cols_to_map, output_data_dir)