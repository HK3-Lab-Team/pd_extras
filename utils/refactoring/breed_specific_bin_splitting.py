import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from pd_extras.utils.dataframe_with_info import DataFrameWithInfo, FeatureOperation, \
    copy_df_info_with_new_df
from pd_extras.utils.refactoring.feature_enum import OperationTypeEnum

BREED_SPECIFIC_BIN_COLUMN_SUFFIX = '_bin_breed_specific'


def get_increasing_thresholds_for_bin_splitting(
        range_min: float, range_max: float,
        bin_thresh_increase: float, bin_count: int
) -> Tuple[List[float], List[float]]:
    """
    This splits the interval [range_min, range_max] in bins that increase by range_increase.
    So it returns the thresholds computed as:
    '''
    bin[i] = range_increase * bin[i-1]
    threshold[i] = threshold[i-1] + bin[i]
    '''
    The bin[0] is computed so that the final number of bins (following that rule) is 'bin_count' and it covers
    the whole range following the rules above.

    Parameters
    ----------
    range_min: float
        Minimum value of the range to split
    range_max: float
        Maximum value of the range to split
    bin_thresh_increase: float
        How much the next bin is increased w.r.t the previous bin (bin[i] = range_increase * bin[i-1])
    bin_count:
        Number of bins that you want to create

    Returns
    -------
    thresh_list: List
        List of thresholds that split the range between range_min and range_max
    bin_size_list: List
        List of bin sizes (i.e. threshold[i+1] - threshold[i]). It can be useful if reused in
        later computations (without recalculating)

    """
    tot_range = range_max - range_min
    # Set the first threshold as the range_min (minus small value so that range_min
    # is included in further computation)
    thresh_list = [range_min - range_min * 1e-10, ]
    # Check if the bins are supposed to be the same size, or not
    if bin_thresh_increase == 1:
        thresh_list.extend([tot_range / bin_count * (i + 1) + range_min for i in range(bin_count - 1)])
        bin_size_list = [tot_range / bin_count, ] * bin_count
    else:
        # Calculate the starting bin so that the bin_count can fit in range
        geometric_serie_sum = (1 - bin_thresh_increase ** bin_count) / (1 - bin_thresh_increase)
        new_bin_range = tot_range / geometric_serie_sum
        bin_size_list = [new_bin_range, ]
        # 'bin_count - 1' because the last threshold must correspond to 'range_max'
        for _ in range(bin_count - 1):
            # New threshold
            new_bin_thresh = new_bin_range + thresh_list[-1]
            thresh_list.append(new_bin_thresh)
            # Compute new bin_size by increasing previous one
            new_bin_range = bin_thresh_increase * bin_size_list[-1]
            bin_size_list.append(new_bin_range)
    # Overwrite the last bin so it ends at range_max
    bin_size_list[-1] = range_max - thresh_list[-1]
    thresh_list.append(range_max)

    return thresh_list, bin_size_list


def get_bin_upp_low_value_from_thresholds(extra_bin_size: float, thresh_list: Tuple,
                                          bin_sizes: Tuple = None) -> List[Tuple[float, float]]:
    """
    The function will compute age bins adding some extra values (overlapped) in every age bin
    in order to limit threshold effects.
    E.g. extra_bin_size = 0.5 -> The bin used for the samples in range
        [threshold[i], threshold[i+1]] will be:
            [ (threshold[i] - bin_size[i-1] * 0.5)  ;  (threshold[i+1] + bin_size[i+1] * 0.5) ]

    Parameters
    ----------
    extra_bin_size: float
        This indicates how much extra values will be considered in the age_bin. This is the
        portion of next/previous bin that will be included in the bin. If set to 0, the age bin
        will be calculated as interval between two thresholds [threshold[i], threshold[i+1]].
    thresh_list: Tuple
        Tuple of the age bin thresholds (that separate the age bins). They must include minimum and
        maximum value if you want to include all the data
    bin_sizes: Tuple[Tuple]
        These are just the intervals between two thresholds [threshold[i], threshold[i+1]].
        If not provided, they will be computed based on thresh_list. This is for performances sake.
        because when provided this computation will not be required (it may have been computed during
        bin_thresholds computation by '_split_range_increasing_thresholds' function. Default set to None.

    Returns
    -------
    bins_list: List[Tuple[float, float]]
        List of the bins. Each element is a tuple with the lower and upper value
    """
    if bin_sizes is None:
        bin_sizes = []
        for i in range(len(thresh_list) - 1):
            bin_sizes.append(thresh_list[i + 1] - thresh_list[i])
    # Add first bin
    bins_list = [(thresh_list[0], thresh_list[1] + bin_sizes[0] * extra_bin_size), ]
    for i in range(len(thresh_list) - 3):
        bin_min = thresh_list[i + 1] - bin_sizes[i] * extra_bin_size
        bin_max = thresh_list[i + 2] + bin_sizes[i + 2] * extra_bin_size
        bins_list.append((bin_min, bin_max))
    # Add the last bin with very large number as upper value in order to include every older
    # patients from future dataset (test_set)
    bins_list.append((thresh_list[-2] - bin_sizes[-2] * extra_bin_size, thresh_list[-1] + 1))
    return bins_list


def get_bin_list_per_single_breed(df_breed: pd.DataFrame, column_to_split: str,
                                  bin_thresh_increase: float = 1.1, bin_count: int = 20,
                                  mongrels_age_bins: Tuple = (), start_from_zero: bool = False,
                                  bin_thresholds: Tuple[float] = None, sample_count_threshold: int = 20
                                  ) -> Tuple[List, Dict]:
    """
    This function computes threshold values for bin splitting for a single breed (df_breed). Then it uses these to
    compute the bin upper and lower values.

    Parameters
    ----------
    df_breed: pandas.DataFrame
        Slice of the original DataFrame which contains the samples from a specific breed
    column_to_split: str
        Name of the column whose values we want to split into bins
    mongrels_age_bins:
        These are the age_bins computed for mongrel breed and they will be used for the
        less populated breeds so the age range is more reliable
    bin_thresh_increase: float
        How much the next bin is increased w.r.t the previous bin (bin[i] = range_increase * bin[i-1])
    bin_count:
        Number of bins that you want to create
    start_from_zero: bool
        Option to force the age range to start from 0 and not from the 'column_to_split' minimum value
    bin_thresholds: Tuple[float]
        This is an optional argument to manually force some thresholds for the age bin splitting. If provided,
        the bins are calculated based on these (bin_size_increase is still required). A bin will be associated
        only to the data points that are in the range between the first value and the last value
    sample_count_threshold: int
        When computing the range of the age values, some breeds may have too few samples to make a
        reliable range estimation. So this value indicates the sample count threshold under which the mongrel's
        age range will be used.

    Returns
    -------
    List
        List of bin ranges computed
    Dict
        Map from the bin_id to related range of values in bin
    """
    # If there are very few samples per breed, use mongrels age range
    if df_breed.shape[0] < sample_count_threshold:
        if mongrels_age_bins == ():
            raise ValueError(f"No mongrel_age_bins argument provided, but the breed "
                             f"{df_breed['BREED'].unique()[0]} has too few samples (<{sample_count_threshold})"
                             f" so the age range computed would not be reliable")
        else:
            column_bin_to_range_map = {bin_id: bin_range for (bin_id, bin_range) in enumerate(mongrels_age_bins)}
            return mongrels_age_bins, column_bin_to_range_map
    else:
        # If bin_thresholds are manually forced, do not compute and use them
        if bin_thresholds is None:
            if start_from_zero:
                range_min = 0
            else:
                range_min = df_breed[column_to_split].min()
            bin_thresholds, bin_sizes = get_increasing_thresholds_for_bin_splitting(
                range_min=range_min,
                range_max=df_breed[column_to_split].max(),
                bin_thresh_increase=bin_thresh_increase,
                bin_count=bin_count
            )
        else:
            bin_sizes = None
        bins_list = get_bin_upp_low_value_from_thresholds(
            extra_bin_size=0., thresh_list=bin_thresholds,
            bin_sizes=bin_sizes
        )
        column_bin_to_range_map = {bin_id: bin_range for (bin_id, bin_range) in enumerate(bins_list)}
        return bins_list, column_bin_to_range_map


def create_df_with_overlapping_bins_single_breed(df_breed: pd.DataFrame, column_to_split, bins_list,
                                                 new_column_name) -> pd.DataFrame:
    """
    It splits the samples from DataFrame 'df_breed' into age bins according to bins_list definition.
    The bins_list may define overlapping bins and in this case overlapping rows from different age bins are
    repeated so that a groupby function can be applied

    Parameters
    ----------
    df_breed
    column_to_split: str
        Name of the column whose values we want to split into bins
    bins_list
    new_column_name

    Returns
    -------

    """
    overlapped_df = pd.DataFrame()
    col_array = df_breed[column_to_split].values
    # Instantiate an empty DataFrame with the new column added
    for id_bin, bin_range in enumerate(bins_list):
        # Select the df_info rows in the age bin
        range_samples_bool_map = np.logical_and(np.greater_equal(col_array, bin_range[0]),
                                                np.less_equal(col_array, bin_range[1]))

        single_bin_df = df_breed.loc[range_samples_bool_map, :].copy()
        # Associate to bin_df slice the appropriate index of the age_bin it belongs to
        single_bin_df.loc[:, new_column_name] = int(id_bin)
        overlapped_df = overlapped_df.append(single_bin_df, ignore_index=True)
        # overlapped_df.loc[ , new_column_name] = int(id_bin)
    overlapped_df = overlapped_df.reset_index(drop=True)
    # Convert to appropriate format
    overlapped_df[new_column_name] = overlapped_df[new_column_name].astype('Int16')
    return overlapped_df


def add_column_bins_to_single_breed(df_breed: pd.DataFrame, column_to_split, bins_list,
                                    new_column_name) -> pd.DataFrame:
    """
    It splits the samples from DataFrame 'df_breed' into age bins according to bins_list definition.
    The bins_list may define overlapping bins and in this case overlapping rows from different age bins are
    repeated so that a groupby function can be applied

    Parameters
    ----------
    df_breed
    column_to_split: str
        Name of the column whose values we want to split into bins
    bins_list
    new_column_name

    Returns
    -------

    """
    # TODO: Check if bins list contains overlapping bins. In that case we need to create a new DataFrame
    col_array = df_breed[column_to_split].values
    # Instantiate an empty DataFrame with the new column added
    for id_bin, bin_range in enumerate(bins_list):
        # Select the df_info rows in the age bin
        range_samples_bool_map = np.logical_and(np.greater_equal(col_array, bin_range[0]),
                                                np.less_equal(col_array, bin_range[1]))
        # Associate to bin_df slice the appropriate index of the age_bin it belongs to
        df_breed.loc[range_samples_bool_map, new_column_name] = int(id_bin)
    # Convert to appropriate format
    df_breed[new_column_name] = df_breed[new_column_name].astype('Int16')
    return df_breed


def _apply_compute_age_bin_per_breed(df_breed: pd.DataFrame, column_to_split: str,
                                     new_column_name: str,
                                     column_bin_to_range_map_per_breed: Dict, bin_thresh_increase: float = 1.1,
                                     bin_count: int = 20, mongrels_age_bins: Tuple = (),
                                     bin_thresholds: Tuple[float] = None, sample_count_threshold: int = 20,
                                     start_from_zero: bool = False) -> pd.DataFrame:
    """
    This function is supposed to be used in a '.apply()' function from 'add_breed_specific_age_bin'
    in order to compute age_bin for each breed (df_breed) of the original DataFrame

    Parameters
    ----------
    df_breed: pandas.DataFrame
        Slice of the original DataFrame which contains the samples from a specific breed
    column_to_split: str
        Name of the column whose values we want to split into bins
    new_column_name: str
        Name of the new column that will contain the new bin identifiers (the encoded values)
    column_bin_to_range_map_per_breed: Dict
        This Dict will contain the map between the breed + related bin_ids and the range of values in bin.
        This is populated through the function calls by .apply() so that each breed is mapped.
    mongrels_age_bins:
        These are the age_bins computed for mongrel breed and they will be used for the
        less populated breeds so the age range is more reliable
    bin_thresh_increase: float
        How much the next bin is increased w.r.t the previous bin (bin[i] = range_increase * bin[i-1])
    bin_count:
        Number of bins that you want to create
    bin_thresholds: Tuple[float]
        This is an optional argument to manually force some thresholds for the age bin splitting. If provided,
        the bins are calculated based on these (bin_size_increase is still required). A bin will be associated
        only to the data points that are in the range between the first value and the last value
    sample_count_threshold: int
        When computing the range of the age values, some breeds may have too few samples to make a
        reliable range estimation. So this value indicates the sample count threshold under which the mongrel's
        age range will be used.
    start_from_zero: bool
        Option to force the age range to start from 0 and not from the 'column_to_split' minimum value

    Returns
    -------
    pandas.DataFrame
        The same df_breed data with an additional column containing the breed_specific_age_bin
    """
    breed_age_bins, column_bin_to_range_map = get_bin_list_per_single_breed(
        df_breed=df_breed,
        column_to_split=column_to_split,
        bin_thresh_increase=bin_thresh_increase,
        bin_count=bin_count,
        bin_thresholds=bin_thresholds,
        sample_count_threshold=sample_count_threshold,
        mongrels_age_bins=mongrels_age_bins,
        start_from_zero=start_from_zero
    )
    breed = df_breed.name
    column_bin_to_range_map_per_breed[breed] = column_bin_to_range_map
    before_count = df_breed['AGE'].count()
    df_breed = add_column_bins_to_single_breed(
        df_breed, column_to_split=column_to_split,
        bins_list=breed_age_bins, new_column_name=new_column_name
    )
    if df_breed['AGE'].count() != before_count:
        print(f"Before: {before_count}\tAfter: {df_breed['AGE'].count()}")
    return df_breed

def _get_samples_with_breed_not_nan(df: pd.DataFrame, ) -> Tuple[bool, pd.DataFrame, Tuple]:
    """

    Find the samples with NaN breed values and drop them (they will be reinserted after computation)

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    Tuple[bool, pd.DataFrame, Tuple]:
        - contains_na_breed_samples: bool
            This is to tell if any samples with no breed defined were found
        - not_na_df: pd.DataFrame
            Same DataFrame as df argument, but wih no samples with no breed defined
        - na_breed_samples_ids: Tuple
            List of ids of the samples with no breed in initial 'df' argument
            (in order to get reinserted after)
    """
    breed_na_samples_bool_map = df['BREED'].isna()
    breed_na_count = np.sum(breed_na_samples_bool_map)
    if breed_na_count != 0:
        logging.warning(f"There are {breed_na_count} samples with no Breed values. No computation is "
                        f"possible for these samples")
        contains_na_breed_samples = True
        not_na_df = df.loc[np.logical_not(breed_na_samples_bool_map)]
        na_breed_samples_ids = df.loc[breed_na_samples_bool_map].index
        return contains_na_breed_samples, not_na_df, na_breed_samples_ids
    else:
        contains_na_breed_samples = False
        not_na_df = df
        return contains_na_breed_samples, not_na_df, ()

def add_breed_specific_bin_id_to_df(df_info: DataFrameWithInfo, column_to_split: str,
                                    new_column_name: str, bin_thresh_increase: float = 1.1,
                                    bin_count: int = 20, bin_thresholds: Tuple[float] = None,
                                    sample_count_threshold: int = 20, start_from_zero: bool = False
                                    ) -> Tuple[DataFrameWithInfo, Dict[str, Dict]]:
    """
    This function adds an extra column containing the bin identifier for the 'column_to_split', 
    computed as follows.
    First, the df_info is split according to breed. Then for each breed we look for the range of
    age values and we split the range into bins according to the arguments provided.
    We finally associate the row to the age_bin id according to its age value

    Parameters
    ----------
    df_info: DataFrameWithInfo
        Input Data we use to compute age_bin_ids
    column_to_split: str
        Name of the column whose values we want to split into bins
    new_column_name: str
        Name of the newly created column that will contain bin id
    bin_thresh_increase: float
        How much the next bin is increased w.r.t the previous bin (bin[i] = range_increase * bin[i-1])
    bin_count:
        Number of bins that you want to create
    bin_thresholds: Tuple[float]
        This is an optional argument to manually force some thresholds for the age bin splitting. If provided,
        the bins are calculated based on these (bin_size_increase is still required). A bin will be associated
        only to the data points that are in the range between the first value and the last value.
        The thresholds will be used as the lowest value of the bin.
    sample_count_threshold: int
        When computing the range of the age values, some breeds may have too few samples to make a
        reliable range estimation. So this value indicates the sample count threshold under which the mongrel's
        age range will be used.
    start_from_zero: bool
        Option to force the age range to start from 0 and not from the 'column_to_split' minimum value

    Returns
    -------
    DataFrameWithInfo
        New DataFrame with the additional column. The new column is defined as global variable AGE_BIN_COLUMN_BREED
    Dict[str, Dict]
        Mapping from (breed, bin id) to related range of 'column_to_split' values
    """
    contains_na_breed_samples, not_na_df, na_breed_samples_ids = \
        _get_samples_with_breed_not_nan(df_info.df)
    # Compute the bins from the most populated breed so that the least populated can use these
    # bins when lacking infos about the range
    mongrels_age_bins, _ = get_bin_list_per_single_breed(
        df_breed=not_na_df[not_na_df['BREED'] == 'MONGREL'],
        bin_thresh_increase=bin_thresh_increase,
        column_to_split=column_to_split,
        bin_count=bin_count,
        bin_thresholds=bin_thresholds,
        sample_count_threshold=sample_count_threshold,
        start_from_zero=start_from_zero
    )
    column_bin_to_range_map_per_breed = {}
    # For each breed compute the appropriate age_bins and add the new column AGE_BIN_COLUMN_BREED
    # containing the age_bins that the row belongs to
    df_with_bin_column = not_na_df.groupby('BREED').apply(
        _apply_compute_age_bin_per_breed,
        column_to_split=column_to_split,
        new_column_name=new_column_name,
        column_bin_to_range_map_per_breed=column_bin_to_range_map_per_breed,
        mongrels_age_bins=mongrels_age_bins,
        bin_thresh_increase=bin_thresh_increase,
        bin_count=bin_count,
        bin_thresholds=bin_thresholds,
        sample_count_threshold=sample_count_threshold,
        start_from_zero=start_from_zero
    )
    # Reinsert the samples with NaN breed values
    if contains_na_breed_samples:
        df_with_bin_column = df_with_bin_column.append(df_info.df.iloc[na_breed_samples_ids])
    df_with_bin_column.reset_index(drop=True, inplace=True)
    # Create DataFrameWithInfo with same instance attribute as df_info, but with the new bin_column
    age_bin_df_info = copy_df_info_with_new_df(df_info=df_info, new_pandas_df=df_with_bin_column)
    age_bin_df_info.add_operation(FeatureOperation(
        operation_type=OperationTypeEnum.BIN_SPLITTING,
        original_columns=column_to_split,
        derived_columns=new_column_name,
        encoded_values_map=column_bin_to_range_map_per_breed,
    ))
    return age_bin_df_info, column_bin_to_range_map_per_breed
