import datetime
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .datasim import TestDataSet, from_tuples
from .datasim_util import (
    ChangeColumnDType,
    Compose,
    InsertNaNs,
    InsertNewValues,
    InsertOutOfScaleValues,
    InsertSubstringsByIndex,
    ReplaceSubstringsByValue,
    SubstringReplaceMapByIndex,
    SubstringReplaceMapByValue,
)

WHOLE_WORD_REPLACE_DICT = {
    "---": pd.NA,
    ".": pd.NA,
    "ASSENTI": pd.NA,
    "PRESENTI": pd.NA,
    "non disponibile": pd.NA,
    "NV": pd.NA,
    "-": pd.NA,
    "Error": pd.NA,
    "None": pd.NA,
    "NAN": pd.NA
    #     '0%': '0'
}
CHAR_REPLACE_DICT = {"°": "", ",": "."}


def _create_datetime_formatted_list(sample_size: int, datetime_format: str):
    date_list = []
    for i in range(sample_size):
        date_element = datetime.datetime(
            year=2015, month=7, day=2
        ) + datetime.timedelta(days=i)
        date_element = date_element.strftime(datetime_format)
        date_list.append(date_element)

    return date_list


def _int_data_list(sample_size: int, step: int = 1, bias: int = 0):
    return [step * i + bias for i in range(sample_size)]


def _float_data_list(sample_size: int, step: float, bias: float = 0):
    return [step * i + bias for i in range(sample_size)]


def _string_data_list(sample_size: int, prefix: str = "value"):
    return [f"{prefix}_{i}" for i in range(sample_size)]


def _categ_data_list(sample_size: int, categories_count: int, prefix: str = ""):
    """
    Create list of categorical data

    The function creates a list with ``sample_size`` elements that are equally split
    into ``categories_count`` subgroups that have the same repeated value (i.e.
    the category).

    Parameters
    ----------
    sample_size : int
        Number of elements in the resulting list
    categories_count : int
        Number of different categories contained in the resulting list
    prefix : str, optional
        String that will be the prefix for each category. If set to "", the
        categorical values will be integers. Default set to "".

    Returns
    -------
    List
        List of ``sample_size`` categorical samples.
    """
    samples_per_category = sample_size // categories_count
    data_list = []
    for cat in range(categories_count - 1):
        data_list.extend(
            [cat if prefix == "" else f"{prefix}{cat}"] * samples_per_category
        )
    data_list.extend(
        [(categories_count - 1) if prefix == "" else f"{prefix}{categories_count-1}"]
        * (sample_size - (categories_count - 1) * samples_per_category)
    )
    return data_list


def _bool_data_list(sample_size: int, true_sample_count: int):
    return [True] * true_sample_count + [False] * (sample_size - true_sample_count)


def _create_ideal_test_dataset(sample_size: int) -> TestDataSet:
    """
    Create an ideal dataset with no NaNs or typos and many different types of features

    The TestDataSet created has ``sample_size`` samples and 35 features with
    different type values:
    - float, int
    - categorical string and int
    - NaN only, mostly NaN (all NaN but 5 string values)
    - same string/int repeated value
    - mixed values (str+float, str+int, int+float)
    - datetime, date, year only
    There are 2 columns for each of these column types (one is supposed a
    `metadata_column` and the other is a not), but only the non-metadata version is
    present for the following value types:
    - big float, small float

    Parameters
    ----------
    sample_size: int
        Number of samples in the returned dataset

    Returns
    -------
    TestDataSet
        TestDataSet instance containing ``sample_size`` samples and 35 features with
        different types
    """
    one_third_samples = sample_size // 3
    one_half_samples = sample_size // 2
    big_float_step = 100000
    big_float = sample_size * big_float_step

    column_tuples = [
        # Metadata columns
        ("metadata_int_col_0", _int_data_list(sample_size)),
        ("metadata_int_col_1", _int_data_list(sample_size, step=10)),
        ("metadata_float_col_0", _float_data_list(sample_size, step=0.0001)),
        ("metadata_str_col_0", _string_data_list(sample_size, prefix="meta1")),
        ("metadata_str_col_1", _string_data_list(sample_size, prefix="meta2")),
        (
            "metadata_str_categ_col",
            ["value0"] * one_half_samples
            + ["value1"] * (sample_size - one_half_samples),
        ),
        (
            "metadata_num_categ_col",
            [1] * one_third_samples
            + [2] * one_third_samples
            + [3] * (sample_size - 2 * one_third_samples),
        ),
        ("metadata_nan_col", [pd.NA] * sample_size),
        # WARNING: sample_size should be higher than 5
        ("metadata_mostly_nan_col", [pd.NA] * (sample_size - 5) + ["samevalue"] * 5),
        ("metadata_samenum_col", [0] * sample_size),
        ("metadata_samestr_col", ["samevalue"] * sample_size),
        (
            "metadata_float_str_mixed_col",
            _string_data_list(one_half_samples)
            + _float_data_list(sample_size - one_half_samples, step=0.5),
        ),
        (
            "metadata_int_float_mixed_col",
            _int_data_list(one_half_samples, step=2)
            + _float_data_list(sample_size - one_half_samples, step=23.1),
        ),
        (
            "metadata_int_str_mixed_col",
            _string_data_list(one_half_samples)
            + _int_data_list(one_half_samples, step=2),
        ),
        (
            "metadata_datetime_col",
            _create_datetime_formatted_list(
                sample_size, datetime_format="%Y-%m-%d %H:%M:%S"
            ),
        ),
        (
            "metadata_date_col",
            _create_datetime_formatted_list(sample_size, datetime_format="%d/%m/%YYYY"),
        ),
        (
            "metadata_onlyyear_col",
            _create_datetime_formatted_list(sample_size, datetime_format="%YYYY"),
        ),
        # Feature Columns
        ("feature_int_col_0", list(range(sample_size))),
        ("feature_int_col_1", [i * 10 for i in range(sample_size)]),
        ("feature_smallfloat_col_0", [i * 0.0001 for i in range(sample_size)]),
        ("feature_smallfloat_col_1", [i * 0.0002 for i in range(sample_size)]),
        (
            "feature_bigfloat_col_0",
            np.arange(0, big_float, step=big_float_step, dtype=np.float32),
        ),
        (
            # Big Float with bias
            "feature_bigfloat_col_1",
            np.arange(
                big_float,
                3 * big_float,
                step=2 * big_float_step,
                dtype=np.float32,
            ),
        ),
        (
            "feature_bool_col",
            _bool_data_list(sample_size, one_half_samples),
        ),
        # Column with numerical values and explicit dtype="object"
        ("feature_forcedstr_num_col", _float_data_list(sample_size, step=1.3)),
        ("feature_str_col_0", [f"value0_{i}" for i in range(sample_size)]),
        ("feature_str_col_1", [f"value1_{i}" for i in range(sample_size)]),
        (
            "feature_str_categ_col",
            _categ_data_list(sample_size, categories_count=2, prefix="categ_"),
        ),
        (
            "feature_int_categ_col",
            _categ_data_list(sample_size, categories_count=3, prefix=""),
        ),
        ("feature_samenum_col", [3] * sample_size),
        ("feature_samestr_col", ["samevalue"] * sample_size),
        (
            "feature_float_str_mixed_col",
            _string_data_list(one_half_samples)
            + _float_data_list(sample_size - one_half_samples, step=0.5),
        ),
        (
            "feature_int_float_mixed_col",
            _int_data_list(one_half_samples, step=2)
            + _float_data_list(sample_size - one_half_samples, step=23.1),
        ),
        (
            "feature_int_str_mixed_col",
            _string_data_list(one_half_samples)
            + _int_data_list(one_half_samples, step=2),
        ),
        (
            "feature_datetime_col",
            _create_datetime_formatted_list(
                sample_size, datetime_format="%Y-%m-%d %H:%M:%S"
            ),
        ),
        (
            "feature_date_col",
            _create_datetime_formatted_list(sample_size, datetime_format="%d/%m/%YYYY"),
        ),
        (
            "feature_onlyyear_col",
            _create_datetime_formatted_list(sample_size, datetime_format="%YYYY"),
        ),
    ]
    test_dataset = from_tuples(column_tuples)

    return test_dataset


class CSVMock:
    @staticmethod
    def csv_with_nans_strings_substrings(
        sample_size: int, wrong_values_count: int, csv_path: Union[str, Path] = "."
    ):
        """
        Create DataFrame with and without errors to test the preprocessing scripts

        This function returns a DataFrame with ``sample_size`` samples and 30 columns.
        The columns are split between the ones containing metadata info and the
        ones containing features. Both subgroups contain columns of different types
        (e.g.: int, float, string, categorical strings/numbers, datetimes).
        Depending on the column types, the function inserts invalid elements:
        - string columns -> NaN values are inserted
        - int columns -> NaN values and wrong strings These are inserted in each column
        - wrong strings ->
        - wrong substrings
        For each type of error, in `df_correct` we expect to find an expected
        correction if available:
        - NaN values -> not replaced
        - wrong strings -> replace with NaN
        - wrong substrings -> corrected so that the final value should be a number
        in df_correct we expect
            to find the same NaN values
        When more than one of these types of invalid elements is present in one column, th
        In each column there will be equally spaced triplets of consecutive invalid values of these
        types (so the distance between one triplet and the next one will be
        ``sample_size`` // ``wrong_values_count``)

        Parameters
        ----------
        sample_size: int
        csv_path: Union[str, Path]
        """
        test_dataset = _create_ideal_test_dataset(sample_size=sample_size)

        feature_columns = [
            "feature_bigfloat_col_0",
            "feature_bigfloat_col_1",
            "feature_str_categ_col",
            "feature_int_categ_col",
            "feature_float_str_mixed_col",
            "feature_int_float_mixed_col",
            "feature_int_str_mixed_col",
            "feature_datetime_col",
            "feature_date_col",
            "feature_onlyyear_col",
            "feature_int_col_0",
            "feature_int_col_1",
            "feature_smallfloat_col_0",
            "feature_smallfloat_col_1",
            "feature_str_col_0",
            "feature_str_col_1",
            "feature_samenum_col",
            "feature_samestr_col",
            "feature_bool_col",
        ]
        metadata_columns = [
            "metadata_str_categ_col",
            "metadata_num_categ_col",
            "metadata_float_str_mixed_col",
            "metadata_int_float_mixed_col",
            "metadata_int_str_mixed_col",
            "metadata_datetime_col",
            "metadata_date_col",
            "metadata_onlyyear_col",
            "metadata_int_col_0",
            "metadata_int_col_1",
            "metadata_float_col_0",
            "metadata_str_col_0",
            "metadata_str_col_1",
            "metadata_nan_col",
            "metadata_mostly_nan_col",
            "metadata_samenum_col",
            "metadata_samestr_col",
        ]
        only_string_features = [
            "feature_str_categ_col",
            "feature_str_col_0",
            "feature_str_col_1",
            "feature_samestr_col",
        ]
        int_only_features = [
            "feature_int_col_0",
            "feature_int_col_1",
            "feature_samenum_col",
            "feature_int_categ_col",
        ]
        float_only_features = [
            "feature_bigfloat_col_0",
            "feature_bigfloat_col_1",
            "feature_smallfloat_col_0",
            "feature_smallfloat_col_1",
            "feature_forcedstr_num_col",
        ]
        datetime_features = [
            "feature_datetime_col",
            "feature_date_col",
        ]
        numerical_features = tuple(
            set(int_only_features) | set(float_only_features) | {"feature_bool_col"}
        )

        insert_nan_str_substr = Compose(
            [
                ChangeColumnDType(
                    # This must be before the conversion to "Int32" for a known
                    # pandas issue (https://github.com/pandas-dev/pandas/issues/25472)
                    column_names=float_only_features + int_only_features,
                    new_dtype="object",
                    dtype_after_fix=np.float64,
                ),
                ChangeColumnDType(
                    column_names=int_only_features,
                    new_dtype="object",
                    dtype_after_fix=np.int64,
                ),
                ChangeColumnDType(
                    column_names={"feature_bool_col"},
                    new_dtype="object",
                    dtype_after_fix=np.bool,
                ),
                InsertNaNs(
                    column_names=metadata_columns + feature_columns,
                    error_count=wrong_values_count,
                    nan_value_to_insert=pd.NA,
                    nan_value_after_fix=pd.NA,
                ),
                InsertNewValues(
                    # Numerical_features become mixed-type because of this operation
                    column_names=numerical_features,
                    error_count=wrong_values_count,
                    replacement_map=WHOLE_WORD_REPLACE_DICT,
                ),
                InsertNewValues(
                    # String type feature do not get corrections
                    column_names=only_string_features,
                    error_count=wrong_values_count,
                    replacement_map={"Error": "Error", "---": "---"},
                ),
                ReplaceSubstringsByValue(
                    column_names=float_only_features,
                    error_count=wrong_values_count,
                    replacement_map_list=[
                        # Replacing '.' with ','
                        SubstringReplaceMapByValue(".", ",", ".", True)
                    ],
                ),
                InsertSubstringsByIndex(
                    column_names=numerical_features,
                    error_count=wrong_values_count,
                    replacement_map_list=[
                        # inserting '°' symbol into a value, that will be corrected
                        # by removing the symbol (leaving numerical value)
                        SubstringReplaceMapByIndex(-1, "°", "", True),
                        # inserting '%' symbol into a value, that will be corrected
                        # by changing the whole value to NaN
                        SubstringReplaceMapByIndex(-1, "%", pd.NA, False),
                    ],
                ),
                # Introducing wrong datetime format (no correction is provided at
                # the moment by PyTrousse, so the expectation is equal to the raw data)
                ReplaceSubstringsByValue(
                    column_names=datetime_features,
                    error_count=wrong_values_count,
                    replacement_map_list=[
                        SubstringReplaceMapByValue("-", "/", "/", True),
                        SubstringReplaceMapByValue(":", ".", ".", True),
                    ],
                ),
                InsertOutOfScaleValues(
                    column_names=numerical_features,
                    error_count=wrong_values_count,
                    upperbound_increase=0.02,
                    lowerbound_increase=0.02,
                ),
            ]
        )
        test_dataset = insert_nan_str_substr(test_dataset)

        dataframe_to_fix_dir = os.path.join(
            csv_path,
            "raw_data_nans_strinsert_substrmodiffloat_symbolsaddedfloat_substrmodifdatetime.csv",
        )
        dataframe_after_fix_dir = os.path.join(
            csv_path,
            "expectation_nans_strinsert_substrmodiffloat_symbolsaddedfloat_substrmodifdatetime.csv",
        )
        test_dataset.dataframe_to_fix.to_csv(dataframe_to_fix_dir, index=False)
        test_dataset.dataframe_after_fix.to_csv(dataframe_after_fix_dir, index=False)

        return (dataframe_to_fix_dir, dataframe_after_fix_dir)
