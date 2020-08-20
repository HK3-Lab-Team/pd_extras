import itertools
import random
from datetime import date

import pandas as pd
import pytest
import sklearn

from ..pd_extras.dataframe_with_info import DataFrameWithInfo, FeatureOperation
from ..pd_extras.feature_enum import OperationTypeEnum


class DataFrameMock:
    @staticmethod
    def df_generic(sample_size):
        """
        Create a generic DataFrame with ``sample_size`` samples and 2 columns.

        The 2 columns of the returned DataFrame contain numerical and string
        values, respectively.

        Parameters
        ----------
        sample_size:
            Number of samples in the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame instance with ``sample_size`` samples and 2 columns:
            one with numerical values and the other with string values only.
        """
        return pd.DataFrame(
            {
                "metadata_num_col": list(range(sample_size)),
                "metadata_str_col": [f"value_{i}" for i in range(sample_size)],
                "exam_num_col_0": list(range(sample_size)),
                "exam_num_col_1": list(range(sample_size)),
                "exam_str_col_0": [f"value_{i}" for i in range(sample_size)],
            }
        )

    @staticmethod
    def df_many_nans(nan_ratio: float, n_columns: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with ``n_columns`` containing ``nan_ratio`` ratio of NaNs.

        DataFrame has 100 rows and ``n_columns``+5 columns. The additional 5 columns
        contain less than ``nan_ratio`` ratio of NaNs.

        Parameters
        ----------
        nan_ratio : float
            Ratio of NaNs that will be present in ``n_columns`` of the DataFrame.
        n_columns : int
            Number of columns that will contain ``nan_ratio`` ratio of NaNs.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with ``n_columns`` containing ``nan_ratio`` ratio of NaNs
            and 5 columns with a lower ratio of NaNs.
        """
        many_nan_dict = {}
        sample_count = 100
        # Create n_columns columns with NaN
        nan_sample_count = int(sample_count * nan_ratio)
        for i in range(n_columns):
            many_nan_dict[f"nan_{i}"] = [pd.NA] * nan_sample_count + [1] * (
                sample_count - nan_sample_count
            )
        # Create not_nan_columns with less than nan_ratio ratio of NaNs
        not_nan_columns = 5
        for j in range(not_nan_columns):
            nan_ratio_per_column = nan_ratio - 0.01 * (j + 1)
            # If nan_ratio_per_column < 0, set 0 samples to NaN (to avoid negative
            # sample counts)
            if nan_ratio_per_column < 0:
                nan_sample_count = 0
            else:
                nan_sample_count = int(sample_count * nan_ratio_per_column)
            many_nan_dict[f"not_nan_{j}"] = [pd.NA] * nan_sample_count + [1] * (
                sample_count - nan_sample_count
            )
        return pd.DataFrame(many_nan_dict)

    @staticmethod
    def df_same_value(n_columns: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with ``n_columns`` containing the same repeated value.

        DataFrame has 100 rows and ``n_columns``+5 columns. The additional 5 columns
        contain different valid values (and a variable count of a repeated value).

        Parameters
        ----------
        n_columns : int
            Number of columns that will contain the same repeated value.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with ``n_columns`` containing the same repeated value
            and 5 columns with some different values.
        """
        random.seed(42)
        same_value_dict = {}
        sample_count = 100
        # Create n_columns columns with same repeated value
        for i in range(n_columns):
            same_value_dict[f"same_{i}"] = [4] * sample_count
        # Create not_same_value_columns with repeated values and random values
        not_same_value_columns = 5
        for j in range(not_same_value_columns):
            same_value_sample_count = int(sample_count * (1 - 0.1 * (j + 1)))
            same_value_dict[f"not_same_{j}"] = [4] * same_value_sample_count + [
                random.random() for _ in range(sample_count - same_value_sample_count)
            ]
        return pd.DataFrame(same_value_dict)

    @staticmethod
    def df_trivial(n_columns: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with ``n_columns`` containing trivial values.

        Half of the trivial columns contains lots of NaN, and the other half contains
        repeated values.
        DataFrame has 100 rows and ``n_columns``+5 columns. The additional 5 columns
        contain random values and a variable count of a repeated value and NaNs.

        Parameters
        ----------
        n_columns : int
            Number of columns that will contain the same repeated value.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with ``n_columns`` containing the same repeated value
            and 5 columns with some different values.
        """
        random.seed(42)
        trivial_dict = {}
        sample_count = 100
        nan_columns = n_columns // 2
        same_value_columns = n_columns - nan_columns
        # Create half of n_columns columns with NaN
        for i in range(nan_columns):
            trivial_dict[f"nan_{i}"] = [pd.NA] * sample_count
        # Create half of n_columns columns with repeated value
        for j in range(same_value_columns):
            trivial_dict[f"same_{j}"] = [4] * sample_count
        # Create 5 more columns with valid values (with NaN, repeated and random values)
        valid_values_columns = 5
        for k in range(valid_values_columns):
            same_value_sample_count = int(sample_count * (1 - 0.05 * (k + 1)) / 2)
            nan_sample_count = int(sample_count * (1 - 0.05 * (k + 1)) / 2)
            random_samples = [
                random.random() * 100
                for _ in range(
                    sample_count - same_value_sample_count - nan_sample_count
                )
            ]
            trivial_dict[f"not_nan_not_same_{k}"] = (
                [4] * same_value_sample_count
                + [pd.NA] * nan_sample_count
                + random_samples
            )
        return pd.DataFrame(trivial_dict)

    @staticmethod
    def df_multi_type(sample_size: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with columns containing values of different types.

        The returned DataFrame has a number of rows equal to the biggest
        value V such that:
        a) V < ``sample_size``
        b) V is divisible by 10.
        The DataFrame has columns as follows:
        1. One column containing boolean values
        2. One column containing string values
        3. One column containing string repeated values ("category" dtype)
        4. One column containing string values (it is meant to simulate metadata)
        5. One column containing numerical values
        6. One column containing numerical repeated values ("category" dtype)
        7. One column containing datetime values
        8. One column containing 'interval' typed values
        9. One column containing values of mixed types
        10. One column containing repeated values
        11. One column containing NaN values (+ 1 numerical value)

        Parameters
        ----------
        sample_size: int
            Number of samples that the returned DataFrame will contain.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with ``sample_size`` samples and 5 columns
            containing values of different types.
        """
        random.seed(42)
        # Get only the part that is divisible by 2 and 5
        sample_size = sample_size // 10 * 10
        bool_col = [True, False, True, True, False] * (sample_size // 5)
        random.shuffle(bool_col)

        df_multi_type_dict = {
            "metadata_num_col": list(range(sample_size)),
            "bool_col": bool_col,
            "string_col": [f"value_{i}" for i in range(sample_size)],
            "str_categorical_col": pd.Series(
                ["category_0", "category_1", "category_2", "category_3", "category_4"]
                * (sample_size // 5),
                dtype="category",
            ),
            "num_categorical_col": pd.Series(
                [0, 1, 2, 3, 4] * (sample_size // 5), dtype="category",
            ),
            "numerical_col": [0.05 * i for i in range(sample_size)],
            "datetime_col": [date(2000 + i, 8, 1) for i in range(sample_size)],
            "interval_col": pd.arrays.IntervalArray(
                [pd.Interval(0, i) for i in range(sample_size)],
            ),
            "mixed_type_col": list(range(sample_size // 2))
            + [f"value_{i}" for i in range(sample_size // 2)],
            "same_col": [2] * sample_size,
            "nan_col": [pd.NA] * (sample_size - 1) + [3],
        }
        return pd.DataFrame(df_multi_type_dict)

    @staticmethod
    def df_column_names_by_type() -> pd.DataFrame:
        """
        Create DataFrame sample that contains column name and types of a generic DataFrame.

        DataFrame has 11 rows and 2 columns. One column called "col_name" contains
        some strings (that represent the column names of another DataFrame sample "df2").
        Another column called "col_type" contains some possible outputs from
        ``src.pd_extras.dataframe_with_info._find_single_column_type`` function
        that describe the type of values contained in the column of "df2".

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with 2 columns containing strings and types respectively.
        """
        return pd.DataFrame(
            [
                {"col_name": "bool_col_0", "col_type": "bool_col"},
                {"col_name": "bool_col_1", "col_type": "bool_col"},
                {"col_name": "string_col_0", "col_type": "string_col"},
                {"col_name": "string_col_1", "col_type": "string_col"},
                {"col_name": "string_col_2", "col_type": "string_col"},
                {"col_name": "numerical_col_0", "col_type": "numerical_col"},
                {"col_name": "other_col_0", "col_type": "other_col"},
                {"col_name": "mixed_type_col_0", "col_type": "mixed_type_col"},
                {"col_name": "mixed_type_col_1", "col_type": "mixed_type_col"},
                {"col_name": "mixed_type_col_2", "col_type": "mixed_type_col"},
                {"col_name": "mixed_type_col_3", "col_type": "mixed_type_col"},
            ]
        )

    @staticmethod
    def df_categorical_cols(sample_size: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with columns containing categorical values

        The returned DataFrame will contain ``sample_size`` samples and 12 columns.
        The columns will be distinguished based on value types (sample_type) and
        number of unique values (unique_value_count).

        Parameters
        ----------
        sample_size: int
            Number of samples in the returned DataFrame

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing ``sample_size`` samples and 12 columns with
            various sample types and number of unique values

        """
        random.seed(42)
        unique_value_counts = (3, 5, 8, 40)
        categ_cols_dict = {}
        mixed_list = [f"string_{i}" for i in range(20)] + [i * 20 for i in range(21)]
        random.shuffle(mixed_list)
        value_per_sample_type = {
            "numerical": [i * 20 for i in range(41)],
            "string": [f"string_{i}" for i in range(41)],
            "mixed": mixed_list,
        }

        for unique_value_count, sample_type in itertools.product(
            unique_value_counts, value_per_sample_type.keys()
        ):
            if sample_size < unique_value_count:
                # Cannot have more unique values than samples
                unique_value_count = sample_size
            # how many times every value will be repeated to fill up the column
            repetitions_per_value = sample_size // unique_value_count
            # This is to always have the same sample_size
            last_value_repetitions = sample_size - repetitions_per_value * (
                unique_value_count - 1
            )
            # Create list of lists containing the same repeated values (category)
            repeated_categories_list = []
            for i in range(unique_value_count - 1):
                repeated_categories_list.append(
                    [value_per_sample_type[sample_type][i]] * repetitions_per_value
                )
            repeated_categories_list.append(
                [value_per_sample_type[sample_type][unique_value_count]]
                * last_value_repetitions
            )
            # Combine lists into one column of the DataFrame
            categ_cols_dict[f"{sample_type}_{unique_value_count}"] = list(
                itertools.chain.from_iterable(repeated_categories_list)
            )
        return pd.DataFrame(categ_cols_dict)

    @staticmethod
    def df_multi_nan_ratio(sample_size: int) -> pd.DataFrame:
        """
        Create pandas DataFrame with columns containing variable ratios of NaN values.

        The returned DataFrame has a number of rows equal to the biggest
        value V such that:
        a) V < ``sample_size``
        b) V is divisible by 10.
        The DataFrame has columns as follows:
        1. One column containing numerical values
        2. One column containing 50% of NaN values
        3. One column containing NaN values + 1 numerical value
        4. One column containing 100% of NaN values

        Parameters
        ----------
        sample_size: int
            Number of samples that the returned DataFrame will contain

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with ``sample_size`` samples and 5 columns
            containing variable ratios of NaN values.
        """
        sample_size = sample_size // 10 * 10
        ratio_50 = int(sample_size * 0.5)
        num_values = [0.05 * i for i in range(sample_size)]
        df_multi_type_dict = {
            "0nan_col": [0.05 * i for i in range(sample_size)],
            "50nan_col": ([pd.NA] * ratio_50) + num_values[:ratio_50],
            "99nan_col": [pd.NA] * (sample_size - 1) + [3],
            "100nan_col": ([pd.NA] * sample_size),
        }
        return pd.DataFrame(df_multi_type_dict)

    @staticmethod
    def df_duplicated_columns(duplicated_cols_count: int) -> pd.DataFrame:
        """
        Create DataFrame with ``duplicated_cols_count`` duplicated columns.

        The returned DataFrame has 5 rows and ``duplicated_cols_count`` + 2 columns.
        Two columns are considered duplicated if they have the same column name

        Parameters
        ----------
        duplicated_cols_count: int
            Number of columns with duplicated name.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with 5 rows and ``duplicated_cols_count`` duplicated
            columns (+ 2 generic columns).
        """
        df_duplicated = pd.DataFrame({"col_0": list(range(5)), "col_3": list(range(5))})
        for _ in range(duplicated_cols_count):
            single_col_df = pd.DataFrame({"duplic_col": list(range(5))})
            df_duplicated = pd.concat([df_duplicated, single_col_df], axis=1)

        return pd.DataFrame(df_duplicated)


class SeriesMock:
    @staticmethod
    def series_by_type(series_type: str):
        col_name = "column_name"
        if "bool" in series_type:
            return pd.Series([True, False, True, pd.NA, False], name=col_name)
        elif "str" in series_type:
            return pd.Series(
                ["value_0", "value_1", "value_2", pd.NA, "value_4"], name=col_name
            )
        elif "float" in series_type:
            return pd.Series([0.05 * i for i in range(4)] + [pd.NA], name=col_name)
        elif "int" in series_type:
            return pd.Series(list(range(4)) + [pd.NA], name=col_name)
        elif "float_int" in series_type:
            return pd.Series([2, 3, 0.1, 4, 5], name=col_name)
        elif "date" in series_type:
            return pd.Series([date.today() for i in range(4)] + [pd.NA], name=col_name)
        elif "category" in series_type:
            return pd.Series(
                ["category_1", "category_1", "category_0", "category_1", "category_0"],
                name=col_name,
                dtype="category",
            )
        elif "interval" in series_type:
            return pd.Series(
                pd.arrays.IntervalArray(
                    [
                        pd.Interval(0, 1),
                        pd.Interval(1, 5),
                        pd.Interval(2, 5),
                        pd.Interval(1, 4),
                        pd.NA,
                    ]
                ),
                name=col_name,
                dtype="Interval",
            )
        elif "mixed_0" in series_type:
            return pd.Series([1, 2, 4, "value_0"] + [pd.NA], name=col_name)
        elif "mixed_1" in series_type:
            return pd.Series([1, 2, False, 4, 5], name=col_name)
        elif "mixed_2" in series_type:
            return pd.Series(["0", 2, 3, 4] + [pd.NA], name=col_name)
