import random
from datetime import date

import pandas as pd


class DataFrameMock:
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
        random.seed = 42
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
    def df_multi_type() -> pd.DataFrame:
        """
        Create pandas DataFrame with columns containing values of different types.

        DataFrame has 5 rows and columns as follows:
        1. One column containing boolean values
        2. One column containing string values
        3. One column containing numerical values
        4. One column containing 'category' typed values
        5. One column containing datetime values
        6. One column containing 'interval' typed values
        7. One column containing values of mixed types

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with 5 columns containing values of different types
        """
        return pd.DataFrame(
            {
                "bool_col": [True, False, True, True, False],
                "string_col": ["value_0", "value_1", "value_2", "value_3", "value_4"],
                "categorical_col": pd.Series(
                    [
                        "category_1",
                        "category_1",
                        "category_0",
                        "category_1",
                        "category_0",
                    ],
                    dtype="category",
                ),
                "numerical_col": [0.05 * i for i in range(5)],
                "datetime_col": [date.today() for i in range(5)],
                "interval_col": pd.arrays.IntervalArray(
                    [
                        pd.Interval(0, 1),
                        pd.Interval(1, 5),
                        pd.Interval(2, 5),
                        pd.Interval(1, 4),
                        None,
                    ]
                ),
                "mixed_type_col": [1, 2, 3, 4, "value_0"],
            }
        )

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
