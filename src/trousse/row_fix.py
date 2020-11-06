import copy
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .dataframe_with_info import DataFrameWithInfo, copy_df_info_with_new_df

logger = logging.getLogger(__name__)


class RowFix:
    """
    This is to fix common errors like mixed types, or little typos in the values
    It contains the methods to fix the errors and to show the results_rf
    """

    def __init__(
        self,
        char_replace_dict: Dict[str, str],
        whole_word_replace_dict: Dict[str, str],
        nan_value: Any = np.nan,
        not_na_string_col_threshold: int = 0.4,
        percentage_to_add_out_of_scale: float = 0.02,
    ):
        """
        Class that fixes common errors like mixed types, or little typos.

        The performed data cleaning should help for column conversion to numeric dtypes.

        Parameters
        ----------
        whole_word_replace_dict : Dict[str, str]
            This string-to-string dict is used to replace the whole value of a
            row with another string/float, or None, so it can be converted
            to float or NaN. The mapped values will be inserted only when the datum
            is exactly identical to the related key.
        char_replace_dict : Dict[str, str]
            This char-to-char dict is used to replace characters inside the strings,
            so they can be converted to numerical
        not_na_string_col_threshold : int
            When we check a column with only string values in order to see if the
            strings are actually numeric values, we try to cast string to numeric and
            we will get NaN if the values are not castable to numeric.  If the ratio
            of "not-NaN values after conversion" / "not-NaN values before conversion
            to numeric" > ``not_na_string_col_threshold``, then the column is
            considered to be numeric and later the script will try to fix some typos
            in remaining NaN. Otherwise the column will be considered as "String" type
        nan_value : Any
        """
        self.nan_value = nan_value
        self.not_na_string_col_threshold = not_na_string_col_threshold
        self.percentage_to_add_out_of_scale = percentage_to_add_out_of_scale
        self.whole_word_replace_dict = whole_word_replace_dict
        self.char_replace_dict = char_replace_dict
        self.errors_before_correction_dict = {}
        self.errors_after_correction_dict = {}
        self.ids_rows_with_remaining_mistakes = set()
        self.ids_rows_with_initial_mistakes = set()

    def _convert_to_float_or_int(self, float_n) -> Any:
        """Choose the appropriate conversion format for a numeric value"""
        try:
            int_n = int(float_n)
            if float_n == int_n:
                return int_n
            else:
                return float_n
        except ValueError:
            return self.nan_value

    def _check_numeric_cols(
        self, df_info: DataFrameWithInfo, col_list: Tuple
    ) -> List[int]:
        """
        Find the columns that have mixed/string-only values interpretable as numbers.
        It will check if the ratio between numeric values (or convertible to ones)
        and the total count of not-NaN values is more than
        the ``not_na_string_col_threshold`` defined in the constructor.

        Parameters
        ----------
        df_info : DataFrameWithInfo

        Returns
        -------
        List[int]
        """
        numeric_cols = []
        for col in col_list:
            numeric_col_serie = pd.to_numeric(df_info.df[col], errors="coerce")
            notna_num_count = numeric_col_serie.count()
            num_valuecount_ratio = notna_num_count / df_info.df[col].count()
            if num_valuecount_ratio > self.not_na_string_col_threshold:
                # Find values that are NaN after conversion (and that were not NaN before)
                lost_values = set(
                    df_info.df[col][df_info.df[col].notna() & numeric_col_serie.isna()]
                )
                logger.info(
                    f"{col} can be converted from String to Numeric. "
                    f"Lost values would be {1- num_valuecount_ratio}: \n{lost_values}"
                )
                numeric_cols.append(col)

        return numeric_cols

    def _populate_non_float_convertible_errors_dict(
        self, full_row: pd.Series, column: str
    ):
        """
        This function is meant to be used with .apply(). So for each row it does the following.
        It will fill up the two arguments:
         "ids_rows_with_initial_mistakes" -> identify ID of the rows with errors (hoping the errors recur in the same)
         "self.errors_before_correction_dict[column]" -> values that are not convertible to float in that column
        """
        try:
            # Try casting to float
            _ = float(full_row[column])
        except (ValueError, TypeError):
            self.ids_rows_with_initial_mistakes.add(full_row.name)
            self.errors_before_correction_dict[column].append(full_row[column])

    def _convert_out_of_scale_values(self, elem, symbol):
        """ Converts '>' and '<' to appropriate value based on 'PERCENTAGE_TO_BE_ADDED' """
        result = str(elem).replace(symbol, "")
        try:
            result = float(result)
            if symbol == ">":
                return self._convert_to_float_or_int(
                    result + self.percentage_to_add_out_of_scale * result
                )
            elif symbol == "<":
                return self._convert_to_float_or_int(
                    result - self.percentage_to_add_out_of_scale * result
                )
            else:
                logger.error(f"You end up using the wrong function to convert {elem}")
        except (ValueError, TypeError):
            logger.error(f"You end up using the wrong function to convert {elem}")

    def _convert_to_float_value(self, full_row, column):
        """
        1. It uses the dict 'char_replace_dict' to replace characters in the element "elem"
        2. It tries to convert that str to float
        3. If it does not work, it tries to replace the whole string (without white spaces)
        using the dict 'self.whole_word_replace_dict'
        4. If nothing worked, it appends the value to the 'errors_after_correction_dict[column]'
        attribute and returns the element 'elem'
        """
        elem = full_row[column]
        try:
            str_to_float = float(elem)
            return self._convert_to_float_or_int(str_to_float)
        except ValueError:
            try:
                # Retry replacing some common mistakes in small parts of the values --> if the only thing
                # remaining is '' or the string can still not be cast to float, ValueError will be raised
                elem = "".join(
                    self.char_replace_dict.get(char, char) for char in str(elem)
                )
                str_to_float = float(elem)
                return self._convert_to_float_or_int(str_to_float)
            except ValueError:
                if "%" in str(elem):
                    # If the value is a percentage, it means that no absolute value can be measured -> None
                    return self.nan_value
                elif ">" in str(elem):
                    # Check if the value was out of scale
                    return self._convert_out_of_scale_values(elem, symbol=">")
                elif "<" in str(elem):
                    return self._convert_out_of_scale_values(elem, symbol="<")

                try:
                    # Try replacing the whole value with the keys in the dict
                    result = self.whole_word_replace_dict[str(elem).strip()]
                    # If None value should be returned, return NAN_VALUE
                    return (
                        self.nan_value
                        if result in [None, ""]
                        else self._convert_to_float_or_int(result)
                    )
                except KeyError:
                    # The whole word could not be replaced by the dict and nothing else worked
                    self.ids_rows_with_remaining_mistakes.add(full_row.name)
                    self.errors_after_correction_dict[column].append(elem)
                    return elem
        except TypeError:
            # NaN value was found, so return NaN
            return self.nan_value

    def fix_typos(
        self, df_info: DataFrameWithInfo, column_list: Tuple = (), verbose: int = 0
    ) -> DataFrameWithInfo:
        """This function is to fix the common errors in the columns "column_list"
        of the pd.DataFrame 'df'

        @param df_info: DataFrameWithInfo
        @param column_list: List of columns that need fixes
        @param verbose: 0 -> No message displayed 1 -> to show performance,
            2 -> to show actual unique errors per column. Default set to 0
        @return: df : pd.DataFrame  with corrections
                 errors_before_correction_dict: Dict with the error list per column before applying the function
                 errors_after_correction_dict: Dict with the error list per column after applying the function

        """
        if column_list == ():
            column_list = df_info.to_be_fixed_cols

        df_converted = copy.copy(df_info.df)

        for c in column_list:
            # Initialize the column key of the dictionaries used to store the errors
            self.errors_before_correction_dict[c] = []
            self.errors_after_correction_dict[c] = []
            # Analyze how many errors are in DF
            df_info.df.apply(
                self._populate_non_float_convertible_errors_dict, column=c, axis=1
            )
            # Fix the errors
            df_converted[c] = df_info.df.apply(
                self._convert_to_float_value, column=c, axis=1
            )
            # Progress bar
            print("=", end="")
        print()

        if verbose:
            logging.info(self.count_errors())

        return copy_df_info_with_new_df(df_info=df_info, new_pandas_df=df_converted)

    def cols_to_correct_dtype(
        self, df_info: DataFrameWithInfo, verbose: int = 0
    ) -> DataFrameWithInfo:
        cols_by_type = df_info.column_list_by_type

        float_cols = set()
        int_cols = set()
        bool_cols = set()

        for col in cols_by_type.numerical_cols:
            col_type = str(type(df_info.df[col].iloc[0]))
            unique_values = df_info.df[col].unique()
            if "bool" in col_type or (
                len(unique_values) == 2
                and unique_values[0] in [0, 1]
                and unique_values[1] in [0, 1]
            ):
                df_info.df[col] = df_info.df[col].astype(np.bool)
                bool_cols.add(col)

            if "float" in col_type:
                df_info.df[col] = df_info.df[col].astype(np.float64)
                float_cols.add(col)
            elif "int" in col_type:
                df_info.df[col] = df_info.df[col].astype("Int32")
                int_cols.add(col)

        bool_cols = bool_cols.union(cols_by_type.bool_cols)
        df_info.df[list(bool_cols)] = df_info.df[list(bool_cols)].astype(np.bool)
        if verbose:
            logger.info(
                f"Casted to INT32: {int_cols}\n Casted to FLOAT64: {float_cols}\n"
                f"Casted to BOOL: {bool_cols}"
            )
        return df_info

    def fix_common_errors(
        self,
        df_info: DataFrameWithInfo,
        set_to_correct_dtype: bool = True,
        verbose: int = 0,
    ) -> DataFrameWithInfo:
        """This function is to fix the common errors in the columns "column_list"
        of the pd.DataFrame 'df'.
        We try to fix:
        1. Mixed columns -> by converting to numbers if possible by using feature_enum.py mappers
        2. String Columns ->
            a. check if they can be treated as numerical columns (by checking how many
        convertible values they contain)
            b. convert the numerical columns as for mixed columns

        @param df_info: DataFrameWithInfo
        @param set_to_correct_dtype: Bool -> Option to choose whether to format every feature
            (int, float, bool columns) to appropriate dtype
        @param verbose: 0 -> No message displayed 1 -> to show performance,
            2 -> to show actual unique errors per column. Default set to 0
        @return: df : pd.DataFrame  with corrections
                 errors_before_correction_dict: Dict with the error list per column before applying the function
                 errors_after_correction_dict: Dict with the error list per column after applying the function
        """
        cols_by_type = df_info.column_list_by_type
        # Get the columns that contain strings, but are actually numerical
        num_cols = self._check_numeric_cols(df_info, col_list=cols_by_type.str_cols)
        # Fix the convertible values
        df_output = self.fix_typos(
            df_info,
            column_list=cols_by_type.mixed_type_cols | set(num_cols),
            verbose=verbose,
        )

        if set_to_correct_dtype:
            df_output = self.cols_to_correct_dtype(df_output, verbose=verbose)

        return df_output

    def print_errors_per_column(self):
        """ This is to print the actual error values, to check the fixes"""
        print("The errors per feature are:")
        for c in self.errors_before_correction_dict.keys():
            print(
                f"{c}: {len(self.errors_before_correction_dict[c])} : {set(self.errors_before_correction_dict[c])}"
                f" ---> {len(self.errors_after_correction_dict[c])} : {set(self.errors_after_correction_dict[c])}"
            )

    def count_errors(self):
        """ This is to count errors before and after fixes"""
        before_errors = 0
        after_errors = 0

        for c in self.errors_before_correction_dict.keys():
            before_errors += len(self.errors_before_correction_dict[c])
            after_errors += len(self.errors_after_correction_dict[c])

        print(
            f"\n Rows with initial mistakes: {len(self.ids_rows_with_initial_mistakes)}"
        )

        print(
            f"\n Total:  BEFORE: {before_errors} errors  -->  AFTER: {after_errors} errors"
        )


# =============================================================================================

# TODO: Still need to be fixed and implemented
#     def _add_fixed_features():
#         pass
#
#     # for k in exam_mixed_type_cols:
#     #     if len(errors_after_correction_dict[k]) == 0:
#     #         df_feat_analysis.med_exam_col_list.append(k)
#
#     def remove_wrong_rows(df, col_list):
#
#         errors_before_correction_dict = {}
#         errors_after_correction_dict = {}
#
#         for c in col_list:
#             # Initialize the column key of the dictionaries used to store the errors
#             errors_before_correction_dict[c] = []
#             errors_after_correction_dict[c] = []
#             df.apply(populate_non_float_convertible_errors_dict,
#                      error_list=errors_before_correction_dict[c])
#
#         print(errors_before_correction_dict)
#
#
#     def remove_wrong_rows(df):
#
#         # Create a boolean map for the rows with mistakes
#         wrong_row_index_list = df.apply(lambda row: is_row_with_mistakes(row[problematic_cols], row['p'], ck, ch),
#                                         axis=1)
#
#         df = df.drop(df[wrong_row_index_list].index)

# ==============================================================================================
