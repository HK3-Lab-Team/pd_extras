import numpy as np
import pandas as pd


class _DfConvertToMixedType:
    """
    Convert values from "object"-typed ``column`` column to appropriate format.

    When pandas package reads from CSV file, the columns that are not completely
    consistent with a single type are stored with dtype = "object" and every value
    is converted to "string".
    This FeatureOperation subclass convert the string values to numeric, boolean
    or datetime values where possible.
    The transformed column will still have dtype="object" but the inferred type will
    be "mixed" which allows a correct column categorization by Dataset class.
    By default the converted column overwrites the related original column.
    To store the result of conversion in another column, ``derived_column``
    parameter has to be set with the name of the corresponding column name.

    This cannot be a FeatureOperation because it is used on a DataFrame before
    the Dataset instance creation.

    Parameters
    ----------
    column : str
        Name of the column with string values that may be converted.
    derived_column : str, optional
        Name of the column where to store the conversion result. Default is None,
        meaning that the converted values are stored in the original column.

    Returns
    -------
    pandas.DataFrame
        The new DataFrame containing the column with converted values.

    See also
    --------
    ConvertToMixedType : Wrapping of this class into a FeatureOperation to apply
        this to a Dataset
    """

    def __init__(
        self,
        column: str,
        derived_column: str = None,
    ):
        self.column = column
        self.derived_column = derived_column
        self._converted_values = None
        self._col_dtype = None

    def _update_converted_values(self, new_converted_values: pd.Series) -> None:
        """
        Insert the new converted values into the related class attribute

        This method adds the new ``new_converted_values`` to the "_converted_values"
        attribute only if the new values have not already been converted. This is
        to prevent to convert numeric values into datetime values for instance,
        by setting the conversion order properly.

        Parameters
        ----------
        new_converted_values : pd.Series
            Pandas Series containing the column values that have been converted
            to a specific type, while all the other values are NaN.
        """
        value_ids_to_insert = np.where(
            np.logical_and(self._converted_values.isna(), new_converted_values.notna())
        )[0]
        self._converted_values[value_ids_to_insert] = new_converted_values[
            value_ids_to_insert
        ]

    @staticmethod
    def _is_column_fully_convertible(
        converted_col: pd.Series, original_col: pd.Series
    ) -> bool:
        """
        Check if the analyzed column has values with the same type only

        This method checks whether the converted column ``converted_col`` has the
        same NaN count as ``original_col`` because that would mean that the
        conversion has been applied to each value and the column can be fully
        converted to that type.

        Parameters
        ----------
        converted_col : pd.Series
            Pandas Series containing the column values after conversion
        original_col : pd.Series
            Pandas Series containing the column values before conversion

        Returns
        -------
        bool
            True if ``converted_col`` has the same NaN count as ``original_col``,
            False otherwise.
        """
        return (converted_col.isna() == original_col.isna()).all()

    def _maybe_update_col_dtype(
        self, converted_col: pd.Series, original_col: pd.Series
    ) -> None:
        """
        Check if the analyzed column has values with the same type only

        This method checks whether the converted column ``converted_col`` has the
        same NaN count as ``original_col`` because that would mean that the
        conversion has been applied to each value and the column can be fully
        converted to that type.

        Parameters
        ----------
        converted_col : pd.Series
            Pandas Series containing the column values after conversion
        original_col : pd.Series
            Pandas Series containing the column values before conversion
        """
        is_column_fully_convertible = self._is_column_fully_convertible(
            converted_col, original_col
        )

        if is_column_fully_convertible and self._col_dtype is None:
            if converted_col.dtype == "int":
                # Change to Int32 because 'int' dtype is not nullable
                self._col_dtype = "Int32"
            else:
                self._col_dtype = converted_col.dtype

    def _convert_to_numeric_mixed_types(self, col_serie: pd.Series) -> None:
        """
        Convert 'object'-typed values to numerical when possible.

        This method analyzes the pandas Series ``col_serie`` looking
        values that can be interpreted as numbers (even if they are string-typed)
        (e.g. '2' -> 2). The found numbers are converted to the appropriate
        numeric type, while the others are set to NaN. The result is added to the
        other conversion results.

        Parameters
        ----------
        col_serie : pd.Series
            Series containing the values that will be analyzed
        """
        numeric_col = pd.to_numeric(col_serie, errors="coerce")
        self._maybe_update_col_dtype(numeric_col, col_serie)
        self._update_converted_values(numeric_col)

    def _convert_to_boolean_mixed_types(self, df: pd.DataFrame) -> None:
        """
        Convert 'object'-typed values to boolean when possible.

        This static method analyzes the column ``col`` in pandas DataFrame ``df``
        and maps the string values "True" and "False" (when present) into the
        related boolean values True and False respectively.
        The result is added to the other conversion results.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the column that will be analyzed.
        """
        bool_map = {"True": True, "False": False}
        col = self.column
        if df[col].dtype != np.dtype("O"):
            # No conversion can be performed if the dtype is not 'object'
            return df
        else:
            converted_df = df.replace({col: bool_map})
            # Set to NaN all the values that were not converted and use the new
            # column as argument for "_update_converted_values" method
            non_bool_ids = np.where(np.equal(converted_df[col], df[col]))[0]
            converted_df[col][non_bool_ids] = pd.NA
            self._maybe_update_col_dtype(converted_df[col], df[col])
            self._update_converted_values(converted_df[col])
            return converted_df

    def _convert_to_datetime_mixed_types(self, col_serie: pd.Series) -> None:
        """
        Convert 'object'-typed values to datetime when possible.

        This static method analyzes the pandas Series ``col_serie`` looking values
        that can be interpreted as datetime values (even if they are string-typed)
        (e.g. '6/12/20' -> 06/12/2020). The found numbers are converted to datetime
        values, while the others are set to NaN. The result is added to the
        other conversion results.

        Parameters
        ----------
        col_serie : pd.Series
            Series containing the values that will be analyzed. It will not be
            modified inplace.
        """
        datetime_col = pd.to_datetime(col_serie, errors="coerce")
        self._maybe_update_col_dtype(datetime_col, col_serie)
        self._update_converted_values(datetime_col)

    def _set_converted_col_dtype(self, col_serie: pd.Series) -> pd.Series:
        """
        Set the new dtype to ``col_serie`` after conversion

        This method sets the new col_dtype to ``col_serie`` column
        TODO: Complete docs

        Parameters
        ----------
        col_serie : pd.Series
            Series containing the values that will be analyzed. It will not be
            modified inplace.
        """
        if self._col_dtype is None:
            # If the _col_dtype is not unique and consistent, convert the column
            # to "object" dtype, otherwise it may have problems with mixed types or NaN
            col_with_dtype = col_serie.astype("object")
        else:
            # If the _col_dtype is unique and consistent, fill NaN with
            # None that will be converted to appropriate value with "astype" call
            col_serie[col_serie.isna()] = None
            col_with_dtype = col_serie.astype(self._col_dtype)

        return col_with_dtype

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ConvertToMixedType operation on a new DataFrame and return it.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to apply the conversion on

        Returns
        -------
        DataFrame
            New DataFrame instance with the conversion applied on
        """
        df_to_convert = df.copy()
        col_to_convert = df_to_convert[self.column].copy()
        # Fill the _converted_values attribute with NaN
        self._converted_values = pd.Series(
            [pd.NA] * len(col_to_convert), dtype="object"
        )
        # These checks must be in precise order where the first check is the most
        # strict. For example the first check can be for numeric values because
        # the numbers must not be interpreted and converted to datetime (this
        # is avoided by caching the  _converted_values list attribute).

        # Convert to numeric the values that are compatible.
        self._convert_to_numeric_mixed_types(col_to_convert)
        # Convert to boolean the values that are compatible
        self._convert_to_boolean_mixed_types(df_to_convert)
        # Convert to datetime the values that are compatible
        self._convert_to_datetime_mixed_types(col_to_convert)

        # Replace the original values with the converted ones
        converted_ids = np.where(self._converted_values.notna())[0]
        col_to_convert[converted_ids] = self._converted_values[converted_ids]

        converted_col = self._set_converted_col_dtype(col_to_convert)

        # Write the converted column into the DataFrame
        if self.derived_column is not None:
            df_to_convert.loc[:, self.derived_column] = converted_col
        else:
            df_to_convert.loc[:, self.column] = converted_col

        return df_to_convert
