import logging

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import DtypeObj


class _StrColumnToConvert:
    """
    Class describing a string column containing values with mixed types.

    When Pandas package loads a column containing values with multiple types,
    each column value will be casted to string (even numeric, bool,
    datetime, ... values).
    The class is meant to be used for columns that contain values with mixed types
    that need to be converted to appropriate types (e.g. '5' converted to 5).

    Parameters
    ----------
    values : pd.Series
        Pandas Series that contains the original values of the column that needs
        to be converted to mixed typed values.
    dtype : DtypeObj, optional
        Value that will be set as new dtype of the column. If the value is not None,
        the new column dtype will be forced to this value and the incompatible
        values will be set to None. If None, the dtype will be updated each time
        new converted values are provided and the last value will be used to convert
        the column. Default set to None

    Private Attributes
    ----------
    _original_values : pd.Series
        Column values with the original values that will be replaced with
        converted values if possible.
    _converted_values : pd.Series
        Pandas Series containing the values of the ``original_values`` column,
        which have been converted to the appropriate type if possible.
        If no conversion was possible, the related values are NaN.
    """

    def __init__(self, values: pd.Series, dtype: DtypeObj = None):

        self._coerce_dtype_conversion = not (dtype is None)
        self._dtype = dtype
        self._original_values = values.copy()
        # Initialize the _converted_values attribute with NaN
        self._converted_values = pd.Series([pd.NA] * len(values), dtype="object")

    @property
    def original_values(self) -> pd.Series:
        """
        Return the original values of the column.

        These are the values that will be replaced with converted values
        where possible.

        Returns
        -------
        pd.Series
            Pandas Series containing the original values of the column.
        """
        return self._original_values

    @property
    def dtype(self) -> DtypeObj:
        """
        Return the appropriate column dtype.

        Each time new converted values are added to this instance, the new column
        dtype is recomputed.
        Particularly, if each value of the column has a consistent conversion to
        the same type, the column dtype will be changed to the corresponding dtype
        On the other hand, if column values are interpreted with multiple types,
        the column dtype will remain "object"

        Returns
        -------
        DtypeObj
            Value corresponding to the column dtype according to the converted
            values that have been added/associated to this instance column
        """
        if self._dtype is None:
            # If the _col_dtype is not unique and consistent, the column dtype
            # will be "object", that is the universal dtype
            return "object"
        elif self._dtype == "int":
            # Change to Int32 because 'int' dtype is not nullable
            return pd.Int32Dtype()
        else:
            return self._dtype

    @property
    def converted_values(self) -> pd.Series:
        """
        Return only the values from original column that could be converted.

        Returns
        -------
        pd.Series
            Values from the original column that have been converted and added as
            converted values to this instance.
        """
        return self._converted_values

    @property
    def original_with_converted_values(self) -> pd.Series:
        """
        Return the original column with the new converted values.

        The method replaces all the not-NaN converted values into the original
        column ones, such that only the values from ``original_values`` that could be
        converted, will be replaced.
        Then the method converts the column to the appropriate dtype (based on
        the converted values that could be converted), replaces the NaNs with
        the appropriate NaN format (for datetime column, it will be replaced
        by NaT value) and returns the resulting pandas Series.

        Returns
        -------
        pd.Series
            Pandas Series combining the converted values from ``converted_values``
            with the ``original_values`` (when the value could not be converted).
            This is also converted to the appropriate dtype.
        """
        orig_with_conv_values = self.original_values.copy()
        # Replace the original values with the converted ones
        converted_ids = np.where(self.converted_values.notna())[0]
        orig_with_conv_values.loc[converted_ids] = self.converted_values[converted_ids]

        # If the _col_dtype is unique and consistent, fill NaN with
        # None. These will then be converted to appropriate value with "astype"
        # conversion
        orig_with_conv_values.loc[orig_with_conv_values.isna()] = None

        if self._coerce_dtype_conversion:
            # The dtype was explictly requested in the constructor, so if some
            # value cannot be converted, we inform the user and use dtype='object'
            orig_with_conv_values = self._safe_convert_to_dtype(orig_with_conv_values)
        else:
            # The dtype was inferred, so if some value cannot be converted, it
            # should raise an error because something went wrong in computation
            orig_with_conv_values = orig_with_conv_values.astype(
                self.dtype,
                errors="raise",
            )

        return orig_with_conv_values

    def _safe_convert_to_dtype(self, col_serie: pd.Series) -> pd.Series:
        """
        Convert ``col_serie`` to the instance dtype, if possible

        If the conversion of some ``col_serie`` values is not possible, the user
        will be informed and the column will be converted to dtype='object'.

        Parameters
        ----------
        col_serie : pd.Series
            Pandas series that will be converted to the new dtype

        Returns
        -------
        pd.Series
            Pandas Series converted to the new dtype
        """
        try:
            converted_col_serie = col_serie.astype(
                self.dtype,
                errors="raise",
            )
        except ValueError as e:
            # The column dtype cannot be changed to dtype.
            # 1. If the dtype was inferred, something went wrong in computation
            # 2. If the dtype was explictly requested, we inform the user
            logging.warning(
                f"{e}\nThe requested column dtype cannot be used on the column"
                "and 'object' dtype will be used instead."
            )
            converted_col_serie = col_serie.astype(
                self.dtype,
                errors="ignore",
            )
        return converted_col_serie

    def _is_single_typed_column(
        self,
        new_converted: pd.Series,
    ) -> bool:
        """
        Check if the analyzed column has values with the same type only

        This method checks whether the converted column ``new_converted`` has the
        same NaN count as the original column because that would mean that the
        conversion has been applied to each value and the column can be fully
        converted to that type.
        It also checks if the values have been converted previously with a more
        appropriate type. This is because the conversions are sorted according
        to priority and, for instance, boolean values can be interpreted
        as numeric, but the column should not be converted to numeric.

        Parameters
        ----------
        original_col : pd.Series
            Pandas Series containing the original column values before conversion
        new_converted : pd.Series
            Pandas Series containing the column values that have been converted
            to a single specific type, while all the other values are NaN.

        Returns
        -------
        bool
            True if ``converted_col`` has the same NaN count as ``original_col``,
            and if no value has already been converted to another more appropriate
            type. False otherwise.
        """
        return (new_converted.isna() == self.original_values.isna()).all() and (
            self.converted_values.isna()
        ).all()

    def _updated_dtype(self, new_converted: pd.Series) -> DtypeObj:
        """
        Update the column dtype according to the ``new_converted`` values

        This method checks whether the converted column ``converted_col`` has the
        same NaN count as ``original_col`` because that would mean that the
        conversion has been applied to each value and the column can be fully
        converted to that type.

        Parameters
        ----------
        new_converted : pd.Series
            Pandas Series containing the column values that have been converted
            to a specific type.

        Returns
        -------
        DtypeObj
            Dtype of the ``new_converted`` pandas Series if all the not-NaN values
            from ``original_col`` were appropriately converted to ``new_converted``
            values (e.g.: The boolean "True" can be converted to integer "1", but
            the conversion is not considered appropriate because it is not its
            native type). If this condition is not fulfilled, the previous found
            column dtype is found (None if it was not found).
        """
        is_column_fully_convertible = self._is_single_typed_column(new_converted)

        if is_column_fully_convertible and self._dtype is None:

            self._dtype = new_converted.dtype

        return self._dtype

    def update_converted_values(self, new_converted: pd.Series) -> None:
        """
        Store the converted values from ``new_converted`` into the instance.

        This method adds the ``new_converted`` to the ``previously_converted``
        only if the new values were not previously converted
        (i.e. where "previously_converted`` is NaN).
        This is, for instance, to prevent the conversion of numeric values
        into datetime values by converting the numeric values of the column
        before converting datetime values.

        Parameters
        ----------
        new_converted : pd.Series
            Pandas Series containing the column values that have been converted
            to a single specific type, while all the other values are NaN.
        """
        value_ids_to_insert = np.where(
            np.logical_and(self._converted_values.isna(), new_converted.notna())
        )[0]
        self._converted_values.loc[value_ids_to_insert] = new_converted[
            value_ids_to_insert
        ]

        self._updated_dtype(new_converted)


class _ConvertDfToMixedType:
    """
    Convert values from "object"-typed ``column`` column to appropriate format.

    When pandas package reads from CSV file, the columns that are not completely
    consistent with a single type are stored with dtype = "object" and every value
    is converted to "string".
    This class converts the string values of a column in a pandas DataFrame
    to numeric, boolean or datetime values where possible.
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

    Private Attributes
    ------------------
    _converted_values : pd.Series
        Pandas Series storing only the values of the ``column`` that were
        converted when calling the instance. It contains NaN if the values of the
        original column have not yet been converted (or not convertible to types
        different than string).
    _col_dtype : DtypeObj
        Dtype of the ``new_converted`` pandas Series if all the not-NaN values
        from ``original_col`` were appropriately converted to ``new_converted``
        values (e.g.: The boolean "True" can be converted to integer "1", but
        the conversion is not considered appropriate because it is not its
        native type). If this condition is not fulfilled, the previous found
        column dtype is found (None if it was not found).

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

    @staticmethod
    def _convert_numeric_values(column: _StrColumnToConvert) -> _StrColumnToConvert:
        """
        Convert 'object'-typed values to numerical when possible.

        This static method analyzes the the ``column`` original values looking
        for those that can be interpreted as numbers (even if they are string-typed)
        (e.g. '2' -> 2). The found numbers are converted to the appropriate
        numeric type, while the others are set to NaN. The result is added to the
        other conversion results.

        Parameters
        ----------
        column : _MixedColumn
            Column containing the values that need to be converted

        Returns
        -------
        column : _MixedColumn
            Column containing the new converted values
        """
        numeric_col = pd.to_numeric(column.original_values, errors="coerce")
        column.update_converted_values(numeric_col)
        return column

    @staticmethod
    def _convert_bool_values(column: _StrColumnToConvert) -> _StrColumnToConvert:
        """
        Convert 'object'-typed values to boolean when possible.

        This static method analyzes the ``column`` original values and maps the
        string values "True" and "False" (when present) into the
        related boolean values True and False respectively.
        The result is added to the other conversion results.

        Parameters
        ----------
        column : _MixedColumn
            Column containing the values that need to be converted

        Returns
        -------
        column : _MixedColumn
            Column containing the new converted values
        """
        bool_map = {"True": True, "False": False}
        # Conversion can be performed only if the dtype is not 'object'
        if column.dtype == np.dtype("O"):
            converted_col = column.original_values.replace(
                to_replace=bool_map, inplace=False
            )
            # Set to NaN all the values that were not converted or boolean and use
            # the new column as argument for "_update_converted_values" method
            non_bool_ids = np.where(
                np.logical_not(np.isin(converted_col, [True, False]))
            )[0]
            converted_col[non_bool_ids] = pd.NA

            column.update_converted_values(converted_col)
        return column

    @staticmethod
    def _convert_datetime_values(column: _StrColumnToConvert) -> _StrColumnToConvert:
        """
        Convert 'object'-typed values to datetime when possible.

        This static method analyzes the ``column`` original values looking for those
        that can be interpreted as datetime values (even if they are string-typed)
        (e.g. '6/12/20' -> 06/12/2020). The found numbers are converted to datetime
        values, while the others are set to NaN. The result is added to the
        other conversion results.

        Parameters
        ----------
        column : _MixedColumn
            Column containing the values that need to be converted

        Returns
        -------
        column : _MixedColumn
            Column containing the new converted values
        """
        datetime_col = pd.to_datetime(column.original_values, errors="coerce")
        column.update_converted_values(datetime_col)
        return column

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
        col_to_convert = _StrColumnToConvert(values=df_to_convert[self.column])

        # These checks must be in precise order where the first check is the most
        # strict. For example the checks priority must be:
        # 1. Boolean: No boolean can be interpreted as number
        # 2. Numeric: No number can be interpreted and converted to datetime
        # 3. datetime
        # These misinterpretation are avoided by caching the  _converted_values
        # list attribute.

        # Convert to boolean the values that are compatible
        col_to_convert = self._convert_bool_values(col_to_convert)
        # Convert to numeric the values that are compatible.
        col_to_convert = self._convert_numeric_values(col_to_convert)
        # Convert to datetime the values that are compatible
        col_to_convert = self._convert_datetime_values(col_to_convert)

        # Write the converted column into the DataFrame
        if self.derived_column is not None:
            df_to_convert.loc[
                :, self.derived_column
            ] = col_to_convert.original_with_converted_values
        else:
            df_to_convert.loc[
                :, self.column
            ] = col_to_convert.original_with_converted_values

        return df_to_convert

    # def _set_converted_col_dtype(self, col_serie: pd.Series) -> pd.Series:
    #     """
    #     Set the new dtype to ``col_serie`` after conversion.

    #     This method updates, if possible, the ``col_serie`` column dtype.
    #     Particularly, if each value of the column has been consistently interpreted
    #     with a single type, the column will be converted to that dtype
    #     (and NaNs will be converted coherently with the new dtype).
    #     On the other hand, if column values are interpreted with multiple types,
    #     the column will maintain the dtype="object" (and the column values will
    #     have multiple types).

    #     Parameters
    #     ----------
    #     col_serie : pd.Series
    #         Series containing the values that will be analyzed. It will not be
    #         modified inplace.

    #     Returns
    #     -------
    #     pd.Series
    #         Column with the same values as ``col_serie`` and the dtype set
    #         according to the value types.
    #     """
    #     if self._col_dtype is None:
    #         # If the _col_dtype is not unique and consistent, convert the column
    #         # to "object" dtype, otherwise it may have problems with mixed types or NaN
    #         col_with_dtype = col_serie.astype("object")
    #     else:
    #         # If the _col_dtype is unique and consistent, fill NaN with
    #         # None that will be converted to appropriate value with "astype" call
    #         col_serie[col_serie.isna()] = None
    #         col_with_dtype = col_serie.astype(self._col_dtype)

    #     return col_with_dtype

    # @property
    # def col_dtype(self):
    #     """
    #     Return the appropriate column dtype after conversion.

    #     This method returns the ``col_serie`` column dtype according to the
    #     values that could be converted
    #     Particularly, if each value of the column has been consistently interpreted
    #     with a single type, the column will be converted to that dtype
    #     (and NaNs will be converted coherently with the new dtype).
    #     On the other hand, if column values are interpreted with multiple types,
    #     the column will maintain the dtype="object" (and the column values will
    #     have multiple types).

    #     Parameters
    #     ----------
    #     col_serie : pd.Series
    #         Series containing the values that will be analyzed. It will not be
    #         modified inplace.

    #     Returns
    #     -------
    #     pd.Series
    #         Column with the same values as ``col_serie`` and the dtype set
    #         according to the value types.
    #     """
    #     if self._col_dtype is None:
    #         # If the _col_dtype is not unique and consistent, the column dtype
    #         # will be "object", that is the universal dtype
    #         return "object"
    #     else:
    #         return self._col_dtype

    # @staticmethod
    # def _replace_converted_values(
    #     original_values: pd.Series, converted_values: pd.Series
    # ) -> pd.Series:
    #     """
    #     Replace the new converted values into the original column values.

    #     This method replaces the ``converted_values`` into the ``original_values``
    #     for all the values that are not NaN (i.e. the values from
    #     ``original_values`` that could be converted).

    #     Parameters
    #     ----------
    #     original_values : pd.Series
    #         Column values with the original values that will be replaced with
    #         converted values if possible.
    #     converted_values : pd.Series
    #         Pandas Series containing the values of the ``original_values`` column,
    #         which have been converted to the appropriate type if possible.
    #         If no conversion was possible, the related values are NaN.

    #     Returns
    #     -------
    #     pd.Series
    #         Pandas Series combining the converted values from ``converted_values``
    #         with the ``original_values`` (when the value could not be converted).
    #     """
    #     original_values = original_values.copy()
    #     # Replace the original values with the converted ones
    #     converted_ids = np.where(converted_values.notna())[0]
    #     original_values.loc[converted_ids] = converted_values[converted_ids]

    #     return original_values

    # def _add_converted_values(self, new_converted: pd.Series):
    #     """
    #     Insert values from ``new_converted`` into ``previously_converted``

    #     This method adds the ``new_converted`` to the ``previously_converted``
    #     only if the new values were not previously converted
    #     (i.e. where "previously_converted`` is NaN).
    #     This is, for instance, to prevent the conversion of numeric values
    #     into datetime values by converting the numeric values of the column
    #     before converting datetime values.

    #     Parameters
    #     ----------
    #     new_converted : pd.Series
    #         Pandas Series containing the column values that have been converted
    #         to a single specific type, while all the other values are NaN.
    #     """
    #     value_ids_to_insert = np.where(
    #         np.logical_and(self._converted_values.isna(), new_converted.notna())
    #     )[0]
    #     self._converted_values.loc[value_ids_to_insert] = new_converted[
    #         value_ids_to_insert
    #     ]
    #     return self._converted_values

    # def _is_single_typed_column(
    #     self,
    #     original_col: pd.Series,
    #     new_converted: pd.Series,
    # ) -> bool:
    #     """
    #     Check if the analyzed column has values with the same type only

    #     This method checks whether the converted column ``converted_col`` has the
    #     same NaN count as ``original_col`` because that would mean that the
    #     conversion has been applied to each value and the column can be fully
    #     converted to that type.
    #     It also checks if the values have been converted previously with a more
    #     appropriate type. This is because the conversions are sorted according
    #     to priority and, for instance, boolean values can be interpreted
    #     as numeric, but the column should not be converted to numeric.

    #     Parameters
    #     ----------
    #     original_col : pd.Series
    #         Pandas Series containing the original column values before conversion
    #     new_converted : pd.Series
    #         Pandas Series containing the column values that have been converted
    #         to a single specific type, while all the other values are NaN.

    #     Returns
    #     -------
    #     bool
    #         True if ``converted_col`` has the same NaN count as ``original_col``,
    #         and if no value has already been converted to another more appropriate
    #         type. False otherwise.
    #     """
    #     return (new_converted.isna() == original_col.isna()).all() and (
    #         self._converted_values.isna()
    #     ).all()

    # def _update_col_dtype(
    #     self,
    #     original_col: pd.Series,
    #     new_converted: pd.Series,
    # ) -> DtypeObj:
    #     """
    #     Check if the analyzed column has values with the same type only.

    #     This method checks whether the converted column ``converted_col`` has the
    #     same NaN count as ``original_col`` because that would mean that the
    #     conversion has been applied to each value and the column can be fully
    #     converted to that type.

    #     Parameters
    #     ----------
    #     original_col : pd.Series
    #         Pandas Series containing the column values before conversion
    #     new_converted : pd.Series
    #         Pandas Series containing the column values after conversion

    #     Returns
    #     -------
    #     DtypeObj
    #         Dtype of the ``new_converted`` pandas Series if all the not-NaN values
    #         from ``original_col`` were appropriately converted to ``new_converted``
    #         values (e.g.: The boolean "True" can be converted to integer "1", but
    #         the conversion is not considered appropriate because it is not its
    #         native type). If this condition is not fulfilled, the previous found
    #         column dtype is found (None if it was not found).
    #     """
    #     is_column_fully_convertible = self._is_single_typed_column(
    #         original_col, new_converted
    #     )

    #     if is_column_fully_convertible and self._col_dtype is None:
    #         if new_converted.dtype == "int":
    #             # Change to Int32 because 'int' dtype is not nullable
    #             self._col_dtype = pd.Int32Dtype
    #         else:
    #             self._col_dtype = new_converted.dtype

    #     return self._col_dtype
