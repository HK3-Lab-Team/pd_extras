# encoding: utf-8

"""Functions that make mocking with pytest easier and more readable."""

import random

import pandas as pd

from unittest.mock import ANY, call  # noqa # isort:skip
from unittest.mock import create_autospec, patch, PropertyMock  # isort:skip


def class_mock(request, q_class_name, autospec=True, **kwargs):
    """Return mock patching class with qualified name *q_class_name*.

    The mock is autospec'ed based on the patched class unless the optional
    argument *autospec* is set to False. Any other keyword arguments are
    passed through to Mock(). Patch is reversed after calling test returns.
    """
    _patch = patch(q_class_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def function_mock(request, q_function_name, autospec=True, **kwargs):
    """Return mock patching function with qualified name *q_function_name*.

    Patch is reversed after calling test returns.
    """
    _patch = patch(q_function_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def initializer_mock(request, cls, autospec=True, **kwargs):
    """Return mock for __init__() method on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(
        cls, "__init__", autospec=autospec, return_value=None, **kwargs
    )
    request.addfinalizer(_patch.stop)
    return _patch.start()


def instance_mock(request, cls, name=None, spec_set=True, **kwargs):
    """Return mock for instance of *cls* that draws its spec from the class.

    The mock will not allow new attributes to be set on the instance. If
    *name* is missing or |None|, the name of the returned |Mock| instance is
    set to *request.fixturename*. Additional keyword arguments are passed
    through to the Mock() call that creates the mock.
    """
    name = name if name is not None else request.fixturename
    return create_autospec(cls, _name=name, spec_set=spec_set, instance=True, **kwargs)


def method_mock(request, cls, method_name, autospec=True, **kwargs):
    """Return mock for method *method_name* on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, method_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def property_mock(request, cls, prop_name, **kwargs):
    """Return mock for property *prop_name* on class *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, prop_name, new_callable=PropertyMock, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


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