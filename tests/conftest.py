import os
import shutil
from pathlib import Path
from trousse import feature_operations as fop
import pytest


@pytest.fixture(scope="module")
def temporary_data_dir(request) -> Path:
    """
    Create a temporary directory for test data and delete it after test end.

    The temporary directory is created in the working directory and it is
    named "temp_test_data_folder".
    The fixture uses a finalizer that deletes the temporary directory where
    every test data was saved. Therefore every time the user calls tests that
    use this fixture (and save data inside the returned directory), at the end
    of the test the finalizer deletes this directory.

    Parameters
    ----------

    Returns
    -------
    Path
        Path where every temporary file used by tests is saved.
    """
    temp_data_dir = Path(os.getcwd()) / "temp_test_data_folder"
    try:
        os.mkdir(temp_data_dir)
    except FileExistsError:
        pass

    def remove_temp_dir_created():
        shutil.rmtree(temp_data_dir)

    request.addfinalizer(remove_temp_dir_created)
    return temp_data_dir


@pytest.fixture
def fillna_col0_col1():
    return fop.FillNA(columns=["col0"], derived_columns=["col1"], value=0)


@pytest.fixture
def fillna_col1_col4():
    return fop.FillNA(columns=["col1"], derived_columns=["col4"], value=0)


@pytest.fixture
def fillna_col4_none():
    return fop.FillNA(columns=["col4"], derived_columns=None, value=0)


@pytest.fixture
def fillna_col1_col2():
    return fop.FillNA(columns=["col1"], derived_columns=["col2"], value=0)


@pytest.fixture
def replacestrings_exam_num_col_0_replaced_exam_num_col_0_a_b():
    return fop.ReplaceStrings(
        columns=["exam_num_col_0"],
        derived_columns=["replaced_exam_num_col_0"],
        replacement_map={"a": "b"},
    )
