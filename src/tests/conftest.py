import os
import shutil
from pathlib import Path

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
