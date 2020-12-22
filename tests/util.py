# encoding: utf-8

import os

import pandas as pd


def load_expectation(expectation_file_name, type_=None):  # pragma: no cover
    """Returns pd.DataFrame related to the *expectation_file_name*.
    Expectation file path is rooted at tests/expectations.
    """
    thisdir = os.path.dirname(__file__)
    expectation_file_path = os.path.abspath(
        os.path.join(thisdir, "expectations", f"{expectation_file_name}.{type_}")
    )
    if type_ == "csv":
        expectation_data = pd.read_csv(expectation_file_path)
    else:
        raise Exception("Type format not recognized")
    return expectation_data
