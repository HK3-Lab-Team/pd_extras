import logging
import shelve
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from trousse import feature_operations as fop
from trousse.dataset import (
    Dataset,
    _ColumnListByType,
    _find_single_column_type,
    copy_dataset_with_new_df,
    get_df_from_csv,
    read_file,
)
from trousse.exceptions import NotShelveFileError

from ..dataset_util import DataFrameMock, SeriesMock
from ..fixtures import CSV


class Describe_Dataset:
    @pytest.mark.parametrize(
        "nan_ratio, n_columns, expected_nan_columns",
        [
            (0.8, 2, {"nan_0", "nan_1"}),
            (0.8, 1, {"nan_0"}),
            (0.8, 0, set()),
            (
                0.0,
                2,
                {
                    "nan_0",
                    "nan_1",
                    "not_nan_0",
                    "not_nan_1",
                    "not_nan_2",
                    "not_nan_3",
                    "not_nan_4",
                },
            ),
            (1.0, 2, {"nan_0", "nan_1"}),
        ],
    )
    def test_nan_columns(self, request, nan_ratio, n_columns, expected_nan_columns):
        df = DataFrameMock.df_many_nans(nan_ratio, n_columns)
        dataset = Dataset(df_object=df)

        nan_columns = dataset.nan_columns(nan_ratio - 0.01)

        assert len(nan_columns) == len(expected_nan_columns)
        assert isinstance(nan_columns, set)
        assert nan_columns == expected_nan_columns

    @pytest.mark.parametrize(
        "n_columns, expected_constant_columns",
        [(2, {"same_0", "same_1"}), (1, {"same_0"}), (0, set())],
    )
    def test_constant_columns(self, request, n_columns, expected_constant_columns):
        df = DataFrameMock.df_same_value(n_columns)
        dataset = Dataset(df_object=df)

        constant_cols = dataset.constant_cols

        assert len(constant_cols) == len(expected_constant_columns)
        assert isinstance(constant_cols, set)
        assert constant_cols == expected_constant_columns

    @pytest.mark.parametrize(
        "n_columns, expected_trivial_columns",
        [
            (4, {"nan_0", "nan_1", "same_0", "same_1"}),
            (2, {"nan_0", "same_0"}),
            (0, set()),
        ],
    )
    def test_trivial_columns(self, request, n_columns, expected_trivial_columns):
        df = DataFrameMock.df_trivial(n_columns)
        dataset = Dataset(df_object=df)

        trivial_columns = dataset.trivial_columns

        assert len(trivial_columns) == len(expected_trivial_columns)
        assert isinstance(trivial_columns, set)
        assert trivial_columns == expected_trivial_columns

    @pytest.mark.parametrize(
        "sample_size, expected_categ_cols",
        [
            (
                50,
                {
                    "numerical_3",
                    "numerical_5",
                    "string_3",
                    "string_5",
                    "mixed_3",
                    "mixed_5",
                },
            ),
            (
                100,
                {
                    "numerical_3",
                    "numerical_5",
                    "string_3",
                    "string_5",
                    "mixed_3",
                    "mixed_5",
                },
            ),
            (
                3000,
                {
                    "numerical_3",
                    "numerical_5",
                    "numerical_8",
                    "string_3",
                    "string_5",
                    "string_8",
                    "mixed_3",
                    "mixed_5",
                    "mixed_8",
                },
            ),
            (
                15000,
                {
                    "numerical_3",
                    "numerical_5",
                    "numerical_8",
                    "numerical_40",
                    "string_3",
                    "string_5",
                    "string_8",
                    "string_40",
                    "mixed_3",
                    "mixed_5",
                    "mixed_8",
                    "mixed_40",
                },
            ),
        ],
    )
    def test_get_categorical_cols(self, request, sample_size, expected_categ_cols):
        df_categ = DataFrameMock.df_categorical_cols(sample_size)
        dataset = Dataset(df_object=df_categ)

        categ_cols = dataset._get_categorical_cols(col_list=df_categ.columns)

        assert isinstance(categ_cols, set)
        assert categ_cols == expected_categ_cols

    @pytest.mark.parametrize(
        "feature_cols, expected_column_list_type",
        [
            (
                {"metadata_num_col"},
                _ColumnListByType(
                    mixed_type_cols=set(),
                    constant_cols=set(),
                    numerical_cols={"metadata_num_col"},
                    med_exam_col_list={"metadata_num_col"},
                    str_cols=set(),
                    str_categorical_cols=set(),
                    num_categorical_cols=set(),
                    other_cols=set(),
                    bool_cols=set(),
                ),
            ),
            (
                {
                    "metadata_num_col",
                    "mixed_type_col",
                    "same_col",
                    "float_col",
                    "int_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                    "string_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "str_categorical_col",
                    "str_forced_categorical_col",
                    "datetime_col",
                },
                _ColumnListByType(
                    mixed_type_cols={"mixed_type_col"},
                    constant_cols={"same_col"},
                    numerical_cols={
                        "int_col",
                        "float_col",
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                        "metadata_num_col",
                    },
                    med_exam_col_list={
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "int_col",
                        "float_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                        "metadata_num_col",
                    },
                    str_cols={
                        "string_col",
                        "str_categorical_col",
                        "str_forced_categorical_col",
                    },
                    str_categorical_cols={
                        "str_categorical_col",
                        "str_forced_categorical_col",
                    },
                    num_categorical_cols={
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "nan_col",
                    },
                    other_cols={"datetime_col"},
                    bool_cols={"bool_col"},
                ),
            ),
            (
                None,
                _ColumnListByType(
                    mixed_type_cols={"mixed_type_col"},
                    constant_cols={"same_col"},
                    numerical_cols={
                        "float_col",
                        "int_col",
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                    },
                    med_exam_col_list={
                        "float_col",
                        "int_col",
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "bool_col",
                        "interval_col",
                        "nan_col",
                    },
                    str_cols={
                        "string_col",
                        "str_categorical_col",
                        "str_forced_categorical_col",
                    },
                    str_categorical_cols={
                        "str_categorical_col",
                        "str_forced_categorical_col",
                    },
                    num_categorical_cols={
                        "int_categorical_col",
                        "int_forced_categorical_col",
                        "nan_col",
                    },
                    other_cols={"datetime_col"},
                    bool_cols={"bool_col"},
                ),
            ),
        ],
    )
    def test_column_list_by_type(self, feature_cols, expected_column_list_type):
        df_multi_type = DataFrameMock.df_multi_type(sample_size=200)
        dataset = Dataset(
            df_object=df_multi_type,
            metadata_cols=("metadata_num_col",),
            feature_cols=feature_cols,
        )

        col_list_by_type = dataset._columns_type

        assert isinstance(col_list_by_type, _ColumnListByType)
        assert col_list_by_type == expected_column_list_type

    @pytest.mark.parametrize(
        "feature_cols, expected_med_exam_col_list",
        [
            (
                {
                    "float_col",
                    "int_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                    "metadata_num_col",
                },
                {
                    "float_col",
                    "int_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                    "metadata_num_col",
                },
            ),
            (
                {
                    "float_col",
                    "int_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                },
                {
                    "float_col",
                    "int_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                },
            ),
            (
                None,
                {
                    "float_col",
                    "int_col",
                    "int_categorical_col",
                    "int_forced_categorical_col",
                    "bool_col",
                    "interval_col",
                    "nan_col",
                },
            ),
        ],
    )
    def test_med_exam_col_list(self, feature_cols, expected_med_exam_col_list):
        df_multi_type = DataFrameMock.df_multi_type(sample_size=200)
        dataset = Dataset(
            df_object=df_multi_type,
            metadata_cols=("metadata_num_col",),
            feature_cols=feature_cols,
        )

        med_exam_col_list = dataset.med_exam_col_list

        assert isinstance(med_exam_col_list, set)
        assert med_exam_col_list == expected_med_exam_col_list

    @pytest.mark.parametrize(
        "duplicated_cols_count, expected_contains_dupl_cols_bool",
        [(0, False), (4, True), (2, True)],
    )
    def test_contains_duplicated_features(
        self, request, duplicated_cols_count, expected_contains_dupl_cols_bool
    ):
        df_duplicated_cols = DataFrameMock.df_duplicated_columns(duplicated_cols_count)
        dataset = Dataset(df_object=df_duplicated_cols)

        contains_duplicated_features = dataset.check_duplicated_features()

        assert isinstance(contains_duplicated_features, bool)
        assert contains_duplicated_features is expected_contains_dupl_cols_bool

    def test_show_columns_type(self, request):
        # df_col_names_by_type = DataFrameMock.df_column_names_by_type()
        # expected_cols_to_type_map = {
        #     "bool_col_0": "bool_col",
        #     "bool_col_1": "bool_col",
        #     "string_col_0": "string_col",
        #     "string_col_1": "string_col",
        #     "string_col_2": "string_col",
        #     "numerical_col_0": "float_col", "int_col",
        #     "other_col_0": "other_col",
        #     "mixed_type_col_0": "mixed_type_col",
        #     "mixed_type_col_1": "mixed_type_col",
        #     "mixed_type_col_2": "mixed_type_col",
        #     "mixed_type_col_3": "mixed_type_col",
        # }

        # TODO: Check "print" output or make the method easy to test and then complete test
        pass

    @pytest.mark.parametrize(
        "metadata_columns, original_columns, derived_columns, expected_metadata_cols",
        [
            (
                tuple(["metadata_num_col"]),
                ["metadata_num_col"],
                ["derived_metadata_num_col"],
                {"metadata_num_col", "derived_metadata_num_col"},
            ),
            # TODO: add case with a FeatureOperation that can accept more than
            # one derived_column
        ],
    )
    def test_add_operation_with_derived_columns(
        self,
        request,
        metadata_columns,
        original_columns,
        derived_columns,
        expected_metadata_cols,
    ):
        df = DataFrameMock.df_generic(10)
        dataset = Dataset(
            df_object=df,
            metadata_cols=metadata_columns,
        )
        feat_op = fop.FillNA(
            columns=original_columns, derived_columns=derived_columns, value=0
        )

        dataset.add_operation(feat_op)

        for column in original_columns + derived_columns:
            # Check if the operation is added to each column
            assert feat_op in dataset.operations_history[column]
        assert dataset.metadata_cols == expected_metadata_cols

    @pytest.mark.parametrize(
        "metadata_columns, original_columns, expected_metadata_cols",
        [
            (
                ("metadata_num_col", "metadata_str_col"),
                ["metadata_num_col"],
                {"metadata_num_col", "metadata_str_col"},
            ),
            # TODO: add case with a FeatureOperation that can accept more than
            # one derived_column
        ],
    )
    def test_add_operation_with_no_derived_columns(
        self,
        request,
        metadata_columns,
        original_columns,
        expected_metadata_cols,
    ):
        df = DataFrameMock.df_generic(10)
        dataset = Dataset(
            df_object=df,
            metadata_cols=metadata_columns,
        )
        feat_op = fop.FillNA(columns=original_columns, derived_columns=None, value=0)

        dataset.add_operation(feat_op)

        for column in original_columns:
            # Check if the operation is added to each column
            assert feat_op in dataset.operations_history[column]
        assert dataset.metadata_cols == expected_metadata_cols

    def test_add_operation_on_previous_one(self, request, dataset_with_operations):
        feat_op = fop.FillNA(columns=["col1"], derived_columns=["col5"], value=0)

        dataset_with_operations.add_operation(feat_op)

        added_op = dataset_with_operations.operations_history[2]
        # Check if the previous operations are still present
        assert len(dataset_with_operations.operations_history) == 3
        assert isinstance(added_op, fop.FillNA)
        assert added_op.columns == ["col1"]
        assert added_op.derived_columns == ["col5"]

    def test_to_be_fixed_cols(self):
        df = DataFrameMock.df_multi_type(10)
        dataset = Dataset(df_object=df)

        to_be_fixed_cols = dataset.to_be_fixed_cols

        assert type(to_be_fixed_cols) == set
        assert len(to_be_fixed_cols) == 1
        assert to_be_fixed_cols == {"mixed_type_col"}

    @pytest.mark.parametrize(
        "col_id_list, expected_columns_name",
        [
            ([0, 1, 2], {"string_col", "bool_col", "metadata_num_col"}),
            ([0], {"metadata_num_col"}),
            ([], set()),
        ],
    )
    def test_convert_column_id_to_name(self, col_id_list, expected_columns_name):
        df = DataFrameMock.df_multi_type(10)
        dataset = Dataset(df_object=df)

        columns_name = dataset.convert_column_id_to_name(col_id_list)

        assert type(columns_name)
        assert columns_name == expected_columns_name

    def test_str(self):
        df = DataFrameMock.df_multi_type(10)
        dataset = Dataset(df_object=df)
        expected_str = (
            "Columns with:\n\t1.\tMixed types: "
            "\t\t1\n\t2.\tNumerical types (float/int): \t8\n\t3.\tString types: "
            "\t\t3\n\t4.\tBool types: \t\t1\n\t5.\tOther types: \t\t1\nAmong these "
            "categories:\n\t1.\tString categorical columns: 2\n\t2.\tNumeric categorical"
            " columns: 3\n\t3.\tMedical Exam columns (numerical, no metadata): 8\n\t4."
            "\tOne repeated value: 1\nColumns with many NaN: 0"
        )

        str_ = str(dataset)

        assert type(str_) == str
        assert expected_str == str_


@pytest.mark.parametrize(
    "series_type, expected_col_type_dict",
    [
        ("bool", {"col_name": "column_name", "col_type": "bool_col"}),
        ("string", {"col_name": "column_name", "col_type": "string_col"}),
        ("category", {"col_name": "column_name", "col_type": "string_col"}),
        ("float", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("int", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("float_int", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("interval", {"col_name": "column_name", "col_type": "numerical_col"}),
        ("date", {"col_name": "column_name", "col_type": "other_col"}),
        ("mixed_0", {"col_name": "column_name", "col_type": "mixed_type_col"}),
        ("mixed_1", {"col_name": "column_name", "col_type": "mixed_type_col"}),
        ("mixed_2", {"col_name": "column_name", "col_type": "mixed_type_col"}),
    ],
)
def test_find_single_column_type(request, series_type, expected_col_type_dict):
    serie = SeriesMock.series_by_type(series_type)

    col_type_dict = _find_single_column_type(serie)

    assert col_type_dict == expected_col_type_dict


def test_copy_dataset_with_new_df(dataset_with_operations):
    new_df = DataFrameMock.df_generic(10)

    new_dataset = copy_dataset_with_new_df(
        dataset=dataset_with_operations, new_pandas_df=new_df
    )

    assert isinstance(new_dataset, Dataset)
    conserved_attributes = new_dataset.__dict__.keys() - {"_data"}
    for k in conserved_attributes:
        assert new_dataset.__dict__[k] == dataset_with_operations.__dict__[k]
    assert new_dataset.data.equals(new_df)


def test_copy_dataset_with_new_df_log_warning(caplog, dataset_with_operations):
    new_df = DataFrameMock.df_generic(10)
    reduced_new_df = new_df.drop(["exam_num_col_0"], axis=1)

    copy_dataset_with_new_df(
        dataset=dataset_with_operations, new_pandas_df=reduced_new_df
    )

    assert caplog.record_tuples == [
        (
            "root",
            logging.WARNING,
            "Some columns of the previous Dataset instance "
            + "are being lost, but information about operation on them "
            + "is still present",
        )
    ]


def test_to_file(dataset_with_operations, tmpdir):
    filename = tmpdir.join("export_raise_fileexistserr")

    dataset_with_operations.to_file(filename)

    my_shelf = shelve.open(str(filename))
    assert len(my_shelf.keys()) == 1
    exported_dataset = list(my_shelf.values())[0]
    my_shelf.close()
    assert isinstance(exported_dataset, Dataset)
    # This is to identify attribute errors easier
    conserved_attributes = exported_dataset.__dict__.keys() - {"_data"}
    for k in conserved_attributes:
        assert exported_dataset.__dict__[k] == dataset_with_operations.__dict__[k]
    assert exported_dataset.data.equals(dataset_with_operations.data)


def test_to_file_raise_fileexistserror(dataset_with_operations, create_generic_file):
    filename = create_generic_file

    with pytest.raises(FileExistsError) as err:
        dataset_with_operations.to_file(filename)

    assert isinstance(err.value, FileExistsError)
    assert (
        f"File {filename} already exists. If overwriting is not a problem, "
        + "set the 'overwrite' argument to True"
        == str(err.value)
    )


def test_read_file(export_dataset_with_operations_to_file_fixture):
    (
        expected_imported_dataset,
        exported_dataset_path,
    ) = export_dataset_with_operations_to_file_fixture

    imported_dataset = read_file(exported_dataset_path)

    assert isinstance(imported_dataset, Dataset)
    # This is to identify attribute errors easier
    conserved_attributes = imported_dataset.__dict__.keys() - {"_data"}
    for k in conserved_attributes:
        assert imported_dataset.__dict__[k] == expected_imported_dataset.__dict__[k]
    pd.testing.assert_frame_equal(imported_dataset.data, expected_imported_dataset.data)


def test_read_file_raise_notshelvefileerror(create_generic_file):
    with pytest.raises(NotShelveFileError) as err:
        read_file(create_generic_file)

    assert isinstance(err.value, NotShelveFileError)
    assert (
        f"The file {create_generic_file} was not created by 'shelve' module or no "
        f"db type could be determined" == str(err.value)
    )


def test_read_file_raise_typeerror(create_generic_shelve_file):
    with pytest.raises(TypeError) as err:
        read_file(create_generic_shelve_file)

    assert isinstance(err.value, TypeError)
    assert "The object is not a Dataset instance, but it is <class 'str'>" == str(
        err.value
    )


def test_df_from_csv():
    csv_path = CSV.dummy
    expected_df = pd.DataFrame({"header1": [1, 2, 3], "header2": [1, 2, 3]})

    df = get_df_from_csv(csv_path)

    assert isinstance(df, pd.DataFrame)
    pd.testing.assert_frame_equal(df, expected_df)


def test_df_from_csv_notfound():
    csv_path = "fake/path.csv"

    df = get_df_from_csv(csv_path)

    assert df is None


# ====================
#      FIXTURES
# ====================


@pytest.fixture()
def create_generic_file(tmpdir) -> Path:
    """
    Create and store a generic file using Python built-in functions.

    At the end of tests, this file is removed by the finalizer of the
    'tmpdir' fixture.

    Returns
    -------
    pathlib.Path
        Path of the saved file
    """
    filename = tmpdir.join("generic_file_with_string")
    text_file = open(filename, "w")
    text_file.write("Generic File")
    text_file.close()
    return filename


@pytest.fixture()
def create_generic_shelve_file(tmpdir) -> Path:
    """
    Create and store a generic file using 'shelve' module.

    At the end of tests, this file is removed by the finalizer of the
    'tmpdir' fixture.

    Returns
    -------
    pathlib.Path
        Path of the saved file
    """
    filename = tmpdir.join("generic_shelve_file_with_string")
    my_shelf = shelve.open(str(filename), "n")  # 'n' for new
    my_shelf["shelve_data"] = "Generic File"
    my_shelf.close()
    return filename


@pytest.fixture(scope="function")
def dataset_with_operations(fillna_col0_col1, fillna_col1_col4) -> Dataset:
    """
    Create Dataset instance with not empty ``operations_history`` attribute.

    Returns
    -------
    Dataset
        Dataset instance containing FeatureOperation instances
        in the `operations_history` attribute
    """
    dataset = Dataset(df_object=DataFrameMock.df_generic(10))

    dataset.add_operation(fillna_col0_col1)
    dataset.add_operation(fillna_col1_col4)

    return dataset


@pytest.fixture
def export_dataset_with_operations_to_file_fixture(
    dataset_with_operations, tmpdir
) -> Tuple[Dataset, Path]:
    """
    Export a Dataset instance to a file.

    The Dataset instance is created by the fixture ``dataset_with_operations``
    and it is exported using "shelve" module to a file named ``exported_dataset_ops_fixture`` inside
    the folder returned by the fixture ``tmpdir``.

    Returns
    -------
    Dataset
        Dataset instance (created by ``dataset_with_operations`` fixture)
        that is exported to the file
    Path
        Path of the directory where the Dataset instance is saved

    """
    exported_dataset_path = tmpdir / "exported_dataset_ops_fixture"
    my_shelf = shelve.open(str(exported_dataset_path), "n")
    my_shelf["dataset"] = dataset_with_operations
    my_shelf.close()

    return dataset_with_operations, exported_dataset_path
