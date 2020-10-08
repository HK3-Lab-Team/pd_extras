# PyTrousse

[![Coverage Status](https://coveralls.io/repos/github/HK3-Lab-Team/pytrousse/badge.svg?branch=coveralls)](https://coveralls.io/github/HK3-Lab-Team/pytrousse?branch=master)
[![Build Status](https://travis-ci.com/HK3-Lab-Team/pytrousse.svg?branch=master)](https://travis-ci.com/HK3-Lab-Team/pytrousse)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HK3-Lab-Team/pytrousse.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/pytrousse/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HK3-Lab-Team/pytrousse.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/pytrousse/context:python)

## WIP ⚠️

PyTrousse collects into one toolbox a set of data wrangling procedures tailored for composing reproducible analytics pipelines. Data transformations include encoding, binning, scaling, strings replacement, NaN filling, column type conversion, data anonymization.

## Getting started
The user can install PyTrousse in his/her Python virtual environment by cloning this repository:

```bash
$ git clone https://github.com/HK3-Lab-Team/pytrousse.git
```

and by running the following command:

```bash
$ cd pytrousse
$ pip install .
```

## Main Features

### Tracing the path from raw data

PyTrousse transformations are progressively wrapped internally with the data, thus linking all stages of data preprocessing for future reproducibility. 

Along with processed data, every `Dataset` object document how the user performed the analysis, in order to reproduce it in the future and to address questions about how the analysis was carried out months, years after the fact.

The traced data path can be inspected through `operation_history` attribute.

```python
>>> dataset.operations_history
```
```bash
[
    FillNA(
        columns=["column_with_nan"],
        derived_columns=["column_filled"],
        value=0,
    ),
    ReplaceSubstrings(
        columns=["column_invalid_values"],
        derived_columns=["column_valid_values"],
        replacement_map={",": ".", "°": ""},
    ),
]
```

### Automatic column data type detection

Wouldn't it be cool to have full column data type detection for your data?

PyTrousse expands Pandas tools for data type inference. Automatic identification is provided on an enlarged set of types (categorical, numerical, boolean, mixed, strings, etc.) using heuristic algorithms.

```python
>>> import trousse
>>> dataset = trousse.read_csv("path/to/csv")

>>> dataset.str_categorical_columns
```
```bash
{"dog_breed", "fur_color"}
```
You can also get the name of boolean columns, numerical columns (i.e. containing integer and float values) or constant columns.
```python
>>> dataset.bool_columns
```
```bash
{"is_vaccinated"}
```
```python
>>> dataset.numerical_columns
```
```bash
{"weight", "age"}
```

### Composable data transformations

What about having an easy API for all those boring data preprocessing steps?

Along with the common preprocessing utilities (for encoding, binning, scaling, etc.), PyTrousse provides tools for noisy data handling and for data anonymization.

```python
>>> from trousse.feature_operations import Compose, FillNA, ReplaceSubstrings

>>> fillna_replacestrings = Compose(
...     [
...         FillNA(
...             columns=["column_with_nan"],
...             derived_columns=["column_filled"],
...             value=0,
...         ),
...         ReplaceSubstrings(
...             columns=["column_invalid_values"],
...             derived_columns=["column_valid_values"],
...             replacement_map={",": ".", "°": ""},
...         ),
...     ]
... )

>>> dataset = fillna_replacestrings(dataset)
```

### Integrated tools for synthetic data generation

PyTrousse aids automated testing by inverting the data transformation operators. Generation of testing fixtures and injection of errors is automatically available (more information [here](https://github.com/HK3-Lab-Team/pytrousse/blob/master/tests/README.py)).
