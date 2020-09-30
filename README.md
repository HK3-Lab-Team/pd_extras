# PyTrousse

[![Coverage Status](https://coveralls.io/repos/github/HK3-Lab-Team/pytrousse/badge.svg?branch=coveralls)](https://coveralls.io/github/HK3-Lab-Team/pytrousse?branch=master)
[![Build Status](https://travis-ci.com/HK3-Lab-Team/pytrousse.svg?branch=master)](https://travis-ci.com/HK3-Lab-Team/pytrousse)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HK3-Lab-Team/pytrousse.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/pytrousse/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HK3-Lab-Team/pytrousse.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HK3-Lab-Team/pytrousse/context:python)

## WIP ⚠️

PyTrousse helps to handle tabular data with many features/columns and to automatically track all the preprocessing steps performed on the dataset.
This will help the data scientist to easily reproduce the entire preprocessing pipeline.

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

## Features

### Easy column type inference
Wouldn't it be cool to have full column data type detection for your data?

For example, PyTrousse can automatically identify categorical columns in your dataset using heuristic algorithms.

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
    [
        FillNA(
            column="column_with_nan",
            derived_column="column_filled",
            value=0,
        ),
        ReplaceSubstrings(
            column="column_invalid_values",
            derived_column="column_valid_values",
            replacement_map={",": ".", "°": ""},
        ),
    ]
)

>>> dataset = fillna_replacestrings(dataset)
```

### Preprocessing pipeline tracking and export

Every stage in the transformation process from raw data to clean data should be linked to prior stages. 
This helps you to document how you performed the analysis, in order to reproduce it in the future. Also, it allows you to address questions about how the analysis was carried out months, years after the fact.

PyTrousse automagically tracks every dataset transformation which can be inspected through `operation_history` attribute.

```python
>>> dataset.operations_history
```
```bash
[
    FillNA(
        column="column_with_nan",
        derived_column="column_filled",
        value=0,
    ),
    ReplaceSubstrings(
        column="column_invalid_values",
        derived_column="column_valid_values",
        replacement_map={",": ".", "°": ""},
    ),
]
```
