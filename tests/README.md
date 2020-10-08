### Integrated tools for synthetic data generation

PyTrousse aids developers in automated testing by creating simulated raw data in a controlled way along with the data expected after appropriate data wrangling operations.
Particularly, this tool can support the data transformation operator testing by inverting their behavior through error injection.

Raw and expected data are wrapped in a `TestDataset` object that, following PyTrousse behavior, can be modified by applying the inverted data transformation operators.
```python
>>> import pandas as pd
>>> from .datasim import from_pandas
>>> from .datasim_util import (
...         Compose,
...         InsertOutOfScaleValues,
...         ReplaceSubstringsByValue,
...         SubstringReplaceMapByValue,
... )

>>> df = pd.DataFrame(
...     {"float_column": list(range(10)), "int_column": [0.2 * i for i in range(10)]}
... )
>>> test_dataset = from_pandas(df)
>>> insert_errors = Compose(
...     [
...         ReplaceSubstringsByValue(
...             column_names=["float_column"],
...             error_count=2,
...             replacement_map_list=[
...                 SubstringReplaceMapByValue(
...                     substr_to_replace=".",
...                     new_substring=",",
...                     value_after_fix=".",
...                     fix_with_substring=True,
...                 )
...             ],
...         ),
...         InsertOutOfScaleValues(
...             column_names=["int_column"],
...             error_count=2,
...             upperbound_increase=0.02,
...             lowerbound_decrease=0.02,
...         ),
...     ]
... )
>>> test_dataset = insert_errors(test_dataset)

>>> test_dataset.dataframe_to_fix
```
```bash
   float_column int_column
0             0          0
1             1        0.2
2             2       >1.8
3             3        0.6
4             4        0.8
5             5          1
6             6        1.2
7             7       <0.0
8             8        1.6
9             9        1.8
```
```python
>>> test_dataset.dataframe_after_fix
```
```bash
   float_column  int_column
0             0       0.000
1             1       0.200
2             2       1.836
3             3       0.600
4             4       0.800
5             5       1.000
6             6       1.200
7             7       0.000
8             8       1.600
9             9       1.800
```
