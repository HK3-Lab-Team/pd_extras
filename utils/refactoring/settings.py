import numpy as np

# Ratio R used to select which columns are categorical. Column is categorical when:
# ("Number of column unique values" < 7 ) or
#       "Number of column unique values" < "Not-NaN values count in column" / R
# R = How many times a unique value is repeated in column (in average)
CATEG_COL_THRESHOLD = 300

# When we check a column with only string values in order to see if the strings are actually numeric values,
# we try to cast string to numeric and we will get NaN if the values are not castable to numeric.
# If the ratio of "not-NaN values after conversion" / "not-NaN values before conversion to numeric" >
#     NOT_NA_STRING_COL_THRESHOLD --> the column is considered to be numeric and later the script will
# try to fix some typos in remaining NaN. Otherwise the column will be considered as "String" type
NOT_NA_STRING_COL_THRESHOLD = 0.4
PERCENTAGE_TO_BE_ADDED_OUT_OF_SCALE_VALUES = 0.02
NAN_VALUE = np.nan

# ===== Mappings to correct some typos in data =====
# Map that replaces these words (keys) with their values when a datum is exactly identical to that key
WHOLE_WORD_REPLACE_DICT = {
    '---': None,
    '.': None,
    'ASSENTI': None,
    'PRESENTI': None,
    'non disponibile': None,
    'NV': None,
    '-': None,
    'Error': None,
    'None': None,
    'NAN': None
    #     '0%': '0'
}
# Map that replaces these characters (keys) with their values in order to
# correct some small typos in the single datum
CHAR_REPLACE_DICT = {
    "°": "",
    ",": "."
}