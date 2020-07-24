from enum import Enum
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class OperationTypeEnum(Enum):
    CATEGORICAL_ENCODING = "categ_encoding"
    BIN_SPLITTING = "bin_splitting"
    FEAT_COMBOS_ENCODING = "feature_combination_encoding"


ENCODED_COLUMN_SUFFIX = '_enc'


class EncodingFunctions(Enum):
    # Enum for possible encoding functions

    # Classic Encoders
    # convert string labels to integer values 1 through k. Ordinal.
    ORDINAL = OrdinalEncoder
    # one column for each value to compare vs. all other values. Nominal, ordinal.
    ONEHOT = OneHotEncoder
