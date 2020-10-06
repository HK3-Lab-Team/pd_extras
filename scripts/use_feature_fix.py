import os
import sys

from trousse.dataset import Dataset
from trousse.feature_enum import EncodingFunctions
from trousse.feature_fix import encode_single_categorical_column

sys.path.append("../..")

CWD = os.path.abspath(os.path.dirname("__file__"))
# DB_SMVET = os.path.join('/home/lorenzo-hk3lab/WorkspaceHK3Lab', 'smvet','data', 'Sani_15300_anonym.csv')
# SEGMENTATION_DATA = os.path.join(CWD, '..', 'segmentation', 'resources', 'dense_areas_percentage.csv')
DB_CORRECT = os.path.join(CWD, "..", "..", "data", "Sani_15300_anonym.csv")
dataset = Dataset(metadata_cols=(), data_file=DB_CORRECT)
print(dataset.data.columns)
col = "SEX"
dataset = encode_single_categorical_column(
    dataset, col_name=col, encoding=EncodingFunctions.ONEHOT
)
print("end")
