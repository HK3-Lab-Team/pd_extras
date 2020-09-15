import os
import time

from trousse.dataset import Dataset

df_sani_dir = os.path.join(
    "/home/lorenzo-hk3lab/WorkspaceHK3Lab/",
    "smvet",
    "data",
    "Sani_15300_anonym.csv",
)

metadata_cols = (
    "GROUPS	TAG	DATA_SCHEDA	NOME	ID_SCHEDA	COMUNE	PROV	MONTH	YEAR	BREED	SEX	AGE	"
    "SEXUAL STATUS	BODYWEIGHT	PULSE RATE	RESPIRATORY RATE	TEMP	BLOOD PRESS MAX	BLOOD "
    "PRESS MIN	BLOOD PRESS MEAN	BODY CONDITION SCORE	HT	H	DEATH	TIME OF DEATH	"
    "PROFILO_PAZIENTE	ANAMNESI_AMBIENTALE	ANAMNESI_ALIMENTARE	VACCINAZIONI	FILARIOSI	GC_SEQ"
)
metadata_cols = tuple(metadata_cols.replace("\t", ",").split(","))

df_sani = Dataset(metadata_cols=metadata_cols, data_file=df_sani_dir)

time0 = time.time()
print(df_sani.column_list_by_type)
print(time.time() - time0)
whole_word_replace_dict = {
    "---": None,
    ".": None,
    "ASSENTI": "0",
    "non disponibile": None,
    "NV": None,
    "-": None,
    "Error": None,
    #     '0%': '0'
}

char_replace_dict = {"°": "", ",": "."}

# df_feat_analysis = DfFeatureAnalysis(df_sani, metadata_cols=metadata_cols)
# # mixed_type_cols, numerical_col_list, str_col_list, other_col_list = df_feat_analysis.get_column_list_by_type()
# df_feat_analysis.show_column_type_infos()
