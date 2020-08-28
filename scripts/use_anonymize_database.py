import os

from pd_extras.anonymize_database import anonymize_data

private_cols_to_map = ["CLIENTE", "INDIRIZZO", "CIVICO", "COMUNE", "PROV"]
private_cols_to_remove = ["CLIENTE", "INDIRIZZO", "CIVICO"]

df_sani_dir = os.path.join(os.getcwd(), "data", "Sani 15300.csv")
output_data_dir = os.path.join(os.getcwd(), "data")
df_sani = pd.read_csv(df_sani_dir)

anonymize_data(
    df_sani, "Sani_15300", private_cols_to_remove, private_cols_to_map, output_data_dir,
)
