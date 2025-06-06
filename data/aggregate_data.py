import dask.dataframe as dd
import pyarrow as pa

# Define the datasets and their paths
input_files = {
    'cloud': './data/cloud/df_final.parquet',
    'clvr': './data/clvr/df_final.parquet',
    'defend': './data/defend/df_final.parquet',
    'diagnode': './data/diagnode/df_final.parquet',
    'gskalb': './data/gskalb/df_final.parquet',
    # 'islet': './data/itx/df_final.parquet',
    'jaeb_t1d': './data/jaeb_t1d/df_final.parquet'
}

# Columns you want to keep
columns_to_use = [
    'id', 'timestamp', 'time_bins', 'glucose mmol/l', 'sex', 'age', 'weight',
    'height', 'cpep_0_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min',
    'cpep_120_min', 'cpep_auc', 'hb_a1c_local', 'total_ins_dose', 'glucose_0_min',
    'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min', 'beta2_score'
]

# Columns that must be float (ensure type consistency)
numeric_columns = [
    'weight', 'height', 'cpep_0_min', 'cpep_30_min', 'cpep_60_min',
    'cpep_90_min', 'cpep_120_min', 'cpep_auc', 'hb_a1c_local', 'total_ins_dose', 'glucose_0_min',
    'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min', 'beta2_score'
]

# Accumulate cleaned Dask DataFrames here
dfs = []

for name, path in input_files.items():
    print(f"Loading {name} from {path}...")
    df = dd.read_parquet(path, engine='pyarrow')

    df = df[columns_to_use]
    df['id'] = df['id'].astype(str).map(lambda x: f"{name}_{x}", meta=('id', 'str'))

    # Check and print rows with problematic values before type conversion
    for col in numeric_columns:
        bad_rows = df[df[col].astype(str).str.contains("<|>|[a-zA-Z]", na=False)]
        if bad_rows.shape[0].compute() > 0:
            print(f"\n❗ Problematic values found in dataset '{name}', column '{col}':")
            print(bad_rows[['id', col]].compute())

    # Convert numeric columns to float
    for col in numeric_columns:
        df[col] = df[col].astype(float)

    dfs.append(df)

# Combine all cleaned datasets
final_ddf = dd.concat(dfs, interleave_partitions=True)

# Write Parquet (schema inferred from consistent dtypes)
print("Saving to Parquet...")
final_ddf.to_parquet(
    './data/aggregated_data.parquet',
    write_index=False,
    engine='pyarrow',
    overwrite=True
)

# final_df = dd.read_parquet(f'./data/aggregated_data.parquet')
# print(final_df.columns)
# print(final_df.isna().sum().compute())