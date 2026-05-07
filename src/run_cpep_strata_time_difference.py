from pathlib import Path

import dask.dataframe as dd

try:
    from data_analysis import ALL_STUDIES, DATASET_CONFIG, DataAnalysis
except ModuleNotFoundError:
    from src.data_analysis import ALL_STUDIES, DATASET_CONFIG, DataAnalysis


def build_lightweight_analysis() -> DataAnalysis:
    analysis = DataAnalysis.__new__(DataAnalysis)
    analysis.datasets = {}
    analysis.datasets_summary = {}
    analysis.analyzed_datasets = {}

    project_root = Path(__file__).resolve().parent.parent
    missing = []
    for study_name in ALL_STUDIES:
        study_config = DATASET_CONFIG.get(study_name)
        if study_config is None:
            missing.append(f"{study_name} (missing config)")
            continue
        parquet_path = project_root / study_config["csv_folder"] / "df_final.parquet"
        if not parquet_path.exists():
            missing.append(str(parquet_path))
            continue
        analysis.datasets[study_name] = dd.read_parquet(parquet_path)

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "The lightweight runner could not find required preprocessed study files:\n"
            f"{missing_text}"
        )

    return analysis


if __name__ == "__main__":
    analysis = build_lightweight_analysis()
    analysis.cpep_strata_time_difference()
