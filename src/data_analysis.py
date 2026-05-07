import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, mannwhitneyu, shapiro, kruskal, spearmanr, linregress, t, gmean, f_oneway, chi2, norm, rankdata
from itertools import combinations
import seaborn as sns
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
import importlib
import math
from PIL import Image
from pathlib import Path
import os
import re
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import patsy
from itx_package import ITXData
from hupa_ucm_package import HUPA_UCM_Data

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

#TODO:
# N_readings
# Days of CGM data
# Cleaning Data more appropiately
# Create time_bins more appropriately (e.g. using visit not timestamps). Make sure then to see if there are any duplicated for each time_bin.

'''
Time Bin Creation:
Bandit:
    - CGM: Visit (Visit 2 ...)
    - HbA1c: Visit (Visit 2 ...)
    - C-peptide: Visit (Visit 2 ...)
    - Glucose: Visit (Visit 2 ...)
    - Insulin: Visit (Visit 2 ...)
    - Extra: Visit (Visit 2 ...)

CLOUD:
    - CGM: no visit. Use taylor_time_bins()
    - HbA1c: VisitCollected (Base, 3, 6, 9, 12, 15, 18, 21, 24 months)
    - C-peptide: VisitCollected (Baseline, 6,12,24 months)
    - Glucose: VisitCollected (Baseline, 6,12,24 months)
    - Insulin: Visits (CloudVisitInfo.txt)
    - Extra: Visits (CloudVisitInfo.txt)

CLVR:
    - CGM: Visit (Weeks)
    - HbA1c: Visit (Weeks)
    - C-peptide: Visit (Weeks)
    - Glucose: Visit (Weeks)
    - Insulin: Visit (Weeks)
    - Extra: Visit (Weeks)
'''

scaler = MinMaxScaler()
itx_dataloader = None
hupa_dataloader = None

SAVE_CSV = True
DATASET_CONFIG = {
    "bandit": {"csv_folder": "./data/studies/bandit", "preprocess": "preprocess_bandit"},
    "cloud": {"csv_folder": "./data/studies/cloud", "preprocess": "preprocess_cloud"},
    "clvr": {"csv_folder": "./data/studies/clvr", "preprocess": "preprocess_clvr"},
    "defend": {'csv_folder': './data/studies/defend', "preprocess": 'preprocess_defend'},
    "diagnode": {"csv_folder": "./data/studies/diagnode", "preprocess": "preprocess_diagnode"},
    "gskalb": {'csv_folder': './data/studies/gskalb', 'preprocess': 'preprocess_gskalb'},
    "hupa_ucm" : {'csv_folder': './data/studies/hupa_ucm', 'preprocess': 'preprocess_hupa_ucm'},
    # "itx": {"csv_folder": "./data/studies/itx", "preprocess": "preprocess_itx"},
    "jaeb_healthy": {"csv_folder": "./data/studies/jaeb_healthy", "preprocess": "preprocess_jaeb_healthy"},
    "jaeb_t1d": {"csv_folder": "./data/studies/jaeb_t1d", "preprocess": "preprocess_jaeb_t1d"}
}

# """ Workstream 1 """
# POSITIVE_STUDIES = ['clvr']
# NEGATIVE_STUDIES = ['bandit', 'cloud', 'defend', 'diagnode', 'gskalb', 'jaeb_t1d']
# ALL_STUDIES = POSITIVE_STUDIES + NEGATIVE_STUDIES

""" Workstream 2 """
POSITIVE_STUDIES = ['clvr']
NEGATIVE_STUDIES = ['cloud']
ALL_STUDIES = POSITIVE_STUDIES + NEGATIVE_STUDIES

TREATMENT_GROUP_1 = ['mdi', 'placebo', 'control', 'non-hcl']
TREATMENT_GROUP_2 = ['cl', 'hcl', 'verapamil', 'diamyd', 'active', 'csii']

FEATURE_LABELS_WITH_UNITS = {
    'total_ins_dose': 'Total Insulin Dose (Units/KG per day)',
    'beta2_score': 'Beta2 Score',
    'hb_a1c': 'HbA1c (%)',
    'gmi': 'GMI (%)',
    'TIR': 'Time In Range (% 3.9–10 mmol/L)',
    'TITR': 'Time In Tight Range (% 3.9–7.8 mmol/l)',
    'TBR_Lvl_1': 'Time Below Range Lvl 1 (% 3.0–3.9 mmol/l)',
    'TBR_Lvl_2': 'Time Below Range Lvl 2 (% <3.0 mmol/l)',
    'TAR_Lvl_1': 'Time Above Range Lvl 1 (% 10.0–13.9 mmol/l)',
    'TAR_Lvl_2': 'Time Above Range Lvl 2 (% >13.9 mmol/l)',
    'cv_percent': 'Coefficient of Variation (%)',
    'log_cpep_auc': 'Log C-Peptide AUC (nmol/L·min)',
    'cpep_auc_preservation': 'C-peptide AUC Preservation (%)',
    'cpep_auc': 'C-peptide AUC (nmol/l)'
}
CGM_CORE_ENDPOINTS = {'percent_wear_time','TIR','TITR','TBR_Lvl_1','TBR_Lvl_2','TAR_Lvl_1',
                      'TAR_Lvl_2','mean_glucose','median_glucose', 'min_glucose','max_glucose',
                      'std_glucose','cv_percent','GVP'}
CGM_ENDPOINTS = CGM_CORE_ENDPOINTS | {'hb_a1c', 'gmi', 'total_ins_dose'}

#Remove Tolerance for filling all entries for columns such as cpep, hba1c, and etc.
TOLERANCE_PD_DAYS = pd.Timedelta(days=30)
TOLERANCE_DAYS = 30


def get_itx_dataloader() -> ITXData:
    global itx_dataloader
    if itx_dataloader is None:
        itx_dataloader = ITXData()
    return itx_dataloader

def get_hupa_dataloader() -> HUPA_UCM_Data:
    global hupa_dataloader
    if hupa_dataloader is None:
        hupa_dataloader = HUPA_UCM_Data()
    return hupa_dataloader

class DataAnalysis:
    def __init__(self):
        """Initializes and loads all datasets dynamically."""
        self.datasets = {}
        self.datasets_summary = {}
        self.analyzed_datasets = {}
        self.load_data()

    def load_data(self) -> None:
        """
        Loads datasets from CSV/Parquet, applies preprocessing if needed,
        and stores raw, summary, and analysis results in class attributes.
        
        Updates:
            - self.datasets (main data)
            - self.datasets_summary (summary stats)
            - self.analyzed_datasets (statistical analysis)
        """

        print('\n\tLoading, summarizing, and analyzing datasets...')
        for dataset_name, config in DATASET_CONFIG.items():
            csv_folder = config["csv_folder"]
            if not os.path.exists(csv_folder):
                print(f"Warning: {csv_folder} not found. Skipping {dataset_name}.")
                continue

            print(f"Loading {dataset_name} from {csv_folder}...")

            if os.path.exists(f'{csv_folder}/df_final.parquet') and os.path.exists(f'{csv_folder}/df_summary.csv'):
                df = dd.read_parquet(f'{csv_folder}/df_final.parquet')
                df_summary = pd.read_csv(f'{csv_folder}/df_summary.csv')

            else: 
                df, df_summary = self._preprocess_data(csv_folder, config["preprocess"])
                df.to_parquet(f'{csv_folder}/df_final.parquet', index=False)
                df_summary.to_csv(f'{csv_folder}/df_summary.csv', index=False)
                    
            self.datasets[dataset_name] = df
            self.datasets_summary[dataset_name] = df_summary
            print('Loading and Summary Done.')

            if os.path.exists(f'{csv_folder}/df_analysis.csv'):
                df_analysis = pd.read_csv(f'{csv_folder}/df_analysis.csv')
            else:
                df_analysis = self.statistical_analysis(df)
                df_analysis.to_csv(f'{csv_folder}/df_analysis.csv', index=False)

            self.analyzed_datasets[dataset_name] = df_analysis
            print('Analysis Done.\n')

        print('\tLoading Done.\n')

    """ Preprocess Data """
    def _preprocess_data(self, csv_folder: str, preprocess_func_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the specified preprocessing function from this class or the ITX package.

        Args:
            csv_folder: Path to the dataset folder.
            preprocess_func_name: Name of the preprocessing function.

        Returns:
            A tuple containing the processed dataframe and its summary.
        """
        preprocess_func = getattr(self, preprocess_func_name, None)

        if preprocess_func:
            return preprocess_func(csv_folder, SAVE_CSV)
        
    def fill_static_within_time_bin(self, df: pd.DataFrame, extra_exclude: set[str] | None = None) -> pd.DataFrame:
        """
        Forward/backward fill static columns within each (id, time_bin) group.
        
        Only fills non-dynamic fields (excludes identifiers, timestamps, CGM values).
        """
        group_cols = ['id', 'time_bin']
        exclude = {'id', 'time_bin', 'timestamp', 'timestamp_type', 'visit', 'visit_date', 'glucose mmol/l', 'timestamp_seconds', 'dy'}
        if extra_exclude:
            exclude |= set(extra_exclude)
        cols_to_fill = [col for col in df.columns if col not in exclude]
        for col in cols_to_fill:
            df[col] = df.groupby(group_cols)[col].transform(lambda s: s.ffill().bfill())
        return df

    def merge_nearest_nonempty_by_dy(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the nearest right-side row within +/- TOLERANCE_DAYS by id and day,
        considering only rows that contain at least one non-empty feature value.
        """
        right_merge = right.drop(columns=['time_bin'], errors='ignore').copy()
        value_columns = [col for col in right_merge.columns if col not in {'id', 'dy'}]
        if not value_columns:
            return left

        right_merge[value_columns] = right_merge[value_columns].replace(r'^\s*$', np.nan, regex=True)
        right_merge = right_merge[right_merge[value_columns].notna().any(axis=1)]
        if right_merge.empty:
            return left

        right_merge = (
            right_merge
            .sort_values(by=['dy', 'id'])
            .reset_index(drop=True)
            .rename(columns={'dy': 'match_dy'})
        )

        merged = pd.merge_asof(
            left,
            right_merge,
            by='id',
            left_on='cgm_window_dy',
            right_on='match_dy',
            direction='nearest',
            tolerance=TOLERANCE_DAYS
        )
        return merged.drop(columns=['match_dy'], errors='ignore')

    def preprocess_bandit(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        def normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
            if 'id' in df.columns:
                df['id'] = df['id'].astype('string').str.strip()
            if 'visit' in df.columns:
                df['visit'] = df['visit'].astype('string').str.replace(r'\s+', ' ', regex=True).str.strip()
            return df

        def merge_unscheduled_by_event_date(
            df_left: pd.DataFrame,
            df_right: pd.DataFrame,
            value_columns: list[str],
            tolerance_days: int = 30
        ) -> pd.DataFrame:
            unscheduled_visits = {'Unscheduled On-Site Visit', 'Unscheduled Remote Visit'}
            if df_right.empty:
                return df_left

            df_right_unscheduled = df_right[
                df_right['visit'].isin(unscheduled_visits) & df_right['event_date'].notna()
            ][['id', 'visit', 'event_date'] + value_columns].copy()
            if df_right_unscheduled.empty:
                return df_left

            # merge_asof requires the join keys to be globally sorted. Sorting by
            # id first can still leave timestamp/event_date non-monotonic overall.
            df_left_sorted = (
                df_left.dropna(subset=['timestamp'])
                .sort_values(by=['timestamp', 'id'], kind='mergesort')
                .reset_index(drop=False)
            )
            df_right_sorted = (
                df_right_unscheduled.dropna(subset=['event_date'])
                .sort_values(by=['event_date', 'id'], kind='mergesort')
                .reset_index(drop=True)
            )

            df_nearest = pd.merge_asof(
                df_left_sorted,
                df_right_sorted,
                by='id',
                left_on='timestamp',
                right_on='event_date',
                direction='nearest',
                tolerance=pd.Timedelta(days=tolerance_days),
                suffixes=('', '_unscheduled')
            )

            for col in value_columns:
                unscheduled_col = f'{col}_unscheduled'
                if unscheduled_col in df_nearest.columns:
                    df_nearest[col] = df_nearest[col].combine_first(df_nearest[unscheduled_col])

            drop_cols = ['visit_unscheduled', 'event_date'] + [
                f'{col}_unscheduled' for col in value_columns if f'{col}_unscheduled' in df_nearest.columns
            ]
            df_nearest.drop(columns=[col for col in drop_cols if col in df_nearest.columns], inplace=True)

            return df_nearest.sort_values(by='index').drop(columns=['index']).reset_index(drop=True)
        
        df_cgm = pd.read_csv(f'{csv_folder}/original/BANDIT_CGM_selfp.csv')
        df_cgm.rename(columns={'visit': 'visit', 'id': 'id'}, inplace=True)
        df_cgm = normalize_merge_keys(df_cgm)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], errors='coerce')
        df_cgm['glucose mmol/l'] = pd.to_numeric(df_cgm['glucose mmol/l'], errors='coerce')
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l']).copy()
        df_cgm_clean = df_cgm_clean.sort_values(by=['id', 'timestamp']).reset_index(drop=True)
        
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(
            lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime'
        )
        df_cgm_clean = self.get_taylor_time_bins(df=df_cgm_clean, offset=0, dy_available=False)
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'visit', 'glucose mmol/l']]

        df_visit_info_raw = df_cgm_clean[['id', 'visit', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()
        df_visit_time_bin_frequency = (
            df_visit_info_raw.groupby(['visit', 'time_bin'])
            .size()
            .reset_index(name='n')
            .sort_values(by=['visit', 'n', 'time_bin'], ascending=[True, False, True])
        )
        df_visit_time_bin_frequency['max_n'] = df_visit_time_bin_frequency.groupby('visit')['n'].transform('max')
        tied_visit_time_bins = df_visit_time_bin_frequency[df_visit_time_bin_frequency['n'] == df_visit_time_bin_frequency['max_n']]
        tied_visits = tied_visit_time_bins.groupby('visit').size()
        tied_visits = tied_visits[tied_visits > 1]

        df_visit_time_bin_canonical = (
            df_visit_time_bin_frequency.drop_duplicates(subset=['visit'], keep='first')[['visit', 'time_bin']]
            .rename(columns={'time_bin': 'canonical_time_bin'})
            .sort_values(by='visit')
        )

        df_visit_time_bin_frequency.to_csv(f'{csv_folder}/csv_files/visit_time_bin_frequency.csv', index=False)
        df_visit_time_bin_canonical.to_csv(f'{csv_folder}/csv_files/visit_time_bin_canonical.csv', index=False)

        if not tied_visits.empty:
            tied_visit_summary = (
                tied_visit_time_bins[tied_visit_time_bins['visit'].isin(tied_visits.index)]
                .groupby('visit')['time_bin']
                .apply(lambda s: sorted(pd.unique(s)))
                .reset_index(name='tied_time_bins')
                .sort_values(by='visit')
            )
            raise ValueError(
                "Cannot build canonical visit-to-time_bin mapping for BANDIT because some visits have a tie for the most common time bin:\n"
                f"{tied_visit_summary.to_string(index=False)}"
            )

        df_cgm_clean = df_cgm_clean.merge(df_visit_time_bin_canonical, on='visit', how='left')
        df_cgm_clean['time_bin'] = df_cgm_clean['canonical_time_bin'].combine_first(df_cgm_clean['time_bin'])
        df_cgm_clean.drop(columns=['canonical_time_bin'], inplace=True)

        df_visit_info = df_cgm_clean[['id', 'visit', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()
        df_visit_time_bin_audit = (
            df_visit_info.sort_values(by=['id', 'visit', 'time_bin'])
            .reset_index(drop=True)
        )
        df_visit_time_bin_summary = (
            df_visit_info.groupby(['id', 'visit'])['time_bin']
            .apply(lambda s: sorted(pd.unique(s)))
            .reset_index(name='time_bins')
            .sort_values(by=['id', 'visit'])
        )
        df_visit_time_bin_summary['n_time_bins'] = df_visit_time_bin_summary['time_bins'].apply(len)

        df_visit_info.to_csv(f'{csv_folder}/csv_files/visit_info_cgm.csv', index=False)
        df_visit_time_bin_audit.to_csv(f'{csv_folder}/csv_files/visit_time_bin_audit.csv', index=False)
        df_visit_time_bin_summary.to_csv(f'{csv_folder}/csv_files/visit_time_bin_summary.csv', index=False)

        df_hba1c = pd.read_csv(f'{csv_folder}/original/BANDIT_HBA.csv')
        df_hba1c = df_hba1c[['Subject unique ID', 'Event name', 'Event date', 'HbA1c']]
        df_hba1c.rename(columns={'Subject unique ID': 'id', 'Event name': 'visit', 'Event date': 'event_date', 'HbA1c': 'hb_a1c'}, inplace=True)
        df_hba1c = normalize_merge_keys(df_hba1c)
        df_hba1c['event_date'] = pd.to_datetime(df_hba1c['event_date'], errors='coerce', dayfirst=True)
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])

        df_mmtt = pd.read_csv(f'{csv_folder}/original/BANDIT_MTT.csv')
        df_mmtt = df_mmtt[['Subject unique ID', 'Event name', 'Event date', 'Pre Mixed-Meal Plasma C-peptide',
                        'Pre Mixed-Meal Plasma Glucose', '0 Mins Plasma C-peptide', '0 Mins Plasma Glucose',
                        '15 Mins Plasma C-peptide', '15 Mins Plasma Glucose','30 Mins Plasma C-peptide',
                        '30 Mins Plasma Glucose', '60 Mins Plasma C-peptide', '60 Mins Plasma Glucose',
                        '90 Mins Plasma C-peptide', '90 Mins Plasma Glucose', '120 Mins Plasma C-peptide',
                        '120 Mins Plasma Glucose']]
        df_mmtt.rename(columns={'Subject unique ID': 'id', 'Event name': 'visit', 'Event date': 'event_date', 'Pre Mixed-Meal Plasma C-peptide': 'cpep_pre10_min',
                               '0 Mins Plasma C-peptide': 'cpep_0_min', '15 Mins Plasma C-peptide': 'cpep_15_min',
                               '30 Mins Plasma C-peptide': 'cpep_30_min', '60 Mins Plasma C-peptide': 'cpep_60_min',
                               '90 Mins Plasma C-peptide': 'cpep_90_min', '120 Mins Plasma C-peptide': 'cpep_120_min',
                               'Pre Mixed-Meal Plasma Glucose': 'glucose_pre10_min', '0 Mins Plasma Glucose': 'glucose_0_min',
                               '15 Mins Plasma Glucose': 'glucose_15_min', '30 Mins Plasma Glucose': 'glucose_30_min',
                               '60 Mins Plasma Glucose': 'glucose_60_min', '90 Mins Plasma Glucose': 'glucose_90_min',
                               '120 Mins Plasma Glucose': 'glucose_120_min'}, inplace=True)
        df_mmtt = normalize_merge_keys(df_mmtt)
        df_mmtt['event_date'] = pd.to_datetime(df_mmtt['event_date'], errors='coerce', dayfirst=True)
        
        df_cpep = df_mmtt[['id', 'visit', 'event_date', 'cpep_pre10_min', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']]
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis=1)

        df_glucose = df_mmtt[['id', 'visit', 'event_date', 'glucose_pre10_min', 'glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                              'glucose_90_min', 'glucose_120_min']]
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                                                     'glucose_90_min', 'glucose_120_min'])
        
        df_insulin = pd.read_csv(f'{csv_folder}/original/BANDIT_IU.csv')
        df_insulin = df_insulin[['Subject unique ID', 'Event name', 'Event date', 'Basal Insulin Average Dose', 'Bolus Insulin Average Dose']]
        df_insulin.rename(columns={'Subject unique ID': 'id', 'Event name': 'visit', 'Event date': 'event_date', 'Basal Insulin Average Dose': 'basal_ins_dose',
                                   'Bolus Insulin Average Dose': 'bolus_ins_dose'}, inplace=True)
        df_insulin = normalize_merge_keys(df_insulin)
        df_insulin['event_date'] = pd.to_datetime(df_insulin['event_date'], errors='coerce', dayfirst=True)
        df_insulin['total_ins_dose'] = df_insulin['basal_ins_dose'] + df_insulin['bolus_ins_dose']
        df_insulin_clean = df_insulin.dropna(subset=['basal_ins_dose', 'bolus_ins_dose'])

        df_insulin_pump = pd.read_csv(f'{csv_folder}/original/BANDIT_CIP.csv')
        df_insulin_pump = df_insulin_pump[['Subject unique ID', 'Event name', 'Use CSII Insulin Pump']]
        df_insulin_pump.rename(columns={'Subject unique ID': 'id', 'Event name': 'visit', 'Use CSII Insulin Pump': 'insulin_delivery'}, inplace=True)
        df_insulin_pump = normalize_merge_keys(df_insulin_pump)
        insulin_delivery_raw = df_insulin_pump['insulin_delivery'].astype(str).str.strip()
        df_insulin_pump.loc[insulin_delivery_raw == 'Yes', 'insulin_delivery'] = 'CSII'
        df_insulin_pump.loc[insulin_delivery_raw == 'No', 'insulin_delivery'] = 'Non-CSII'

        df_extra = pd.read_csv(f'{csv_folder}/original/BANDIT participants_Dx to randomization and baseline characteristics.csv')
        df_extra = df_extra[['Subject unique ID', 'Age Range', 'gender', 'race', 'Tx']]
        df_extra.rename(columns={'Subject unique ID': 'id', 'Age Range': 'age', 'gender': 'sex',
                                 'Tx': 'treatment_arm'}, inplace=True)
        df_extra = normalize_merge_keys(df_extra)
        
        df_height_weight = pd.read_csv(f'{csv_folder}/original/BANDIT_weights and heights.csv')
        df_height_weight = df_height_weight.loc[
            :, df_height_weight.columns.str.contains(r'Subject unique ID|Weight|Height', case=False, regex=True)
        ]
        df_height_weight = df_height_weight.loc[
            :, ~df_height_weight.columns.str.contains(r'Assessed', case=False, regex=True)
        ]
        id_col = 'Subject unique ID'
        value_cols = [col for col in df_height_weight.columns if col != id_col]

        df_height_weight_long = df_height_weight.melt(
            id_vars=[id_col],
            value_vars=value_cols,
            var_name='source_column',
            value_name='value'
        )
        df_height_weight_long[['visit', 'metric']] = df_height_weight_long['source_column'].str.extract(
            r'^(.*?)\(\d+\).*?(Weight|Height)\s*$'
        )
        df_height_weight_long = df_height_weight_long.dropna(subset=['visit', 'metric'])
        df_height_weight_long['visit'] = df_height_weight_long['visit'].str.strip()
        df_height_weight_long['metric'] = df_height_weight_long['metric'].str.lower()

        df_height_weight = df_height_weight_long.pivot_table(
            index=[id_col, 'visit'],
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_height_weight.rename(columns={id_col: 'id'}, inplace=True)
        df_height_weight.rename_axis(None, axis=1, inplace=True)
        df_height_weight = df_height_weight.reindex(columns=['id', 'visit', 'weight', 'height'])
        df_height_weight = normalize_merge_keys(df_height_weight)
        df_height_weight['weight'] = pd.to_numeric(df_height_weight['weight'], errors='coerce')
        df_height_weight['height'] = pd.to_numeric(df_height_weight['height'], errors='coerce')
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])

        df_final = df_cgm_clean.copy()
        print(df_final.shape)
        df_final = df_final.merge(df_extra, on=['id'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_height_weight, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_hba1c_clean[['id', 'visit', 'hb_a1c']], on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(
            df_cpep[['id', 'visit', 'cpep_pre10_min', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc']],
            on=['id','visit'],
            how='left'
        )
        print(df_final.shape)
        df_final = df_final.merge(
            df_glucose[['id', 'visit', 'glucose_pre10_min', 'glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min']],
            on=['id','visit'],
            how='left'
        ) # Bucket 2
        print(df_final.shape)
        df_final = df_final.merge(df_insulin_pump, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_insulin[['id', 'visit', 'basal_ins_dose', 'bolus_ins_dose', 'total_ins_dose']], on=['id','visit'], how='left') # Bucket 2
        print(df_final.shape)
        df_final = merge_unscheduled_by_event_date(df_final, df_hba1c_clean, ['hb_a1c'])
        df_final = merge_unscheduled_by_event_date(df_final, df_cpep, ['cpep_pre10_min', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc'])
        df_final = merge_unscheduled_by_event_date(df_final, df_glucose, ['glucose_pre10_min', 'glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min'])
        df_final = merge_unscheduled_by_event_date(df_final, df_insulin, ['basal_ins_dose', 'bolus_ins_dose', 'total_ins_dose'])
        df_final = self.fill_static_within_time_bin(df_final)
        self.get_gmi(df_final)
        self.get_beta_2_scores(df_final)
        self.get_beta_3_score(df_final)

        if save_csv:
            df_cgm.to_csv('./data/studies/bandit/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/bandit/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/bandit/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/bandit/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/bandit/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/bandit/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/bandit/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/bandit/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/bandit/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/bandit/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/bandit/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/bandit/csv_files/height_weight/df_clean.csv', index=False) 
            df_extra.to_csv('./data/studies/bandit/csv_files/extra/df_original.csv', index=False)
            df_final.to_csv('./data/studies/bandit/df_final.csv', index=False)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')

        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()

        # min_age = no_dup_df['age'].min()
        # max_age = no_dup_df['age'].max()
        # num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        # num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

        c_peptide_interval = ''
        beta2_interval = ''
        cgm_interval = '15 minutes'

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    # 'Minimum age': [min_age],
                    # 'Maximum age': [max_age],
                    # 'Number of people < 18 years': [num_less_than_18],
                    # 'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}

        return df_final, pd.DataFrame(df_summary)

    def preprocess_cloud(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the CLOUD dataset by cleaning, aligning visits, merging clinical/lab data,
        computing C‑peptide AUC and BETA2, and producing a final longitudinal table plus a summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One-row summary statistics table.
        """
        #Visit Info 
        df_visit_info = pd.read_csv(f'{csv_folder}/original/CloudVisitInfo.txt', sep="|")
        
        df_visit_info['VisitDt'] = pd.to_datetime(df_visit_info['VisitDt'], format="%m/%d/%Y")
        df_visit_info['PtID'] = df_visit_info['PtID'].str.strip()
        df_visit_info.drop_duplicates(subset=['RecID'], inplace=True)
        df_visit_info.rename(columns={'PtID': 'id', 'Visit': 'visit', 'VisitDt': 'visit_date'}, inplace=True)
        df_visit_info = df_visit_info[['id', 'visit', 'visit_date']].dropna(subset=['visit_date']).sort_values(by=['visit_date'])

        df_insulin = pd.read_csv(f'{csv_folder}/original/CloudInsulinTherapy.txt', sep="|")
        df_insulin['PtID'] = df_insulin['PtID'].str.strip()
        df_insulin.rename(columns={'PtID': 'id','TotInsDose' :'total_ins_dose', 'TotBasalDose': 'basal_ins_dose', 'Visit': 'visit'}, inplace=True)
        df_insulin.drop_duplicates(subset=['RecID'], inplace=True)
        df_insulin['bolus_ins_dose'] = df_insulin['total_ins_dose'] - df_insulin['basal_ins_dose']
        df_insulin = df_insulin[['id', 'visit', 'total_ins_dose', 'basal_ins_dose', 'bolus_ins_dose']]
        insulin_visits = df_insulin['visit'].dropna().unique()
        df_visit_info = df_visit_info[df_visit_info['visit'].isin(insulin_visits)].copy()
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose', 'basal_ins_dose'])

        #CGM Data
        df_cgm = pd.read_csv(f'{csv_folder}/original/CloudAbbottCGM.txt', sep="|")
        df_cgm['PtID'] = df_cgm['PtID'].str.strip()
        df_cgm.rename(columns={'PtID':'id', 'DeviceDtTm':'timestamp', 'Glucose': 'glucose mmol/l'}, inplace=True)
        df_cgm.drop_duplicates(subset=['RecID'], inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%m/%d/%Y %I:%M:%S %p", errors ='coerce')
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean = self.get_taylor_time_bins(df=df_cgm_clean, offset=0, dy_available=False).sort_values(by=['timestamp'], ascending=True)
        df_cgm_clean = pd.merge_asof(df_cgm_clean, df_visit_info, by='id', left_on='timestamp', right_on='visit_date', direction='nearest', tolerance=TOLERANCE_PD_DAYS)
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'visit', 'visit_date', 'glucose mmol/l']]
        df_cgm_clean.sort_values(by=['id', 'timestamp'], inplace=True)
        df_visit_info_cgm_raw = df_cgm_clean[['id', 'visit', 'visit_date', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()

        df_visit_time_bin_frequency = (
            df_visit_info_cgm_raw.groupby(['visit', 'time_bin'])
            .size()
            .reset_index(name='n')
            .sort_values(by=['visit', 'n', 'time_bin'], ascending=[True, False, True])
        )
        df_visit_time_bin_frequency['max_n'] = df_visit_time_bin_frequency.groupby('visit')['n'].transform('max')
        tied_visit_time_bins = df_visit_time_bin_frequency[df_visit_time_bin_frequency['n'] == df_visit_time_bin_frequency['max_n']]
        tied_visits = tied_visit_time_bins.groupby('visit').size()
        tied_visits = tied_visits[tied_visits > 1]

        df_visit_time_bin_canonical = (
            df_visit_time_bin_frequency.drop_duplicates(subset=['visit'], keep='first')[['visit', 'time_bin']]
            .rename(columns={'time_bin': 'canonical_time_bin'})
            .sort_values(by='visit')
        )

        df_visit_time_bin_frequency.to_csv(f'{csv_folder}/csv_files/visit_time_bin_frequency.csv', index=False)
        df_visit_time_bin_canonical.to_csv(f'{csv_folder}/csv_files/visit_time_bin_canonical.csv', index=False)

        if not tied_visits.empty:
            tied_visit_summary = (
                tied_visit_time_bins[tied_visit_time_bins['visit'].isin(tied_visits.index)]
                .groupby('visit')['time_bin']
                .apply(lambda s: sorted(pd.unique(s)))
                .reset_index(name='tied_time_bins')
                .sort_values(by='visit')
            )
            raise ValueError(
                "Cannot build canonical visit-to-time_bin mapping because some visits have a tie for the most common time bin:\n"
                f"{tied_visit_summary.to_string(index=False)}"
            )

        df_cgm_clean = df_cgm_clean.merge(df_visit_time_bin_canonical, on='visit', how='left')
        df_cgm_clean['time_bin'] = df_cgm_clean['canonical_time_bin'].combine_first(df_cgm_clean['time_bin'])
        df_cgm_clean.drop(columns=['canonical_time_bin'], inplace=True)

        df_visit_info_cgm = df_cgm_clean[['id', 'visit', 'visit_date', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()
        df_visit_time_bin_audit = (
            df_visit_info_cgm.sort_values(by=['id', 'visit_date', 'visit', 'time_bin'])
            .reset_index(drop=True)
        )
        df_visit_time_bin_summary = (
            df_visit_info_cgm.groupby(['id', 'visit', 'visit_date'])['time_bin']
            .apply(lambda s: sorted(pd.unique(s)))
            .reset_index(name='time_bins')
            .sort_values(by=['id', 'visit_date', 'visit'])
        )
        df_visit_time_bin_summary['n_time_bins'] = df_visit_time_bin_summary['time_bins'].apply(len)

        df_visit_info_cgm.to_csv(f'{csv_folder}/csv_files/visit_info_cgm.csv', index=False)
        df_visit_time_bin_audit.to_csv(f'{csv_folder}/csv_files/visit_time_bin_audit.csv', index=False)
        df_visit_time_bin_summary.to_csv(f'{csv_folder}/csv_files/visit_time_bin_summary.csv', index=False)

        # df_hba1c_local = pd.read_csv(f'{csv_folder}/original/CloudDiabLocalHbA1c.txt', sep="|")
        # df_hba1c_local.rename(columns={'PtID':'id', 'HbA1cTestRes': 'hb_a1c_local'}, inplace=True)
        # df_hba1c_local['HbA1cTestDt'] = pd.to_datetime(df_hba1c_local['HbA1cTestDt'])
        # df_hba1c_local.sort_values(by=['id','HbA1cTestDt'], ascending=True, inplace=True)
        # df_hba1c_local.drop_duplicates(subset=['RecID','id'], inplace=True)
        # df_hba1c_local_clean = df_hba1c_local.dropna(subset=['hb_a1c_local'])
        # df_hba1c_local_clean = df_hba1c_local_clean[['id','visit','HbA1cTestDt', 'hb_a1c_local']]

        df_hba1c_cap = pd.read_csv(f'{csv_folder}/original/CloudLabDataHbA1cCap.txt', sep="|")
        df_hba1c_cap['PtID'] = df_hba1c_cap['PtID'].str.strip()
        df_hba1c_cap.rename(columns={'PtID':'id', 'VisitCollected':'time_bin'}, inplace=True)
        df_hba1c_cap.drop_duplicates(subset=['RecID'], inplace=True)
        df_hba1c_cap['time_bin'] = df_hba1c_cap['time_bin'].replace(
            {r'^(\d+)Mo$': r'Month \1', r'^Base$': 'Baseline'},
            regex=True
        )
        df_hba1c_cap['hb_a1c_cap'] = (df_hba1c_cap['HbA1cMMol']/ 10.929) + 2.15
        df_hba1c_cap.sort_values(by=['id','time_bin'], ascending=True, inplace=True)
        df_hba1c_cap = df_hba1c_cap[['id','time_bin', 'hb_a1c_cap']]
        df_hba1c_cap_clean = df_hba1c_cap.dropna(subset=['time_bin', 'hb_a1c_cap'])
        df_hba1c_cap_clean['time_bin'] = df_hba1c_cap_clean['time_bin'].replace(
            {'Visit15': 'Month 36', 'Visit17': 'Month 48'}
        )
        
        df_hba1c_ven = pd.read_csv(f'{csv_folder}/original/CloudLabDataHbA1cVen.txt', sep="|")
        df_hba1c_ven['PtID'] = df_hba1c_ven['PtID'].str.strip()
        df_hba1c_ven.rename(columns={'PtID':'id', 'VisitCollected':'time_bin'}, inplace=True)
        df_hba1c_ven.drop_duplicates(subset=['RecID'], inplace=True)
        df_hba1c_ven['time_bin'] = df_hba1c_ven['time_bin'].replace(
            {r'^(\d+)Mo$': r'Month \1', r'^Base$': 'Baseline'},
            regex=True
        )
        df_hba1c_ven['hb_a1c_ven'] = (df_hba1c_ven['HbA1cMMol']/ 10.929) + 2.15
        df_hba1c_ven.sort_values(by=['id','time_bin'], ascending=True, inplace=True)
        df_hba1c_ven = df_hba1c_ven[['id', 'time_bin', 'hb_a1c_ven']]
        df_hba1c_ven_clean = df_hba1c_ven.dropna(subset=['time_bin', 'hb_a1c_ven'])
        df_hba1c_ven_clean['time_bin'] = df_hba1c_ven_clean['time_bin'].replace(
            {'Visit15': 'Month 36', 'Visit17': 'Month 48'}
        )
        
        df_cpep = pd.read_csv(f'{csv_folder}/original/CloudLabCPeptide.txt', sep="|")
        df_cpep['PtID'] = df_cpep['PtID'].str.strip()
        df_cpep.rename(columns={'PtID':'id', 'VisitCollected': 'time_bin', 'CPeptide0Min': 'cpep_0_min',
                                'CPeptide10Min': 'cpep_pre10_min', 'CPeptide15Min': 'cpep_15_min', 'CPeptide30Min': 'cpep_30_min',
                                'CPeptide60Min': 'cpep_60_min', 'CPeptide90Min': 'cpep_90_min', 'CPeptide120Min': 'cpep_120_min'}, inplace=True)
        df_cpep.drop_duplicates(subset=['RecID'], inplace=True)
        df_cpep['time_bin'] = df_cpep['time_bin'].replace(
            {r'^(\d+) Months$': r'Month \1', r'^Baseline$': 'Baseline'},
            regex=True
        )
        df_cpep = df_cpep[['id', 'time_bin', 'cpep_pre10_min', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min']]
        cpep_cols = [col for col in df_cpep.columns if col.startswith("cpep_")]
        for col in cpep_cols:
            df_cpep[col] = df_cpep[col]/1000
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_30_min',
                                'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)
    
        df_glucose = pd.read_csv(f'{csv_folder}/original/CloudLabGlucose.txt', sep="|")
        df_glucose['PtID'] = df_glucose['PtID'].str.strip()
        df_glucose.rename(columns={'PtID': 'id', 'VisitCollected': 'time_bin', 'Glucose0Min': 'glucose_0_min',
                                'Glucose10Min': 'glucose_pre10_min', 'Glucose15Min': 'glucose_15_min', 'Glucose30Min': 'glucose_30_min',
                                'Glucose60Min': 'glucose_60_min', 'Glucose90Min': 'glucose_90_min', 'Glucose120Min': 'glucose_120_min'}, inplace=True)
        df_glucose.drop_duplicates(subset=['RecID'], inplace=True)
        df_glucose['time_bin'] = df_glucose['time_bin'].replace(
            {r'^(\d+) Months$': r'Month \1', r'^Baseline$': 'Baseline'},
            regex=True
        )
        df_glucose = df_glucose[['id', 'time_bin', 'glucose_pre10_min', 'glucose_0_min', 'glucose_15_min',
                              'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min']]
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_30_min',
                                                     'glucose_60_min', 'glucose_90_min', 'glucose_120_min'])

        df_extra = pd.read_csv(f'{csv_folder}/original/PtRoster.txt', sep="|")
        df_extra['PtID'] = df_extra['PtID'].str.strip()
        df_extra.rename(columns={'PtID':'id', 'AgeAtConsent': 'age', 'TrtGroup': 'treatment_arm'}, inplace=True)
        df_extra.drop_duplicates(subset=['RecID'], inplace=True)
        df_extra.sort_values(by=['id'], ascending=True, inplace=True)
        df_extra['insulin_delivery'] = None
        df_extra.loc[df_extra['treatment_arm'] == 'CL', 'insulin_delivery'] = 'Hybrid Closed-Loop'
        df_extra.loc[df_extra['treatment_arm'] == 'MDI', 'insulin_delivery'] = 'MDI'
        df_randdt = df_extra[['id', 'RandDt']]
        df_randdt.loc[:, 'RandDt'] = pd.to_datetime(df_extra['RandDt'], format="%m/%d/%Y %I:%M:%S %p", errors ='coerce')
        df_extra = df_extra[['id', 'age', 'treatment_arm', 'insulin_delivery']]
        df_extra_clean = df_extra.dropna(subset=['age'])

        #TODO: Use CloudRecruitment.txt for baseline height
        df_height_weight = pd.read_csv(f'{csv_folder}/original/CloudMMTTProced.txt', sep="|")
        df_height_weight['PtID'] = df_height_weight['PtID'].str.strip()
        df_height_weight.rename(columns={'PtID':'id', 'Visit':'visit', 'Height':'height', 'Weight':'weight'}, inplace=True)
        df_height_weight.drop_duplicates(subset=['RecID'], inplace=True)
        df_height_weight = df_height_weight[['id', 'visit', 'height','weight']]
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])
        
        df_demographic = pd.read_csv(f'{csv_folder}/original/CloudRecruitment.txt', sep="|")
        df_demographic['PtID'] = df_demographic['PtID'].str.strip()
        df_demographic.rename(columns={'PtID':'id', 'Sex':'sex', 'Ethnicity':'ethnicity', 'Race':'race', 'DiagDt' : 'diagnose_date'}, inplace=True)
        df_demographic.drop_duplicates(subset=['RecID'], inplace=True)
        df_demographic = df_demographic[['id', 'sex', 'ethnicity', 'race', 'diagnose_date']]
        df_demographic['sex'] = df_demographic['sex'].map({'M': 0, 'F': 1})

        df_final = df_cgm_clean.copy()
        print(df_final.shape)
        df_final = df_final.merge(df_extra, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_demographic, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_height_weight, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_hba1c_cap_clean, on=['id','time_bin'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_hba1c_ven_clean, on=['id','time_bin'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_cpep, on=['id','time_bin'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_glucose, on=['id','time_bin'], how='left') # Bucket 2
        print(df_final.shape)
        df_final = df_final.merge(df_insulin_clean, on=['id','visit'], how='left') # Bucket 2
        print(df_final.shape)
        
        df_final.rename(columns={'hb_a1c_cap': 'hb_a1c'}, inplace=True)
        df_final = self.fill_static_within_time_bin(df_final)
        df_final['cpep_pre10_min'] = df_final['cpep_pre10_min'].fillna(df_final['cpep_0_min'])
        df_final.sort_values(by=['id', 'timestamp'], inplace=True)
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')

        num_people = no_dup_df.shape[0]

        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()

        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

        c_peptide_interval = ''
        beta2_interval = ''
        cgm_interval = '15 minutes'

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}
        self.print_consort_general(df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c_cap, df_hba1c_ven, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_cap_clean, df_hba1c_ven_clean, df_insulin_clean],
                           df_final=df_final)
        
        df_height_weight_clean = df_height_weight_clean.merge(df_visit_info_cgm[['id', 'visit', 'time_bin']].drop_duplicates(), on=['id', 'visit'], how='left')
        df_insulin_clean = df_insulin_clean.merge(df_visit_info_cgm[['id', 'visit', 'time_bin']].drop_duplicates(), on=['id', 'visit'], how='left')
        time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12', 'Month 15', 'Month 18', 'Month 21', 'Month 24']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c_cap, df_hba1c_ven, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_cap_clean, df_hba1c_ven_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_cap_clean', 'df_hba1c_ven_clean', 'df_insulin_clean'])

        if save_csv:
            df_cgm.to_csv('./data/studies/cloud/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/cloud/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c_cap.to_csv('./data/studies/cloud/csv_files/hb_a1c_cap/df_original.csv', index=False)
            df_hba1c_cap_clean.to_csv('./data/studies/cloud/csv_files/hb_a1c_cap/df_clean.csv', index=False)
            df_hba1c_ven.to_csv('./data/studies/cloud/csv_files/hb_a1c_ven/df_original.csv', index=False)
            df_hba1c_ven_clean.to_csv('./data/studies/cloud/csv_files/hb_a1c_ven/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/cloud/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/cloud/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/cloud/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/cloud/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/cloud/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/cloud/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/cloud/csv_files/weight_height/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/cloud/csv_files/weight_height/df_clean.csv', index=False)
            df_extra.to_csv('./data/studies/cloud/csv_files/extra/df_original.csv', index=False)
            df_extra_clean.to_csv('./data/studies/cloud/csv_files/extra/df_clean.csv', index=False)
            df_visit_info.to_csv('./data/studies/cloud/csv_files/visit_info.csv', index=False)
            df_final.to_csv('./data/studies/cloud/df_final.csv', index=False)

        return df_final, pd.DataFrame(df_summary)

    def preprocess_clvr(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the CLVR dataset: cleans CGM, aligns visits, merges demographics/labs/MMTT,
        computes C‑peptide AUC and BETA2, normalizes units, and outputs a final table plus summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One‑row summary statistics table.
        """  
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/cgmAnalysis.txt', sep="|")
        df_cgm.rename(columns={'PtID':'id', 'DeviceDtTm':'timestamp', 'glucose': 'glucose mmol/l', 'Visit':'visit'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%d%b%Y:%H:%M:%S.%f", errors ='coerce')
        df_cgm = self.get_taylor_time_bins(df=df_cgm, offset=15, dy_available=False)
        df_cgm.loc[df_cgm['time_bin'] == 'Baseline', 'visit'] = 'Randomization'
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm_clean['timestamp_seconds'] = df_cgm_clean.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds())
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean.loc[:,'glucose mmol/l'] = df_cgm_clean[['glucose mmol/l']]/18
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'visit', 'glucose mmol/l']]
        df_visit_info_raw = df_cgm_clean[['id', 'visit', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()
        df_visit_time_bin_frequency = (
            df_visit_info_raw.groupby(['visit', 'time_bin'])
            .size()
            .reset_index(name='n')
            .sort_values(by=['visit', 'n', 'time_bin'], ascending=[True, False, True])
        )
        df_visit_time_bin_frequency['max_n'] = df_visit_time_bin_frequency.groupby('visit')['n'].transform('max')
        tied_visit_time_bins = df_visit_time_bin_frequency[df_visit_time_bin_frequency['n'] == df_visit_time_bin_frequency['max_n']]
        tied_visits = tied_visit_time_bins.groupby('visit').size()
        tied_visits = tied_visits[tied_visits > 1]

        df_visit_time_bin_canonical = (
            df_visit_time_bin_frequency.drop_duplicates(subset=['visit'], keep='first')[['visit', 'time_bin']]
            .rename(columns={'time_bin': 'canonical_time_bin'})
            .sort_values(by='visit')
        )

        df_visit_time_bin_frequency.to_csv(f'{csv_folder}/csv_files/visit_time_bin_frequency.csv', index=False)
        df_visit_time_bin_canonical.to_csv(f'{csv_folder}/csv_files/visit_time_bin_canonical.csv', index=False)

        if not tied_visits.empty:
            tied_visit_summary = (
                tied_visit_time_bins[tied_visit_time_bins['visit'].isin(tied_visits.index)]
                .groupby('visit')['time_bin']
                .apply(lambda s: sorted(pd.unique(s)))
                .reset_index(name='tied_time_bins')
                .sort_values(by='visit')
            )
            raise ValueError(
                "Cannot build canonical visit-to-time_bin mapping for CLVR because some visits have a tie for the most common time bin:\n"
                f"{tied_visit_summary.to_string(index=False)}"
            )

        df_cgm_clean = df_cgm_clean.merge(df_visit_time_bin_canonical, on='visit', how='left')
        df_cgm_clean['time_bin'] = df_cgm_clean['canonical_time_bin'].combine_first(df_cgm_clean['time_bin'])
        df_cgm_clean.drop(columns=['canonical_time_bin'], inplace=True)

        df_visit_info = df_cgm_clean[['id', 'visit', 'time_bin']].dropna(subset=['visit', 'time_bin']).drop_duplicates()
        df_visit_time_bin_audit = (
            df_visit_info.sort_values(by=['id', 'visit', 'time_bin'])
            .reset_index(drop=True)
        )
        df_visit_time_bin_summary = (
            df_visit_info.groupby(['id', 'visit'])['time_bin']
            .apply(lambda s: sorted(pd.unique(s)))
            .reset_index(name='time_bins')
            .sort_values(by=['id', 'visit'])
        )
        df_visit_time_bin_summary['n_time_bins'] = df_visit_time_bin_summary['time_bins'].apply(len)

        df_visit_info.to_csv(f'{csv_folder}/csv_files/visit_info_cgm.csv', index=False)
        df_visit_time_bin_audit.to_csv(f'{csv_folder}/csv_files/visit_time_bin_audit.csv', index=False)
        df_visit_time_bin_summary.to_csv(f'{csv_folder}/csv_files/visit_time_bin_summary.csv', index=False)

        #Extra features
        df_demographic = pd.read_csv(f'{csv_folder}/original/subjectsEnroll.txt', sep="|")
        df_demographic.rename(columns={'PtID':'id', 'Gender':'sex', 'Ethnicity':'ethnicity', 'Race':'race'}, inplace=True)
        df_demographic.drop_duplicates(subset=['id'], inplace=True)
        df_demographic.sort_values(by=['id'], ascending=True, inplace=True)
        df_demographic['insulin_delivery'] = df_demographic['hclGrp'].str.replace(r'^\d+\.', '', regex=True).str.strip()
        df_demographic['treatment_arm'] = df_demographic['drugGrp'].str.replace(r'^\d+\.', '', regex=True).str.strip()
        # df_demographic = df_demographic[df_demographic['treatment_arm'].notna() & (df_demographic['treatment_arm'].str.strip() != '')]
        df_demographic = df_demographic[['id', 'sex', 'ethnicity', 'race', 'treatment_arm', 'insulin_delivery']]
        df_demographic['sex'] = df_demographic['sex'].map({'M': 0, 'F': 1})

        df_height_weight = pd.read_csv(f'{csv_folder}/original/visits.txt', sep="|")
        df_height_weight.rename(columns={'PtID':'id', 'heightCm':'height', 'weightKg':'weight', 'Visit': 'visit', 'ageAtVisit': 'age',
                                     'a1cLab': 'hb_a1c', 'TDIUnits': 'total_ins_dose', 'BasalOrLongActIns' : 'basal_ins_dose'}, inplace=True)
        df_height_weight.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])

        df_mmtt = pd.read_csv(f'{csv_folder}/original/mmttResults.txt', sep="|")
        df_mmtt.rename(columns={'PtID':'id', 'Visit': 'visit'}, inplace=True)
        df_mmtt = df_mmtt.pivot(index=['id', 'visit', 'CollectionDt'], columns='ResultName', values='Value').reset_index()
        df_mmtt.rename(columns={'C-PEP-0': 'cpep_0_min', 'C-PEP-15': 'cpep_15_min', 'C-PEP-30': 'cpep_30_min',
                                'C-PEP-60': 'cpep_60_min', 'C-PEP-90': 'cpep_90_min', 'C-PEP-120': 'cpep_120_min',
                                'GLU-0': 'glucose_0_min', 'GLU-15': 'glucose_15_min', 'GLU-30': 'glucose_30_min',
                                'GLU-60': 'glucose_60_min', 'GLU-90': 'glucose_90_min', 'GLU-120': 'glucose_120_min'}, inplace=True)
        df_mmtt['CollectionDt'] = pd.to_datetime(df_mmtt['CollectionDt'], format = "%d%b%Y")

        df_cpep = df_mmtt[['id', 'visit', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']]
        cpep_cols = [col for col in df_cpep.columns if col.startswith("cpep_")]
        for col in cpep_cols:
            df_cpep[col] = df_cpep[col].apply(
                lambda x: 0.007 if isinstance(x, str) and "<" in x else pd.to_numeric(x, errors='coerce')
            )
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                                'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)

        df_glucose = df_mmtt[['id', 'visit', 'glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min']]
        glucose_cols = [col for col in df_glucose.columns if col.startswith("glucose_")]
        for col in glucose_cols:
            df_glucose[col] = pd.to_numeric(df_glucose[col], errors='coerce')
            df_glucose[col] = df_glucose[col]/18
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min',
                                'glucose_60_min', 'glucose_90_min', 'glucose_120_min'])

        df_hba1c = df_height_weight[['id', 'visit', 'hb_a1c']]
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])

        df_insulin = df_height_weight[['id', 'visit', 'total_ins_dose', 'basal_ins_dose']]
        df_insulin['bolus_ins_dose'] = df_insulin['total_ins_dose'] - df_insulin['basal_ins_dose']
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose', 'basal_ins_dose'])

        df_height_weight = df_height_weight[['id', 'visit','height', 'weight', 'age']]

        df_final = df_cgm_clean.copy()
        print(df_final.shape)
        df_final = df_final.merge(df_demographic, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_height_weight, on=['id', 'visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_hba1c_clean, on=['id', 'visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_cpep, on=['id', 'visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_glucose, on=['id','visit'], how='left') # Bucket 2
        print(df_final.shape)
        df_final = df_final.merge(df_insulin, on=['id', 'visit'], how='left') # Bucket 2
        print(df_final.shape)

        df_final = self.fill_static_within_time_bin(df_final)
        df_final = df_final.sort_values(by=['id', 'timestamp']).reset_index(drop=True)
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)

        df_final['id'] = 'CLVR_' + df_final['id'].astype('str')
        
        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')
        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]
        c_peptide_interval = '0-10-15-30-60-90-120 minutes'
        beta2_interval = ''
        cgm_interval = '5 minutes'
        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}
        
        self.print_consort_general(df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final)
        
        df_visit_info = df_visit_time_bin_canonical.rename(columns={'canonical_time_bin': 'time_bin'})
        df_cpep_clean = df_cpep_clean.merge(df_visit_info, on=['visit'], how='left')
        df_glucose_clean = df_glucose_clean.merge(df_visit_info, on=['visit'], how='left')
        df_height_weight_clean = df_height_weight_clean.merge(df_visit_info, on=['visit'], how='left')
        df_insulin_clean = df_insulin_clean.merge(df_visit_info, on=['visit'], how='left')
        df_hba1c_clean = df_hba1c_clean.merge(df_visit_info, on=['visit'], how='left')
        time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_clean', 'df_insulin_clean'])
              
        if save_csv:
            df_cgm.to_csv('./data/studies/clvr/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/clvr/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/clvr/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/clvr/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/clvr/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/clvr/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/clvr/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/clvr/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/clvr/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/clvr/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/clvr/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/clvr/csv_files/height_weight/df_clean.csv', index=False)
            df_final.to_csv('./data/studies/clvr/df_final.csv', index=False)

        # if 'insulin_delivery' in df_final.columns:
        #     df_final = df_final.drop(columns=['treatment_arm'], errors='ignore')
        #     df_final = df_final.rename(columns={'insulin_delivery': 'treatment_arm'})
        #     df_final.dropna(subset=['treatment_arm'], inplace=True)
            
        return df_final, pd.DataFrame(df_summary)

    def preprocess_defend(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the DEFEND‑2 dataset: cleans CGM, aligns by study day, merges labs/MMTT/demographics,
        computes C‑peptide AUC and BETA2, normalizes units, and outputs a final table plus a summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One‑row summary statistics table.
        """

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/defend_2.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l', 'time_bins': 'time_bin'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'dy', 'glucose mmol/l']]
        df_visit_info = df_cgm_clean[['time_bin','dy']].drop_duplicates(subset=['dy'])

        df_extra_features = pd.read_csv(f'{csv_folder}/original/extra_features.csv')
        df_extra_features.rename(columns={'PtID':'id', 'plcb': 'treatment_arm', 'time_bins': 'time_bin', 'DaysFromEnroll': 'dy'} , inplace=True)

        df_demographic = df_extra_features[['id', 'sex', 'race', 'treatment_arm']]
        df_demographic.drop_duplicates(subset=['id'], inplace=True)
        df_demographic['sex'] = df_demographic['sex'].map({'Male': 0, 'Female': 1})
        df_demographic['insulin_delivery'] = df_demographic['treatment_arm'].map({
            'Control': 'Standard Care',
            'Active': 'Intensive Care',
        })

        df_height_weight = df_extra_features[['id', 'dy', 'age', 'weight', 'height']]
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])

        df_cpep = df_extra_features[['id', 'dy', 'time_bin', 'cpep_fast', 'cpepm10', 'cpep0',
                                     'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
        df_cpep.rename(columns={'cpepm10': 'cpep_pre10_min','cpep0': 'cpep_0_min','cpep15': 'cpep_15_min',
                                'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min','cpep90': 'cpep_90_min',
                                 'cpep120': 'cpep_120_min'}, inplace=True)        
        cpep_cols = [col for col in df_cpep.columns if col.startswith("cpep_")]
        for col in cpep_cols:
            df_cpep[col] = df_cpep[col]*0.331
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_15_min',
                            'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)

        df_glucose = df_extra_features[['id', 'dy', 'time_bin', 'glu0',
                                        'glu15', 'glu30', 'glu60', 'glu90', 'glu120']]
        df_glucose.rename(columns={'glu0': 'glucose_0_min','glu15': 'glucose_15_min',
                                'glu30': 'glucose_30_min', 'glu60': 'glucose_60_min',
                                'glu90': 'glucose_90_min', 'glu120': 'glucose_120_min'}, inplace=True)
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                                  'glucose_90_min', 'glucose_120_min'])
        glucose_cols = [col for col in df_glucose.columns if col.startswith("glucose_")]
        for col in glucose_cols:
            df_glucose[col] = pd.to_numeric(df_glucose[col], errors='coerce')
            df_glucose[col] = df_glucose[col]/18
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                                  'glucose_90_min', 'glucose_120_min'])

        df_insulin = df_extra_features[['id', 'dy', 'insulin']]
        df_insulin.rename(columns={'insulin' :'total_ins_dose'}, inplace=True)
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose'])

        df_hba1c = df_extra_features[['id', 'dy', 'hba1c']]
        df_hba1c.rename(columns={'hba1c': 'hb_a1c'}, inplace=True)
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])

        df_cgm_windows = (
            df_cgm_clean.groupby(['id', 'time_bin'], as_index=False)['dy']
            .min()
            .rename(columns={'dy': 'cgm_window_dy'})
            .sort_values(by=['cgm_window_dy', 'id'])
            .reset_index(drop=True)
        )

        common_ids = (set(df_cpep['id']) & set(df_hba1c_clean['id']) & set(df_cgm_windows['id']) & set(df_glucose['id'])
                       & set(df_height_weight['id']) & set(df_insulin_clean['id'])) # Bucket 2

        df_window_features = df_cgm_windows[df_cgm_windows['id'].isin(common_ids)].copy()
        df_window_features.sort_values(by=['cgm_window_dy', 'id'], inplace=True)

        df_cpep_merge = df_cpep[df_cpep['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_glucose_merge = df_glucose[df_glucose['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_hba1c_merge = df_hba1c_clean[df_hba1c_clean['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_insulin_merge = df_insulin_clean[df_insulin_clean['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_height_weight_merge = df_height_weight[df_height_weight['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)

        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_cpep_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_glucose_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_height_weight_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_hba1c_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_insulin_merge)

        df_window_features.rename(columns={'cgm_window_dy': 'dy'}, inplace=True)

        df_final = df_cgm_clean[df_cgm_clean['id'].isin(common_ids)].copy()
        print(df_final.shape)
        df_final = df_final.merge(df_demographic, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_window_features, on=['id', 'time_bin', 'dy'], how='left')
        print(df_final.shape)
        df_final.sort_values(by=['id','timestamp'], inplace=True)

        df_final = self.fill_static_within_time_bin(df_final, extra_exclude={'dy'})
        df_diagnose = df_extra_features[['id', 'dy', 'diag_dy']]
        df_final = df_final.merge(df_diagnose, left_on=['id','dy'], right_on=['id', 'dy'], how='left')
        print(df_final.shape)
        df_final['diag_dy'] = pd.to_numeric(df_final['diag_dy'], errors='coerce')
        df_final['diagnose_date'] = df_final['timestamp'] + pd.to_timedelta(df_final['diag_dy'], unit='D') + pd.to_timedelta(df_final['dy'], unit='D')
        df_final['diagnose_date'] = (df_final.groupby('id', group_keys=False)['diagnose_date'].transform(lambda s: s.dropna().min())).dt.date
        df_final.drop(columns=['diag_dy'], inplace=True)
        df_final['cpep_pre10_min'] = df_final['cpep_pre10_min'].fillna(df_final['cpep_0_min'])
        df_final['total_ins_dose'] = df_final['total_ins_dose'] * df_final['weight']
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')
        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()
        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]
        c_peptide_interval = '0,15,30:120:30 '
        beta2_interval = {'N/A'}
        cgm_interval = '5 mins'

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}
        self.print_consort_general(df_list_raw=[df_cgm, df_cpep, df_height_weight_clean, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final)
        
        df_height_weight_clean = df_height_weight_clean.merge(df_visit_info, on=['dy'], how='left')
        df_insulin_clean = df_insulin_clean.merge(df_visit_info, on=['dy'], how='left')
        df_hba1c_clean = df_hba1c_clean.merge(df_visit_info, on=['dy'], how='left')
        time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 12']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight_clean, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_clean', 'df_insulin_clean'])
        if save_csv:
            df_cgm.to_csv('./data/studies/defend/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/defend/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/defend/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/defend/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/defend/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/defend/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/defend/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/defend/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/defend/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/defend/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/defend/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/defend/csv_files/height_weight/df_clean.csv', index=False)
            df_final.to_csv('./data/studies/defend/df_final.csv', index=False)

        return df_final, pd.DataFrame(df_summary)
    
    def preprocess_diagnode(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the DIAGNODE dataset: cleans CGM, merges screening/MMTT/insulin/HbA1c,
        computes C‑peptide AUC and BETA2, and outputs a final longitudinal table plus a summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One‑row summary statistics table.
        """

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/CGM_RANDOMISED_preprocessed.csv')

        df_cgm[['id','visit','device_id']] = df_cgm['ID_VISIT_DEVICEID'].str.split('_',expand=True)
        df_cgm = df_cgm[['id','visit','TIMESTAMP','GLUCOSE']]
        df_cgm.rename(columns={'TIMESTAMP': 'timestamp', 'GLUCOSE': 'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format = "%Y-%m-%d %H:%M:%S", errors ='coerce')
        df_cgm['visit'] = df_cgm['visit'].replace(
            {'Visit 1': 'Visit 2'}
        )
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'visit', 'glucose mmol/l']]

        #Extra feature
        df_aljc = pd.read_csv(f'{csv_folder}/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')
        df_aljc.drop_duplicates(subset=['id', 'visit'], inplace=True)
        df_visit_info = df_aljc[['visit', 'visit_name']]
        df_visit_info.drop_duplicates(subset=['visit'], inplace=True)
        df_visit_info.rename(columns={'visit_name': 'time_bin'}, inplace=True)
        df_cgm_clean = df_cgm_clean.merge(df_visit_info, on=['visit'], how='left')

        df_height_weight = df_aljc[['id','visit','treatment','sex','weight','height','age', 'days_from_diagnosis']]
        df_height_weight.rename(columns={'treatment': 'treatment_arm', 'days_from_diagnosis':'dy'}, inplace=True)
        df_height_weight.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])
        
        df_cpep = df_aljc[['id', 'visit', 'cpep0', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
        df_cpep.rename(columns={'cpep0': 'cpep_0_min', 'cpep30': 'cpep_30_min',
                                'cpep60': 'cpep_60_min', 'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min'}, inplace=True)
        df_cpep.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_30_min',
                                'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)

        df_glucose = df_aljc[['id', 'visit', 'gluc0', 'gluc30', 'gluc60',
                              'gluc90', 'gluc120', 'glucose_auc120', 'glucose_fasting']]
        df_glucose.rename(columns={'gluc0': 'glucose_0_min', 'gluc30': 'glucose_30_min', 'gluc60': 'glucose_60_min',
                                   'gluc90': 'glucose_90_min', 'gluc120': 'glucose_120_min'}, inplace=True)
        df_glucose.sort_values(by=['id', 'visit'], inplace=True)
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_30_min', 'glucose_60_min', 'glucose_90_min', 'glucose_120_min'])

        df_insulin = df_aljc[['id', 'visit', 'insulin_dose_upkgpday', 'insulin_pump']]
        df_insulin.rename(columns={'insulin_dose_upkgpday': 'total_ins_dose', 'insulin_pump': 'insulin_delivery'}, inplace=True)
        df_insulin.sort_values(by=['id', 'visit'], inplace=True)
        df_insulin['insulin_delivery'] = df_insulin['insulin_delivery'].map({'No': 'No insulin pump', 'Yes': 'Insulin pump'})
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose'])

        df_hba1c = df_aljc[['id', 'visit', 'hba1c_per']]
        df_hba1c.rename(columns={'hba1c_per' : 'hb_a1c'}, inplace=True)
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])
        df_hba1c_clean.sort_values(by=['id','visit'], ascending=True, inplace=True)

        #Merged dataframe
        df_final = df_cgm_clean.copy()
        print(df_final.shape)
        df_final = df_final.merge(df_height_weight, on=['id','visit'], how='inner')
        print(df_final.shape)
        df_final = df_final.merge(df_hba1c_clean, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_cpep, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_glucose, on=['id','visit'], how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_insulin, on=['id','visit'], how='left')
        print(df_final.shape)

        df_final = self.fill_static_within_time_bin(df_final)
        df_final['dy'] = pd.to_numeric(df_final['dy'], errors='coerce')
        df_final['diagnose_date'] = df_final['timestamp'] - pd.to_timedelta(df_final['dy'], unit='D')
        df_final['diagnose_date'] = (df_final.groupby('id')['diagnose_date'].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NaT))
        df_final['diagnose_date'] = df_final['diagnose_date'].dt.date

        df_final = df_final[['id','timestamp', 'timestamp_type', 'time_bin', 'diagnose_date', 'glucose mmol/l', 'age',
                             'weight', 'height', 'sex', 'treatment_arm', 'insulin_delivery', 'cpep_0_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min',
                             'cpep_120_min', 'cpep_auc', 'hb_a1c', 'glucose_0_min', 'glucose_30_min', 'glucose_60_min',
                             'glucose_90_min', 'glucose_120_min', 'total_ins_dose']]
        df_final['sex'] = df_final['sex'].map({'Male': 0, 'Female': 1})
        df_final['weight'] = pd.to_numeric(df_final['weight'], errors='coerce')
        df_final['total_ins_dose'] = df_final['total_ins_dose'] * df_final['weight']
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)
        df_final['id'] = 'Diagnode_' + df_final['id']
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        '''Data info:'''
        no_dup_df = df_final.drop_duplicates(subset='id')
        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()
        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]
        c_peptide_interval = '30 minutes'
        beta2_interval = '0-6-15-24 months'
        cgm_interval = '15 minutes'
        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}
        
        self.print_consort_general(df_list_raw=[df_cgm, df_hba1c, df_insulin, df_height_weight, df_cpep, df_glucose],
                           df_list_clean=[df_cgm_clean, df_hba1c_clean, df_insulin_clean, df_height_weight_clean, df_cpep_clean, df_glucose_clean],
                           df_final=df_final)

        df_cpep_clean = df_cpep_clean.merge(df_visit_info, on=['visit'], how='left')
        df_height_weight_clean = df_height_weight_clean.merge(df_visit_info, on=['visit'], how='left')
        df_glucose_clean = df_glucose_clean.merge(df_visit_info, on=['visit'], how='left')
        df_hba1c_clean = df_hba1c_clean.merge(df_visit_info, on=['visit'], how='left')
        df_insulin_clean = df_insulin_clean.merge(df_visit_info, on=['visit'], how='left')
        time_bins = ['Baseline', 'Month 6', 'Month 15']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_clean', 'df_insulin_clean'])

        if save_csv:
            df_cgm.to_csv('./data/studies/diagnode/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/diagnode/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/diagnode/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/diagnode/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/diagnode/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/diagnode/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/diagnode/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/diagnode/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/diagnode/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/diagnode/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/diagnode/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/diagnode/csv_files/height_weight/df_clean.csv', index=False)
            df_final.to_csv('./data/studies/diagnode/df_final.csv', index=False)

        return df_final, pd.DataFrame(df_summary)
    
    def preprocess_gskalb(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the GSKALB dataset: cleans CGM, aligns by study day, merges labs/MMTT/demographics,
        computes C‑peptide AUC and BETA2, normalizes units, and outputs a final table plus a summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One‑row summary statistics table.
        """

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/gskalb.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l', 'time_bins': 'time_bin'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'dy', 'glucose mmol/l']]
        df_visit_info = df_cgm_clean[['time_bin','dy']].drop_duplicates(subset=['dy'])

        df_extra_features = pd.read_csv(f'{csv_folder}/original/extra_features.csv')
        df_extra_features.rename(columns={'PtID':'id', 'plcb': 'treatment_arm', 'time_bins': 'time_bin', 'DaysFromEnroll': 'dy'} , inplace=True)
        df_extra_features.loc[(df_extra_features['dy'] >= 390) & (df_extra_features['dy'] < 480), 'time_bin'] = 'Month 15'

        df_demographic = df_extra_features[['id', 'sex', 'race', 'treatment_arm']]
        df_demographic.drop_duplicates(subset=['id'], inplace=True)
        df_demographic['sex'] = df_demographic['sex'].map({'Male': 0, 'Female': 1})

        df_height_weight = df_extra_features[['id', 'dy', 'time_bin', 'age', 'weight', 'height']]
        df_height_weight = (
            df_height_weight.groupby(["id", "dy", "time_bin"], as_index=False)[['weight', 'height','age']]
            .median()
        )
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])

        df_cpep = df_extra_features[['id', 'dy', 'time_bin', 'cpep_fast', 'cpepm10', 'cpep0',
                                     'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
        df_cpep.rename(columns={'cpepm10': 'cpep_pre10_min','cpep0': 'cpep_0_min','cpep15': 'cpep_15_min',
                                'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min','cpep90': 'cpep_90_min',
                                 'cpep120': 'cpep_120_min'}, inplace=True)
        cpep_cols = [col for col in df_cpep.columns if col.startswith("cpep_")]
        for col in cpep_cols:
            df_cpep[col] = df_cpep[col]*0.331
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_15_min',
                            'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)

        df_glucose = df_extra_features[['id', 'dy', 'time_bin', 'glu0',
                                        'glu15', 'glu30', 'glu60', 'glu90', 'glu120']]
        df_glucose.rename(columns={'glu0': 'glucose_0_min','glu15': 'glucose_15_min',
                                'glu30': 'glucose_30_min', 'glu60': 'glucose_60_min',
                                'glu90': 'glucose_90_min', 'glu120': 'glucose_120_min'}, inplace=True)
        glucose_cols = [col for col in df_glucose.columns if col.startswith("glucose_")]
        for col in glucose_cols:
            df_glucose[col] = pd.to_numeric(df_glucose[col], errors='coerce')
            df_glucose[col] = df_glucose[col]/18
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                                  'glucose_90_min', 'glucose_120_min'])
        
        df_insulin = df_extra_features[['id', 'dy', 'time_bin', 'insulin']]
        df_insulin.rename(columns={'insulin' :'total_ins_dose'}, inplace=True)
        df_insulin = (
            df_insulin.groupby(["id", "dy", "time_bin"], as_index=False)['total_ins_dose']
            .median()
        )
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose'])

        df_hba1c = df_extra_features[['id', 'dy', 'time_bin', 'hba1c']]
        df_hba1c.rename(columns={'hba1c': 'hb_a1c'}, inplace=True)
        df_hba1c = (
            df_hba1c.groupby(["id", "dy", "time_bin"], as_index=False)['hb_a1c']
            .median()
        )
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])

        df_cgm_windows = (
            df_cgm_clean.groupby(['id', 'time_bin'], as_index=False)['dy']
            .min()
            .rename(columns={'dy': 'cgm_window_dy'})
            .sort_values(by=['cgm_window_dy', 'id'])
            .reset_index(drop=True)
        )

        common_ids = (set(df_cpep['id']) & set(df_hba1c_clean['id']) & set(df_cgm_windows['id']) & set(df_glucose['id'])
                       & set(df_height_weight['id']) & set(df_insulin['id']))

        df_window_features = df_cgm_windows[df_cgm_windows['id'].isin(common_ids)].copy()
        df_window_features.sort_values(by=['cgm_window_dy', 'id'], inplace=True)

        df_cpep_merge = df_cpep[df_cpep['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_glucose_merge = df_glucose[df_glucose['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_hba1c_merge = df_hba1c_clean[df_hba1c_clean['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_insulin_merge = df_insulin[df_insulin['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_height_weight_merge = df_height_weight[df_height_weight['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)

        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_cpep_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_hba1c_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_height_weight_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_insulin_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_glucose_merge)

        df_window_features.rename(columns={'cgm_window_dy': 'dy'}, inplace=True)

        df_final = df_cgm_clean[df_cgm_clean['id'].isin(common_ids)].copy()
        print(df_final.shape)
        df_final = df_final.merge(df_demographic, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_window_features, on=['id', 'time_bin', 'dy'], how='left')
        print(df_final.shape)

        df_final = self.fill_static_within_time_bin(df_final, extra_exclude={'dy'})
        df_diagnose = df_extra_features[['id', 'dy', 'diag_dy']]
        df_final = df_final.merge(df_diagnose, left_on=['id','dy'], right_on=['id', 'dy'], how='left')
        df_final['diag_dy'] = pd.to_numeric(df_final['diag_dy'], errors='coerce')
        df_final['diagnose_date'] = df_final['timestamp'] + pd.to_timedelta(df_final['diag_dy'], unit='D') + pd.to_timedelta(df_final['dy'], unit='D')
        df_final['diagnose_date'] = (df_final.groupby('id', group_keys=False)['diagnose_date'].transform(lambda s: s.dropna().min())).dt.date
        df_final.drop(columns=['diag_dy'], inplace=True)
        df_final['cpep_pre10_min'] = df_final['cpep_pre10_min'].fillna(df_final['cpep_0_min'])
        df_final['total_ins_dose'] = df_final['total_ins_dose'] * df_final['weight']
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')
        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()
        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]
        c_peptide_interval = '0,15,30:120:30 '
        beta2_interval = {'N/A'}
        cgm_interval = '5 mins'
        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}

        self.print_consort_general(df_list_raw=[df_cgm, df_hba1c, df_insulin, df_height_weight, df_cpep, df_glucose],
                           df_list_clean=[df_cgm_clean, df_hba1c_clean, df_insulin_clean, df_height_weight_clean, df_cpep_clean, df_glucose_clean],
                           df_final=df_final)
        
        time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 12', 'Month 15']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_clean', 'df_insulin_clean'])

        if save_csv:
            df_cgm.to_csv('./data/studies/gskalb/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/gskalb/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/gskalb/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/gskalb/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/gskalb/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/gskalb/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/gskalb/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/gskalb/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/gskalb/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/gskalb/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/gskalb/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/gskalb/csv_files/height_weight/df_clean.csv', index=False)
            df_final.to_csv('./data/studies/gskalb/df_final.csv', index=False)

        return df_final, pd.DataFrame(df_summary)

    def preprocess_hupa_ucm(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the HUPA-UCM dataset loaded via the hupa loader, add timestamp metadata, and summarize.

        Args:
            csv_folder: Path where the processed `df_final.csv` will be written.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Final processed dataset and a one-row summary table.
        """
        hupa_loader = get_hupa_dataloader()
        df_cgm = hupa_loader.get_cgm_data()
        df_final = hupa_loader.get_final_data()
        df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], format="%Y-%m-%dT%H:%M:%S")
        df_final['Hour'] = df_final['timestamp'].dt.hour
        df_final['timestamp_type'] = df_final['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')

        df_final['Date'] = df_final['timestamp'].dt.date
        daily = df_final.groupby(['id','Date'])['total_ins_dose'].sum().reset_index()
        df_final.drop(['total_ins_dose'], axis=1, inplace=True)
        df_final = df_final.merge(daily, on=['id','Date'])
    
        df_final = self.get_taylor_time_bins(df_final, offset=0, dy_available=False)
        self.get_gmi(df_final)
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')

        num_people = no_dup_df['id'].size

        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]

        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

        c_peptide_interval = 'N/A'
        beta2_interval = 'N/A'
        cgm_interval = '15 mins'

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}

        return df_final, pd.DataFrame(df_summary)
    
    def preprocess_itx(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the Islet Transplant (iTx) dataset: loads CGM/final tables via the iTx loader,
        bins timestamps into 3‑month windows, prefixes IDs, and returns the final table plus a summary.

        Args:
            csv_folder: Path where the processed `df_final.csv` will be written.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with time bins and features.
                - pd.DataFrame: One‑row summary statistics table.
        """

        itx_loader = get_itx_dataloader()
        df_cgm = itx_loader.get_cgm_data()
        df_final = itx_loader.get_final_data()
        df_final = self.get_taylor_time_bins(df_final, offset=0, dy_available=False)
        df_final['id'] = 'iTx_' + df_final['id']
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')

        num_people = no_dup_df['id'].size

        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]

        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

        c_peptide_interval = 'N/A'
        beta2_interval = {}
        cgm_interval = {'1' : '15 minutes',
                        '2' : '15 minutes',
                        '3' : '15 minutes',
                        '4' : '15 minutes',
                        '5' : '15 minutes',
                        '6' : '5 minutes',
        }

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}

        return df_final, pd.DataFrame(df_summary)

    def preprocess_jaeb_healthy(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the JAEB healthy dataset: cleans CGM, adds demographics, converts glucose units,
        creates 3-month time bins, and outputs a final table plus a summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One-row summary statistics table.
        """

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/NonDiabDeviceCGM.csv')
        df_cgm.rename(columns={'PtID': 'id', 'DeviceDtDaysFromEnroll': 'dy'}, inplace=True)
        df_cgm = df_cgm[df_cgm['RecordType'] == 'CGM']

        df_cgm['glucose mmol/l'] = df_cgm['Value'] / 18 #Converting mg/dl to mmol/l
        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm.drop(columns=('Value'), axis=1)

        #Cleaning timestamp
        df_cgm.loc[:, 'timestamp_seconds'] = pd.to_timedelta(df_cgm['DeviceTm']).dt.total_seconds() + df_cgm['dy'] * 86400
        df_cgm['time_diff'] = df_cgm.groupby('id')['timestamp_seconds'].diff()

        initial_date = pd.to_datetime("2024-01-01")
        df_cgm['timestamp'] = initial_date + pd.to_timedelta(df_cgm['dy'], unit='D')
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'].astype(str) + ' ' + df_cgm['DeviceTm'])

        # df_cons_intervals = df_cgm[(df_cgm['time_diff'] < 930) & (df_cgm['time_diff']>870)]
        # const_ratio = df_cons_intervals.shape[0]/df_cgm.shape[0]
        # print(const_ratio)

        df_cgm.drop(['RecordType', 'time_diff', 'DeviceTm', 'Value'], axis=1, inplace=True)

        #Adding Age
        df_age = pd.read_csv(f'{csv_folder}/original/NonDiabPtRoster.csv')
        df_age.rename(columns={'PtID': 'id'}, inplace=True)        
        df_age.drop_duplicates(['RecID', 'id'], inplace=True)
        df_age = df_age[['id','AgeAsOfEnrollDt']]

        #Extra features
        df_screening = pd.read_csv(f'{csv_folder}/original/NonDiabScreening.csv')
        df_screening.rename(columns={'PtID': 'id'}, inplace=True)
        df_screening.drop_duplicates(['RecID', 'id'], inplace=True)

        df_screening = df_screening[['id','Weight', 'Height', 'HbA1c', 'Gender', 'Race']]
        df_screening['Gender'] = df_screening['Gender'].map({'M' : 0, 'F': 1})

        df_final = df_cgm.copy()
        df_final = self.get_taylor_time_bins(df_final, offset=0, dy_available=False)
        df_final = df_final.merge(df_age, on='id', how='inner')
        df_final = df_final.merge(df_screening, on='id', how='inner')
        df_final.rename(columns={'DeviceTm':'timestamp', 'Value':'glucose mmol/l', 'AgeAsOfEnrollDt':'age',
                                'Weight':'weight', 'Height':'height', 'HbA1c':'hb_a1c', 'Gender': 'sex'}, inplace=True)
        df_final = df_final[['id','timestamp','timestamp_seconds', 'time_bin', 'glucose mmol/l', 'scaled_glucose',
                             'age', 'weight', 'height', 'sex', 'hb_a1c']]
        # df_final[['cpep_0_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']] = 0
        df_final['id'] = 'JAEB_Healthy_' + df_final['id'].astype(str)
        self.get_gmi(df_final)
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        '''Data info'''
        no_dup_df = df_final.copy()
        no_dup_df = no_dup_df.drop_duplicates('id')
        num_people = no_dup_df.shape[0]

        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]

        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

        c_peptide_interval = 'N/A'
        beta2_interval = {'N/A'}
        cgm_interval = '5'
        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}

        return df_final, pd.DataFrame(df_summary)

    def preprocess_jaeb_t1d(self, csv_folder: str, save_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the JAEB T1D dataset: loads CGM with Dask, converts/aligns timestamps,
        merges labs/demographics by study day, computes C‑peptide AUC/BETA2, and returns final table + summary.

        Args:
            csv_folder: Path to the dataset root containing the `original/` source files.

        Returns:
            A tuple of:
                - pd.DataFrame: Final processed dataset (`df_final`) with merged features.
                - pd.DataFrame: One‑row summary statistics table.
        """

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/2025-03-13_T1D_JAEB_C-path_CGM_clean.csv')
        df_cgm = df_cgm.rename(columns={'PtID': 'id', 'Value': 'glucose mmol/l', 'DeviceTm': 'timestamp',
                                        'DeviceDtDaysFromEnroll':'dy'})
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm = self.get_taylor_time_bins(df=df_cgm, offset=0, dy_available=True)
        df_cgm.loc[(df_cgm['dy'] >30)&(df_cgm['dy'] < 50) , 'time_bin'] = 'Week 6'
        df_cgm_clean = df_cgm.dropna(subset=['timestamp', 'glucose mmol/l'])
        df_cgm_clean = df_cgm_clean.sort_values(by=['id', 'timestamp'])
        df_cgm_clean['Hour'] = df_cgm_clean['timestamp'].dt.hour
        df_cgm_clean['timestamp_type'] = df_cgm_clean['Hour'].apply(lambda x: 'Nocturnal' if 0 <= x <= 5 else 'Daytime')
        df_cgm_clean = df_cgm_clean[['id', 'timestamp', 'timestamp_type', 'time_bin', 'dy', 'glucose mmol/l']]

        #Extra features
        df_extra_features = pd.read_csv(f'{csv_folder}/original/extra_features.csv')
        df_extra_features.rename(columns={'PtID':'id', 'plcb': 'treatment_arm', 'time_bins': 'time_bin', 'DaysFromEnroll': 'dy'} , inplace=True)
        df_extra_features['id'] = df_extra_features['id'].astype(str)
        df_extra_features.loc[df_extra_features['dy'] < 30 , 'time_bin'] = 'Baseline'
        df_extra_features.loc[(df_extra_features['dy'] >30)&(df_extra_features['dy'] < 50) , 'time_bin'] = 'Week 6'

        df_demographic = df_extra_features[['id', 'sex', 'race', 'treatment_arm']]
        df_demographic.drop_duplicates(subset=['id'], inplace=True)
        df_demographic['sex'] = df_demographic['sex'].map({'Male': 0, 'Female': 1})

        df_height_weight = df_extra_features[['id', 'dy', 'time_bin', 'age', 'weight', 'height']]
        df_height_weight = (
            df_height_weight.groupby(["id", "dy", "time_bin"], as_index=False)[['weight', 'height','age']]
            .median()
        )
        df_height_weight_clean = df_height_weight.dropna(subset=['weight'])

        df_cpep = df_extra_features[['id', 'dy', 'time_bin', 'cpep_fast', 'cpepm10', 'cpep0',
                                     'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
        df_cpep.rename(columns={'cpepm10': 'cpep_pre10_min','cpep0': 'cpep_0_min','cpep15': 'cpep_15_min',
                                'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min','cpep90': 'cpep_90_min',
                                 'cpep120': 'cpep_120_min'}, inplace=True)
        cpep_cols = [col for col in df_cpep.columns if col.startswith("cpep_")]
        for col in cpep_cols:
            df_cpep[col] = df_cpep[col]*0.331
        df_cpep_clean = df_cpep.dropna(subset=['cpep_0_min', 'cpep_15_min',
                            'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min'])
        df_cpep['cpep_auc'] = df_cpep_clean.apply(self.get_cpep_auc, axis = 1)

        df_glucose = df_extra_features[['id', 'dy', 'time_bin', 'glu0',
                                        'glu15', 'glu30', 'glu60', 'glu90', 'glu120']]
        df_glucose.rename(columns={'glu0': 'glucose_0_min','glu15': 'glucose_15_min',
                                'glu30': 'glucose_30_min', 'glu60': 'glucose_60_min',
                                'glu90': 'glucose_90_min', 'glu120': 'glucose_120_min'}, inplace=True)
        glucose_cols = [col for col in df_glucose.columns if col.startswith("glucose_")]
        for col in glucose_cols:
            df_glucose[col] = pd.to_numeric(df_glucose[col], errors='coerce')
            df_glucose[col] = df_glucose[col]/18
        df_glucose_clean = df_glucose.dropna(subset=['glucose_0_min', 'glucose_15_min', 'glucose_30_min', 'glucose_60_min',
                                  'glucose_90_min', 'glucose_120_min'])

        df_insulin = df_extra_features[['id', 'dy', 'time_bin', 'insulin']]
        df_insulin.rename(columns={'insulin' :'total_ins_dose'}, inplace=True)
        df_insulin = (
            df_insulin.groupby(["id", "dy", "time_bin"], as_index=False)['total_ins_dose']
            .median()
        )
        df_insulin_clean = df_insulin.dropna(subset=['total_ins_dose'])

        df_hba1c = df_extra_features[['id', 'dy', 'time_bin', 'hba1c']]
        df_hba1c.rename(columns={'hba1c': 'hb_a1c'}, inplace=True)
        df_hba1c_clean = df_hba1c.dropna(subset=['hb_a1c'])
        df_hba1c_clean = (
            df_hba1c_clean.groupby(["id", "dy", "time_bin"], as_index=False)['hb_a1c']
            .median()
        )

        df_cgm_windows = (
            df_cgm_clean.groupby(['id', 'time_bin'], as_index=False)['dy']
            .min()
            .rename(columns={'dy': 'cgm_window_dy'})
            .sort_values(by=['cgm_window_dy', 'id'])
            .reset_index(drop=True)
        )

        common_ids = (set(df_cpep['id']) & set(df_hba1c_clean['id']) & set(df_cgm_windows['id']) & set(df_glucose['id'])
                       & set(df_height_weight['id']) & set(df_insulin['id']))

        df_window_features = df_cgm_windows[df_cgm_windows['id'].isin(common_ids)].copy()
        df_window_features.sort_values(by=['cgm_window_dy', 'id'], inplace=True)

        df_cpep_merge = df_cpep[df_cpep['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_glucose_merge = df_glucose[df_glucose['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_hba1c_merge = df_hba1c_clean[df_hba1c_clean['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_insulin_merge = df_insulin[df_insulin['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)
        df_height_weight_merge = df_height_weight[df_height_weight['id'].isin(common_ids)].sort_values(by=['dy', 'id']).reset_index(drop=True)

        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_cpep_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_hba1c_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_height_weight_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_insulin_merge)
        df_window_features = self.merge_nearest_nonempty_by_dy(df_window_features, df_glucose_merge)

        df_window_features.rename(columns={'cgm_window_dy': 'dy'}, inplace=True)

        df_final = df_cgm_clean[df_cgm_clean['id'].isin(common_ids)].copy()
        print(df_final.shape)
        df_final = df_final.merge(df_demographic, on='id', how='left')
        print(df_final.shape)
        df_final = df_final.merge(df_window_features, on=['id', 'time_bin', 'dy'], how='left')
        print(df_final.shape)

        df_final = self.fill_static_within_time_bin(df_final, extra_exclude={'dy'})
        df_final.loc[(df_final['dy'] >30)&(df_final['dy'] < 50) , 'time_bin'] = 'Month 3'
        df_diagnose = df_extra_features[['id', 'dy', 'diag_dy']]
        df_final = df_final.merge(df_diagnose, left_on=['id','dy'], right_on=['id', 'dy'], how='left')
        df_final['diag_dy'] = pd.to_numeric(df_final['diag_dy'], errors='coerce')
        df_final['diagnose_date'] = df_final['timestamp'] + pd.to_timedelta(df_final['diag_dy'], unit='D') + pd.to_timedelta(df_final['dy'], unit='D')
        df_final['diagnose_date'] = (df_final.groupby('id', group_keys=False)['diagnose_date'].transform(lambda s: s.dropna().min())).dt.date
        df_final.drop(columns=['diag_dy'], inplace=True)
        df_final['cpep_pre10_min'] = df_final['cpep_pre10_min'].fillna(df_final['cpep_0_min'])
        df_final['total_ins_dose'] = df_final['total_ins_dose'] * df_final['weight']
        self.get_beta_2_scores(df_final)
        self.get_gmi(df_final)
        self.get_beta_3_score(df_final)

        '''Data info'''
        no_dup_df = df_final.drop_duplicates(subset='id')
        num_people = no_dup_df.shape[0]
        num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
        num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]
        # num_races = db_screening['Race'].value_counts()
        min_age = no_dup_df['age'].min()
        max_age = no_dup_df['age'].max()
        num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
        num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]
        c_peptide_interval = '0,15,30:120:30 '
        beta2_interval = {'N/A'}
        cgm_interval = '5 mins'

        df_summary = {'Number of patients': [num_people],
                    'Number of males':[num_male],
                    'Number of females': [num_female],
                    'Minimum age': [min_age],
                    'Maximum age': [max_age],
                    'Number of people < 18 years': [num_less_than_18],
                    'Number of people >= 18 years': [num_18_or_older],
                    'C-Peptide interval': [c_peptide_interval],
                    'BETA2 Score interval': [beta2_interval],
                    'CGM data interval': [cgm_interval]}     
        self.print_consort_general(df_list_raw=[df_cgm, df_hba1c, df_insulin, df_height_weight, df_cpep, df_glucose],
                           df_list_clean=[df_cgm_clean, df_hba1c_clean, df_insulin_clean, df_height_weight_clean, df_cpep_clean, df_glucose_clean],
                           df_final=df_final)                 
        
        time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12', 'Month 15', 'Month 18', 'Month 21', 'Month 24']
        self.print_consort_time_bins(time_bins=time_bins, df_list_raw=[df_cgm, df_cpep, df_height_weight, df_glucose, df_hba1c, df_insulin],
                           df_list_clean=[df_cgm_clean, df_cpep_clean, df_height_weight_clean, df_glucose_clean, df_hba1c_clean, df_insulin_clean],
                           df_final=df_final,
                           df_names=['df_cgm_clean', 'df_cpep_clean', 'df_height_weight_clean', 'df_glucose_clean', 'df_hba1c_clean', 'df_insulin_clean'])

        if save_csv:
            df_cgm.to_csv('./data/studies/jaeb_t1d/csv_files/cgm/df_original.csv', index=False)
            df_cgm_clean.to_csv('./data/studies/jaeb_t1d/csv_files/cgm/df_clean.csv', index=False)
            df_hba1c.to_csv('./data/studies/jaeb_t1d/csv_files/hb_a1c/df_original.csv', index=False)
            df_hba1c_clean.to_csv('./data/studies/jaeb_t1d/csv_files/hb_a1c/df_clean.csv', index=False)
            df_cpep.to_csv('./data/studies/jaeb_t1d/csv_files/cpep/df_original.csv', index=False)
            df_cpep_clean.to_csv('./data/studies/jaeb_t1d/csv_files/cpep/df_clean.csv', index=False)
            df_insulin.to_csv('./data/studies/jaeb_t1d/csv_files/insulin/df_original.csv', index=False)
            df_insulin_clean.to_csv('./data/studies/jaeb_t1d/csv_files/insulin/df_clean.csv', index=False)
            df_glucose.to_csv('./data/studies/jaeb_t1d/csv_files/glucose/df_original.csv', index=False)
            df_glucose_clean.to_csv('./data/studies/jaeb_t1d/csv_files/glucose/df_clean.csv', index=False)
            df_height_weight.to_csv('./data/studies/jaeb_t1d/csv_files/height_weight/df_original.csv', index=False)
            df_height_weight_clean.to_csv('./data/studies/jaeb_t1d/csv_files/height_weight/df_clean.csv', index=False)
            df_final.to_csv('./data/studies/jaeb_t1d/df_final.csv', index=False)

        return df_final, pd.DataFrame(df_summary)

    """ Get functions """
    def get_beta_2_scores(self, df: pd.DataFrame) -> None:
        """
        Computes BETA2 score for each record using fasting C-peptide, glucose,
        insulin dose, body weight, and HbA1c. Adds a new column `beta2_score` in place.

        Args:
            df: Input dataframe containing required numeric and lab columns.
                 (weight, height, C-peptide timepoints, insulin, glucose, HbA1c)

        Returns:
            None. Modifies the input dataframe in place by adding `beta2_score`.
        """

        numeric_columns = [
            'weight', 'height', 'cpep_0_min', 'cpep_30_min', 'cpep_60_min',
            'cpep_90_min', 'cpep_120_min', 'hb_a1c', 'total_ins_dose',
            'glucose_0_min', 'glucose_30_min', 'glucose_60_min',
            'glucose_90_min', 'glucose_120_min'
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['cpep_fast'] = df['cpep_0_min']
        df['glucose_fast'] = df['glucose_0_min']
        df['beta2_score'] = (np.sqrt(df['cpep_fast']) * (1 - (df['total_ins_dose']/df['weight'])))/(df['glucose_fast'] * df['hb_a1c']) * 1000

    def get_beta_3_score(self, df: pd.DataFrame) -> None:
        """
        Computes BETA3 score for each record using fasting C-peptide, glucose,
        insulin dose, body weight, and GMI. Adds a new column `beta3_score` in place.

        Args:
            df: Input dataframe containing required numeric and lab columns.
                 (weight, height, C-peptide timepoints, insulin, glucose, GMI)

        Returns:
            None. Modifies the input dataframe in place by adding `beta3_score`.
        """

        numeric_columns = [
            'weight', 'height', 'cpep_0_min', 'cpep_30_min', 'cpep_60_min',
            'cpep_90_min', 'cpep_120_min', 'gmi', 'total_ins_dose',
            'glucose_0_min', 'glucose_30_min', 'glucose_60_min',
            'glucose_90_min', 'glucose_120_min'
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['cpep_fast'] = df['cpep_0_min']
        df['glucose_fast'] = df['glucose_0_min']
        df['beta3_score'] = (np.sqrt(df['cpep_fast']) * (1 - (df['total_ins_dose']/df['weight'])))/(df['glucose_fast'] * df['gmi']) * 1000

    def get_cgm_core_endpoints_general(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Aggregate core CGM endpoints across all patients.

        Groups the input by 'id' and applies `get_cgm_core_endpoints_per_patient` to each
        patient's time series. Non-empty per-patient results are collected and returned
        as a DataFrame.

        Args:
            df (pd.DataFrame): Full dataset containing at least 'id', 'timestamp',
                and 'glucose mmol/l' columns.
            **kwargs (dict): Additional keyword arguments (forwarded; currently unused).

        Returns:
            pd.DataFrame: One row per patient ID with CGM endpoints as columns.
        """
        df_cgm = df.copy()
        df_cgm = df_cgm[['id', 'timestamp', 'glucose mmol/l']]
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])

        core_endpoints = []
        for pt_id, pt_df in df_cgm.groupby(['id']):
            core_endpoint = self.get_cgm_core_endpoints_per_patient(pt_df)
            if core_endpoint:  # ensure it's not empty
                core_endpoint['id'] = pt_id[0]
                core_endpoints.append(core_endpoint)

        return pd.DataFrame(core_endpoints)

    def get_cgm_core_endpoints_per_patient(self, df_group: pd.DataFrame, **kwargs) -> dict[str, float]:
        """
        Compute core CGM (Continuous Glucose Monitoring) endpoints for a single patient.

        Derives glycemic metrics including time in (tight) range, time above/below range,
        wear percentage, glucose variability statistics, and the Length of Line (LoL) metric.

        Args:
            df_group (pd.DataFrame): Subset for one patient/group containing at least
                'timestamp' and 'glucose mmol/l' columns.
            **kwargs (dict): Additional keyword arguments (currently unused; reserved for future extension).

        Returns:
            dict[str, float]: Dictionary of CGM endpoints with keys:
                - percent_wear_time: Percentage of expected wear time covered by valid data.
                - TIR: Time in range 3.9–10.0 mmol/L (% of total time).
                - TITR: Time in tight range 3.9–7.8 mmol/L (% of total time).
                - TBR_Lvl_1: Time below range <3.9 and ≥3.0 mmol/L (% of total time).
                - TBR_Lvl_2: Time below range <3.0 mmol/L (% of total time).
                - TAR_Lvl_1: Time above range >10.0 and ≤13.9 mmol/L (% of total time).
                - TAR_Lvl_2: Time above range >13.9 mmol/L (% of total time).
                - min_glucose: Minimum observed glucose (mmol/L).
                - max_glucose: Maximum observed glucose (mmol/L).
                - std_glucose: Standard deviation of glucose (mmol/L).
                - cv_percent: Coefficient of variation of glucose (%).
                - GVP: Glycemic Variability Percentage

        Notes:
            - Timestamps are converted to pandas datetime.
            - Per‑row time steps are computed and rows with gaps > 0.3 hours (18 minutes) are dropped.
            - If total time or mean glucose is zero, an empty dict is returned.
        """
        df_group = df_group.copy()
        df_group['timestamp'] = pd.to_datetime(df_group['timestamp'])

        df_group['time_diff'] = df_group['timestamp'].diff().dt.total_seconds() / 3600
        median_interval = df_group['time_diff'].median()
        df_group['time_diff'] = df_group['time_diff'].fillna(median_interval)
        df_group = df_group[df_group['time_diff'] <= 0.3]

        total_time = df_group['time_diff'].sum()
        mean_glucose = df_group['glucose mmol/l'].mean()

        if total_time == 0 or mean_glucose == 0:
            return {}
        
        median_glucose = df_group['glucose mmol/l'].median()
        q1_sensor = df_group['glucose mmol/l'].quantile(0.25)
        q3_sensor = df_group['glucose mmol/l'].quantile(0.75)
        min_glucose = df_group['glucose mmol/l'].min()
        max_glucose = df_group['glucose mmol/l'].max()
        std_glucose = df_group['glucose mmol/l'].std()
        cv = std_glucose / mean_glucose * 100
        gvp_min, gvp_hr, LoL_min, LoL_hr = self.length_data(df_group)

        in_range = (df_group['glucose mmol/l'] >= 3.9) & (df_group['glucose mmol/l'] <= 10.0)
        in_tight_range = (df_group['glucose mmol/l'] >= 3.9) & (df_group['glucose mmol/l'] <= 7.8)
        below_3_9 = (df_group['glucose mmol/l'] < 3.9) & (df_group['glucose mmol/l'] >= 3.0)
        below_3_0 = df_group['glucose mmol/l'] < 3.0
        above_10 = (df_group['glucose mmol/l'] > 10.0) & (df_group['glucose mmol/l'] <= 13.9)
        above_13_9 = df_group['glucose mmol/l'] > 13.9

        tir = df_group.loc[in_range, 'time_diff'].sum()
        titr = df_group.loc[in_tight_range, 'time_diff'].sum()
        tbr_1 = df_group.loc[below_3_9, 'time_diff'].sum()
        tbr_2 = df_group.loc[below_3_0, 'time_diff'].sum()
        tar_1 = df_group.loc[above_10, 'time_diff'].sum()
        tar_2 = df_group.loc[above_13_9, 'time_diff'].sum()

        max_wear = (df_group['timestamp'].iloc[-1] - df_group['timestamp'].iloc[0]).total_seconds() / 3600
        wear_pct = (total_time / max_wear) * 100 if max_wear > 0 else 0

        return {
            'percent_wear_time': wear_pct,
            'TIR': (tir / total_time) * 100,
            'TITR': (titr / total_time) * 100,
            'TBR_Lvl_1': (tbr_1 / total_time) * 100,
            'TBR_Lvl_2': (tbr_2 / total_time) * 100,
            'TAR_Lvl_1': (tar_1 / total_time) * 100,
            'TAR_Lvl_2': (tar_2 / total_time) * 100,
            'mean_glucose': mean_glucose,
            'median_glucose': median_glucose,
            'min_glucose': min_glucose,
            'max_glucose': max_glucose,
            'std_glucose': std_glucose,
            'cv_percent': cv,
            'GVP': gvp_min
        }

    def get_cgm_windows(self, df: pd.DataFrame, wear_days: int, wear_prct: int) -> pd.DataFrame:
        """
        Return rows within the first qualifying CGM window per patient and time_bin.

        A qualifying window is `wear_days` consecutive calendar days with at least
        `wear_prct` percent expected wear.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        df = df.dropna(subset=['id', 'time_bin', 'timestamp'])
        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['id', 'time_bin', 'timestamp'])

        def expected_entries(days: int, interval_minutes: float) -> float:
            if pd.isna(interval_minutes) or interval_minutes <= 0:
                return 0
            return (days * 24 * 60) / interval_minutes

        def find_first_window(day_counts: pd.Series, window_days: int, wear_pct: float) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
            if day_counts.empty:
                return None, None

            day_counts = day_counts.sort_index()
            dates = day_counts.index.to_list()
            counts = day_counts.values
            start_idx = 0

            while start_idx < len(dates):
                end_idx = start_idx + 1
                while end_idx < len(dates) and (dates[end_idx] - dates[end_idx - 1]).days == 1:
                    end_idx += 1

                run_len = end_idx - start_idx
                if run_len >= window_days:
                    for offset in range(0, run_len - window_days + 1):
                        window_start = dates[start_idx + offset]
                        window_end = dates[start_idx + offset + window_days - 1]
                        window_sum = int(sum(counts[start_idx + offset:start_idx + offset + window_days]))
                        expected = expected_entries(window_days, interval_minutes)
                        wear = (window_sum / expected) * 100 if expected else 0
                        if wear >= wear_pct:
                            return window_start, window_end

                start_idx = end_idx

            return None, None

        selected_indices = []
        for (patient_id, time_bin), group in df.groupby(['id', 'time_bin']):
            group = group.sort_values('timestamp')
            median_interval = group['timestamp'].diff().dt.total_seconds().dropna().median()
            interval_minutes = median_interval / 60 if not pd.isna(median_interval) else 0
            interval_minutes = 5 if interval_minutes <= 7.5 else 15

            day_counts = group.groupby(group['timestamp'].dt.floor('D')).size()
            window_start, window_end = find_first_window(day_counts, wear_days, wear_prct)
            if window_start is None:
                continue

            mask = (group['timestamp'].dt.floor('D') >= window_start) & (
                group['timestamp'].dt.floor('D') <= window_end
            )
            selected_indices.extend(group.loc[mask].index.tolist())

        return df.loc[selected_indices].sort_values(['id', 'time_bin', 'timestamp'])

    def get_cpep_auc(self, row: pd.Series) -> float:
        """
        Calculate the Area Under the Curve (AUC) of MMTT C‑peptide values.

        Uses trapezoidal integration over fixed time points [0, 30, 60, 90, 120] minutes.

        Args:
            row (pd.Series): Row containing C‑peptide columns
                (`cpep_0_min`, `cpep_30_min`, `cpep_60_min`, `cpep_90_min`, `cpep_120_min`).

        Returns:
            float: Calculated C‑peptide AUC.
        """
        times = [0,30,60,90,120]
        cpep_values = [row[f'cpep_{t}_min'] for t in times]        
        auc = 0
        for i in range(1, len(times)):
            # width = difference in time
            delta_t = times[i] - times[i-1]
            # height = average of consecutive cpep values
            auc += (cpep_values[i] + cpep_values[i-1]) / 2 * delta_t

        duration = times[-1] - times[0]
        return auc / duration

    def get_dataset(self, name: str) -> pd.DataFrame | None:
        """
        Retrieves a processed dataset by name.

        Args:
            name: Dataset key (e.g., study identifier) stored in self.datasets.

        Returns:
            pd.DataFrame | None: The dataset with a `study` column if present;
            otherwise raises an error.
        """
        dataset = self.datasets.get(name, None)
        if dataset is None:
            raise KeyError(f"Dataset '{name}' is not loaded.")

        if hasattr(dataset, 'compute'):
            dataset = dataset.compute()

        return dataset.assign(study=name)
    
    def get_dataset_summary(self, name: str) -> pd.DataFrame | None:
        """
        Retrieves a dataset summary by name.

        Args:
            name: Dataset key stored in self.datasets_summary.

        Returns:
            pd.DataFrame | None: The summary table if present; otherwise None.
        """

        return self.datasets_summary.get(name, None)
    
    def get_gmi(self, df: pd.DataFrame) -> None:
        """
        Adds Glucose Management Indicator (GMI) per participant and time bin.

        GMI is estimated HbA1c derived from CGM mean glucose. The coefficient
        `0.43056` corresponds to the standard mg/dL formula (0.02392) scaled by
        the 18 mg/dL → mmol/L conversion factor.
        """
        df['gmi'] = (
            df.groupby(by=['id', 'time_bin'])['glucose mmol/l']
              # 3.31 + 0.43056 * mean glucose (mmol/L) => estimated HbA1c in %
              .transform(lambda g: 3.31 + 0.43056 * g.mean())
        )

    def get_hypoglycemia_rates(self, df: pd.DataFrame) -> tuple[int, int, int]:
        """
        Calculates hypoglycemia rates from CGM data, including:
        - Level 1 hypoglycemia (3.0 ≤ glucose < 3.9 mmol/L),
        - Level 2 hypoglycemia (glucose < 3.0 mmol/L),
        - Clinically significant hypoglycemia events (continuous glucose < 3.0 mmol/L, 
          detected with 15-minute event windowing).

        Args:
            df: DataFrame with CGM records, must include 'timestamp' and 'glucose mmol/l' columns.

        Returns:
            tuple:
                level_1 (int): Number of Level 1 hypoglycemia readings.
                level_2 (int): Number of Level 2 hypoglycemia readings.
                num_events (int): Number of clinically significant hypoglycemia events.
        """

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_hypo_level_1 = df[(df['glucose mmol/l'] < 3.9) & (df['glucose mmol/l'] >= 3.0)]
        df_hypo_level_2 = df[df['glucose mmol/l'] < 3.0]

        level_1 = df_hypo_level_1.shape[0]
        level_2 = df_hypo_level_2.shape[0]
    
        median_time_interval = (df['timestamp'].diff().median()).seconds / 60
        if median_time_interval < 6:
            df_clinically_significant = df.iloc[::3].reset_index(drop=True)
        else:
            df_clinically_significant = df
        
        event_start = False
        num_events = 0
        for i in range(df_clinically_significant.shape[0]-1):
            current_glucose = df_clinically_significant.iloc[i]['glucose mmol/l']
            next_glucose = df_clinically_significant.iloc[i+1]['glucose mmol/l']
            current_time = df_clinically_significant.iloc[i]['timestamp']
            next_time = df_clinically_significant.iloc[i+1]['timestamp']
            time_diff = (next_time - current_time).total_seconds()/60

            if time_diff < 16:
                if current_glucose < 3.0:
                    if next_glucose < 3.0:
                        event_start = True

                if event_start:
                    if current_glucose > 3.9:
                        if next_glucose > 3.9:
                            event_start = False
                            num_events += 1

            else:
                if event_start:
                    event_start = False
                    if next_glucose > 3.9:
                        num_events += 1

        return level_1, level_2, num_events

    def get_taylor_time_bins(self, df: pd.DataFrame, offset: int, dy_available: bool) -> pd.DataFrame:
        """
        Assigns each patient’s CGM records into predefined Taylor-style time bins 
        (Baseline, Month 3, Month 6, etc.).

        Steps:
            - Converts 'timestamp' to datetime.
            - Computes days since baseline per patient (per 'id').
            - Filters to < 750 days.
            - Assigns time bins based on ranges of days since baseline.
            - Converts 'time_bin' into an ordered categorical variable.

        Args:
            df (pd.DataFrame): Dataset containing at least 'id' and 'timestamp'.

        Returns:
            pd.DataFrame: Input DataFrame with an added categorical column 'time_bin'.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if dy_available:
            df['days_since'] = df['dy']
        else:
            df['days_since'] = df.groupby('id')['timestamp'].transform(lambda x: (x - x.min()).dt.days)
        df = df[df['days_since'] < 750 - offset]

        df.loc[df['days_since'] < 30 - offset, 'time_bin'] = 'Baseline'
        df.loc[(df['days_since'] >= 30 - offset) & (df['days_since'] < 120 - offset), 'time_bin'] = 'Month 3'
        df.loc[(df['days_since'] >= 120 - offset) & (df['days_since'] < 210 - offset), 'time_bin'] = 'Month 6'
        df.loc[(df['days_since'] >= 210 - offset) & (df['days_since'] < 300 - offset), 'time_bin'] = 'Month 9'
        df.loc[(df['days_since'] >= 300 - offset) & (df['days_since'] < 390 - offset), 'time_bin'] = 'Month 12'
        df.loc[(df['days_since'] >= 390 - offset) & (df['days_since'] < 480 - offset), 'time_bin'] = 'Month 15'
        df.loc[(df['days_since'] >= 480 - offset) & (df['days_since'] < 570 - offset), 'time_bin'] = 'Month 18'
        df.loc[(df['days_since'] >= 570 - offset) & (df['days_since'] < 660 - offset), 'time_bin'] = 'Month 21'
        df.loc[(df['days_since'] >= 660 - offset) & (df['days_since'] < 750 - offset), 'time_bin'] = 'Month 24'
        df.drop(columns='days_since', inplace=True)

        return df

    def list_datasets(self) -> list[str]:
        """
        List all datasets currently loaded on the instance.

        Returns:
            list[str]: Keys or names of available datasets.
        """
        return list(self.datasets.keys())

    """ Plot functions """
    def combine_pngs(
        self,
        directory: str,
        output: str = "combined.png",
        cols: int = 3,
        padding: int = 10,
        bg: tuple[int, int, int] = (255, 255, 255),
        ordered_labels: list[str] | None = None,
        include_filter: str | None = None
    ) -> Path:
        """
        Combine all PNGs in a directory into a single grid image.

        Args:
            directory: Folder containing .png files.
            output: Path for the combined PNG.
            cols: Number of columns in the grid.
            padding: Padding between images (pixels).
            bg: RGB tuple for background color.
            ordered_labels: Optional list of substrings that define ordering in filenames
                (e.g., ['Baseline','Month 3',...]).
            include_filter: Optional substring; only PNGs containing this will be included.

        Returns:
            pathlib.Path: Location of the saved combined image.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"{dir_path} does not exist")

        out_path = Path(output)
        pngs = [
            p for p in dir_path.glob("*.png")
            if p.is_file()
            and p.resolve() != out_path.resolve()
            and (include_filter is None or include_filter in p.name)
        ]
        if not pngs:
            raise ValueError(f"No PNG files found in {dir_path}")

        def sort_key(path: Path):
            name = path.name
            if ordered_labels:
                for idx, label in enumerate(ordered_labels):
                    if label in name:
                        return (0, idx, name)
            return (1, name)

        pngs = sorted(pngs, key=sort_key)

        images = [Image.open(p).convert("RGB") for p in pngs]
        max_w = max(im.width for im in images)
        max_h = max(im.height for im in images)
        resized = [im.resize((max_w, max_h), Image.LANCZOS) if (im.width, im.height) != (max_w, max_h) else im for im in images]

        rows = math.ceil(len(resized) / cols)
        canvas_w = cols * max_w + (cols + 1) * padding
        canvas_h = rows * max_h + (rows + 1) * padding
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)

        for idx, im in enumerate(resized):
            r, c = divmod(idx, cols)
            x = padding + c * (max_w + padding)
            y = padding + r * (max_h + padding)
            canvas.paste(im, (x, y))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, format="PNG")
        return out_path

    def plot_AGP(
        self,
        df_group: pd.DataFrame,
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        path: str
    ) -> None:
        """
        Plot an Ambulatory Glucose Profile-style chart comparing patient data to healthy and established references.

        Args:
            df_group: Dataset with patient CGM data (must include timestamp and glucose).
            healthy_reference: Reference CGM dataset for healthy cohort.
            established_reference: Reference CGM dataset for established cohort.
            path: Output file path for the saved PNG.

        Returns:
            None
        """
        df_group = df_group.copy()
        df_group['treatment_arm'] = df_group['treatment_arm'].str.lower()
        df_control = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_1)]
        df_active = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_2)]
        healthy_reference = healthy_reference.copy()
        established_reference = established_reference.copy()

        group_endpoints = self.get_cgm_core_endpoints_general(df_group)
        control_endpoints = self.get_cgm_core_endpoints_general(df_control)
        active_endpoints = self.get_cgm_core_endpoints_general(df_active)
        healthy_endpoints = self.get_cgm_core_endpoints_general(healthy_reference)
        established_endpoints = self.get_cgm_core_endpoints_general(established_reference)

        def _attach_hba1c(endpoints_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
            """Attach per-patient HbA1c medians to endpoint tables for downstream stats."""
            endpoints_with_hba1c = endpoints_df.copy()
            if endpoints_with_hba1c.empty or 'hb_a1c' not in source_df.columns:
                endpoints_with_hba1c['hb_a1c'] = np.nan
                return endpoints_with_hba1c

            hba1c_by_id = (
                source_df[['id', 'hb_a1c']]
                .dropna(subset=['hb_a1c'])
                .groupby('id')['hb_a1c']
                .median()
            )
            endpoints_with_hba1c['hb_a1c'] = endpoints_with_hba1c['id'].map(hba1c_by_id)
            return endpoints_with_hba1c

        group_endpoints = _attach_hba1c(group_endpoints, df_group)
        control_endpoints = _attach_hba1c(control_endpoints, df_control)
        active_endpoints = _attach_hba1c(active_endpoints, df_active)
        healthy_endpoints = _attach_hba1c(healthy_endpoints, healthy_reference)
        established_endpoints = _attach_hba1c(established_endpoints, established_reference)

        df_group['tod'] = df_group['timestamp'].dt.floor('15min').dt.time
        df_control['tod'] = df_control['timestamp'].dt.floor('15min').dt.time
        df_active['tod'] = df_active['timestamp'].dt.floor('15min').dt.time
        healthy_reference['tod'] = healthy_reference['timestamp'].dt.floor('15min').dt.time
        established_reference['tod'] = established_reference['timestamp'].dt.floor('15min').dt.time

        patient_results = df_group.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        patient_results = patient_results.rename(columns={'glucose mmol/l': 'median'})
        control_results = df_control.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        control_results = control_results.rename(columns={'glucose mmol/l': 'median'})
        active_results = df_active.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        active_results = active_results.rename(columns={'glucose mmol/l': 'median'})
        pr_h = healthy_reference.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_h = pr_h.rename(columns={'glucose mmol/l': 'median'})
        pr_e = established_reference.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_e = pr_e.rename(columns={'glucose mmol/l': 'median'})

        group_agp = patient_results.groupby('tod')['median'].quantile(q=[0.10,0.25,0.5,0.75,0.90]).unstack()
        group_agp.columns = ['10th Centile','q1','median','q3','90th Centile']

        # --- Median summary lines for the time series plot ---
        summary_info = []
        summary_sources = [
            ('Group overall', group_endpoints, '#f57c00', 'dashed'),
            ('Group control', control_endpoints, "#0019f5", 'dashed'),
            ('Group active', active_endpoints, "#e5f500", 'dashed'),
            ('Healthy overall', healthy_endpoints, '#00f514', 'dashdot'),
            ('Established overall', established_endpoints, '#f50000', (0, (3, 2)))
        ]
        for label, endpoints_df, color, linestyle in summary_sources:
            median_value = endpoints_df['median_glucose'].median()
            if pd.notna(median_value):
                summary_info.append({
                    'label': label,
                    'value': median_value,
                    'color': color,
                    'linestyle': linestyle
                })

        ref_he = pr_h.groupby('tod')['median'].median().to_frame('value')
        ref_e = pr_e.groupby('tod')['median'].median().to_frame('value')
        ref_c = control_results.groupby('tod')['median'].median().to_frame('value')
        ref_a = active_results.groupby('tod')['median'].median().to_frame('value')

        xh = np.array([t.hour + t.minute/60 + t.second/3600 for t in group_agp.index], dtype=float)

        fig = plt.figure(figsize=(7.5, 7.2))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.8, 4], hspace=0.4)
        ax_table = fig.add_subplot(gs[0])
        ax_plot = fig.add_subplot(gs[1])

        metrics_to_display = [
            ('hb_a1c', 'Hb A1C (%)'),
            ('TIR', 'TIR (%)'),
            ('TITR', 'TITR (%)'),
            ('TBR_Lvl_1', 'TBR Lvl 1 (%)'),
            ('TBR_Lvl_2', 'TBR Lvl 2 (%)'),
            ('TAR_Lvl_1', 'TAR Lvl 1 (%)'),
            ('TAR_Lvl_2', 'TAR Lvl 2 (%)'),
            ('min_glucose', 'Min (mmol/L)'),
            ('max_glucose', 'Max (mmol/L)'),
            ('std_glucose', 'Std (mmol/L)'),
            ('cv_percent', 'CV (%)'),
            ('GVP', 'GVP')
        ]

        def _format_p_value(p_value: float) -> str:
            if pd.isna(p_value):
                return 'n/a'
            if p_value < 0.001:
                return '<0.001**'
            if p_value < 0.05:
                return f'{p_value:.3f}*'
            return f'{p_value:.3f}'

        anova_results = {}
        for column, _ in metrics_to_display:
            samples = [
                group_endpoints[column].dropna(),
                control_endpoints[column].dropna(),
                active_endpoints[column].dropna(),
                established_endpoints[column].dropna()
            ]
            valid_samples = [s for s in samples if len(s) > 0]
            if len(valid_samples) < 2:
                anova_results[column] = np.nan
                continue
            try:
                sample_groups = {
                    f'group_{idx + 1}': sample
                    for idx, sample in enumerate(valid_samples)
                }
                _, anova_p = self.run_oneway_anova_test(sample_groups)
                anova_results[column] = anova_p
            except Exception:
                anova_results[column] = np.nan

        table_rows = []
        for column, label in metrics_to_display:
            values = [
                f'{group_endpoints[column].median():.1f}',
                f'{control_endpoints[column].median():.1f}',
                f'{active_endpoints[column].median():.1f}',
                f'{established_endpoints[column].median():.1f}',
                f'{healthy_endpoints[column].median():.1f}',
            ]
            p_value_fmt = _format_p_value(anova_results.get(column, np.nan))
            table_rows.append([label, *values, p_value_fmt])

        ax_table.axis('off')
        table = ax_table.table(
            cellText=table_rows,
            colLabels=['Metric', 'Group', 'Control', 'Active', 'Established', 'Healthy', 'ANOVA p'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.15)

        fig.text(0.5, 0.985, "Median CGM Endpoints", ha='center', va='top', fontsize=11, fontweight='bold')

        ax_plot.plot(xh, group_agp['median'], label='Median', color='#f57c00')
        ax_plot.plot(xh, ref_he['value'], label='Healthy Reference', color="#00f514")
        ax_plot.plot(xh, ref_e['value'], label='Established Reference', color="#f50000")
        ax_plot.plot(xh, ref_c['value'], label='Control Median', color="#0019f5")
        ax_plot.plot(xh, ref_a['value'], label='Active Median', color="#e5f500")
        ax_plot.fill_between(xh, group_agp['q1'], group_agp['q3'], color='#08306b', alpha=0.7, label='25-75th Centile')
        ax_plot.fill_between(xh, group_agp['10th Centile'], group_agp['90th Centile'], color='#6baed6', alpha=0.5, label='10-90th Centile')
        ax_plot.set_xlim(0, 24)
        ax_plot.set_xticks(
            np.arange(0, 25, 4),
            [f"{int(h):02d}:00" for h in np.arange(0, 25, 4)]
        )
        ax_plot.set_title("Group-level AGP", ha='center', fontsize=11, fontweight='bold')
        ax_plot.set_xlabel("Time of Day")
        ax_plot.set_ylabel('Glucose (mmol/l)')
        ax_plot.set_ylim((4,12))

        legend_handles, legend_labels = ax_plot.get_legend_handles_labels()
        # for entry in summary_info:
        #     line = ax_plot.axhline(
        #         entry['value'],
        #         color=entry['color'],
        #         linestyle=entry['linestyle'],
        #         linewidth=1.2
        #     )
        #     legend_handles.append(line)
        #     legend_labels.append(f"{entry['label']} (Median)")

        ax_plot.legend(
            legend_handles,
            legend_labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False
        )

        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        return None
    
    def plot_AGP_quartiles(
        self,
        df_group: pd.DataFrame,
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        quantile_feature: str,
        path: str
    ) -> None:
        df_group = df_group.copy()
        df_group['treatment_arm'] = df_group['treatment_arm'].str.lower()
        df_control = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_1)]
        df_active = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_2)]
        healthy_reference = healthy_reference.copy()
        established_reference = established_reference.copy()

        group_endpoints = self.get_cgm_core_endpoints_general(df_group)
        control_endpoints = self.get_cgm_core_endpoints_general(df_control)
        active_endpoints = self.get_cgm_core_endpoints_general(df_active)
        healthy_endpoints = self.get_cgm_core_endpoints_general(healthy_reference)
        established_endpoints = self.get_cgm_core_endpoints_general(established_reference)
        
        def _attach_hba1c(endpoints_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
            """Attach per-patient HbA1c medians to endpoint tables for downstream stats."""
            endpoints_with_hba1c = endpoints_df.copy()
            if endpoints_with_hba1c.empty or 'hb_a1c' not in source_df.columns:
                endpoints_with_hba1c['hb_a1c'] = np.nan
                return endpoints_with_hba1c

            hba1c_by_id = (
                source_df[['id', 'hb_a1c']]
                .dropna(subset=['hb_a1c'])
                .groupby('id')['hb_a1c']
                .median()
            )
            endpoints_with_hba1c['hb_a1c'] = endpoints_with_hba1c['id'].map(hba1c_by_id)
            return endpoints_with_hba1c

        group_endpoints = _attach_hba1c(group_endpoints, df_group)
        control_endpoints = _attach_hba1c(control_endpoints, df_control)
        active_endpoints = _attach_hba1c(active_endpoints, df_active)
        healthy_endpoints = _attach_hba1c(healthy_endpoints, healthy_reference)
        established_endpoints = _attach_hba1c(established_endpoints, established_reference)

        df_group['tod'] = df_group['timestamp'].dt.floor('15min').dt.time
        df_control['tod'] = df_control['timestamp'].dt.floor('15min').dt.time
        df_active['tod'] = df_active['timestamp'].dt.floor('15min').dt.time
        healthy_reference['tod'] = healthy_reference['timestamp'].dt.floor('15min').dt.time
        established_reference['tod'] = established_reference['timestamp'].dt.floor('15min').dt.time

        if quantile_feature in CGM_CORE_ENDPOINTS:
            quantile_df = group_endpoints[['id', quantile_feature]].dropna()
            quantile_control = control_endpoints[['id', quantile_feature]].dropna()
            quantile_active = active_endpoints[['id', quantile_feature]].dropna()
        else:
            quantile_df = (
                df_group[['id', quantile_feature]]
                .dropna(subset=[quantile_feature])
                .groupby('id', as_index=False)[quantile_feature]
                .median()
            )
            quantile_control = (
                df_control[['id', quantile_feature]]
                .dropna(subset=[quantile_feature])
                .groupby('id', as_index=False)[quantile_feature]
                .median()
            )
            quantile_active = (
                df_active[['id', quantile_feature]]
                .dropna(subset=[quantile_feature])
                .groupby('id', as_index=False)[quantile_feature]
                .median()
            )

        vals = quantile_df[quantile_feature]
        base_labels = ['Q1', 'Q2', 'Q3', 'Q4']

        if quantile_feature == 'cpep_auc':
            edges = [0, 0.25, 0.5, 0.75, 1]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'cpep_auc_preservation':
            edges = [0, 25, 50, 75, 100]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'beta2_score':
            # negatives = (
            #     df_group[df_group['beta2_score'] < 0][['id', 'time_bin', 'cpep_fast', 'cpep_0_min', 'glucose_fast', 'glucose_0_min',
            #                                            'total_ins_dose', 'weight','hb_a1c', 'beta2_score']]
            #     .dropna(subset=['id'])
            #     .drop_duplicates()
            # )
            # print("Beta2 Score < 0 — subject IDs and time_bin:")
            # print(negatives.to_string(index=False))
            edges = [0, 5, 10, 15]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        else:
            lo, hi = vals.quantile([0.05, 0.95])
            clipped_vals = vals.clip(lo, hi)
            quantile_probs = [0, 0.25, 0.5, 0.75, 1]
            bins = pd.qcut(
                clipped_vals,
                q=quantile_probs,
                labels=None,
                duplicates='drop'
            )

        intervals = list(bins.cat.categories)
        num_bins = len(intervals)
        bin_labels = base_labels[:num_bins]
        if len(bin_labels) < num_bins:
            bin_labels.extend([f'Q{i+1}' for i in range(len(bin_labels), num_bins)])
        ranges = {label: (interval.left, interval.right) for label, interval in zip(bin_labels, intervals)}
        bins = bins.cat.rename_categories(bin_labels)
        ids_by_bin = {label: quantile_df.loc[bins == label, 'id'] for label in bin_labels}

        pr_h = healthy_reference.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_h = pr_h.rename(columns={'glucose mmol/l': 'median'})

        pr_e = established_reference.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_e = pr_e.rename(columns={'glucose mmol/l': 'median'})
        pr_c = df_control.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_c = pr_c.rename(columns={'glucose mmol/l': 'median'})
        pr_a = df_active.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
        pr_a = pr_a.rename(columns={'glucose mmol/l': 'median'})

        for quantile in bin_labels:
            ids = ids_by_bin[quantile]
            if ids.empty:
                continue
            subset = df_group[df_group['id'].isin(ids)]
            subset_control = df_control[df_control['id'].isin(ids)]
            subset_active = df_active[df_active['id'].isin(ids)]
            subset_endpoints = group_endpoints[group_endpoints['id'].isin(ids)]
            subset_control_endpoints = control_endpoints[control_endpoints['id'].isin(ids)]
            subset_active_endpoints = active_endpoints[active_endpoints['id'].isin(ids)]
            patient_results = subset.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
            patient_results = patient_results.rename(columns={'glucose mmol/l': 'median'})
            control_results = subset_control.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
            control_results = control_results.rename(columns={'glucose mmol/l': 'median'})
            active_results = subset_active.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
            active_results = active_results.rename(columns={'glucose mmol/l': 'median'})

            if patient_results.empty:
                continue
            group_agp = patient_results.groupby('tod')['median'].quantile(q=[0.10,0.25,0.5,0.75,0.90]).unstack()
            if group_agp.empty:
                continue
            group_agp.columns = ['10th Centile','q1','median','q3','90th Centile']

            summary_info = []
            summary_sources = [
                ('Group overall', subset_endpoints, '#f57c00', 'dashed'),
                ('Group control', subset_control_endpoints, "#0019f5", 'dashed'),
                ('Group active', subset_active_endpoints, "#e5f500", 'dashed'),
                ('Healthy overall', healthy_endpoints, '#00f514', 'dashdot'),
                ('Established overall', established_endpoints, '#f50000', (0, (3, 2)))
            ]
            for label, endpoints_df, color, linestyle in summary_sources:
                median_value = endpoints_df['median_glucose'].median()
                if pd.notna(median_value):
                    summary_info.append({
                        'label': label,
                        'value': median_value,
                        'color': color,
                        'linestyle': linestyle
                })

            ref_he = pr_h.groupby('tod')['median'].median().to_frame('value')
            ref_e = pr_e.groupby('tod')['median'].median().to_frame('value')
            ref_c = pr_c.groupby('tod')['median'].median().to_frame('value')
            ref_a = pr_a.groupby('tod')['median'].median().to_frame('value')
            subset_ref_c = control_results.groupby('tod')['median'].median().to_frame('value')
            subset_ref_a = active_results.groupby('tod')['median'].median().to_frame('value')

            xh = np.array([t.hour + t.minute/60 + t.second/3600 for t in group_agp.index], dtype=float)

            fig = plt.figure(figsize=(7.5, 7.2))
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.8, 4], hspace=0.4)
            ax_table = fig.add_subplot(gs[0])
            ax_plot = fig.add_subplot(gs[1])

            metrics_to_display = [
                ('hb_a1c', 'Hb A1C (%)'),
                ('TIR', 'TIR (%)'),
                ('TITR', 'TITR (%)'),
                ('TBR_Lvl_1', 'TBR Lvl 1 (%)'),
                ('TBR_Lvl_2', 'TBR Lvl 2 (%)'),
                ('TAR_Lvl_1', 'TAR Lvl 1 (%)'),
                ('TAR_Lvl_2', 'TAR Lvl 2 (%)'),
                ('min_glucose', 'Min (mmol/L)'),
                ('max_glucose', 'Max (mmol/L)'),
                ('std_glucose', 'Std (mmol/L)'),
                ('cv_percent', 'CV (%)'),
                ('GVP', 'GVP')
            ]

            def _format_p_value(p_value: float) -> str:
                if pd.isna(p_value):
                    return 'n/a'
                if p_value < 0.001:
                    return '<0.001**'
                if p_value < 0.05:
                    return f'{p_value:.3f}*'
                return f'{p_value:.3f}'

            anova_results = {}
            for column, _ in metrics_to_display:
                samples = [
                    subset_endpoints[column].dropna(),
                    subset_control_endpoints[column].dropna(),
                    subset_active_endpoints[column].dropna(),
                    established_endpoints[column].dropna()
                ]
                valid_samples = [s for s in samples if len(s) > 0]
                if len(valid_samples) < 2:
                    anova_results[column] = np.nan
                    continue
                try:
                    sample_groups = {
                        f'group_{idx + 1}': sample
                        for idx, sample in enumerate(valid_samples)
                    }
                    _, anova_p = self.run_oneway_anova_test(sample_groups)
                    anova_results[column] = anova_p
                except Exception:
                    anova_results[column] = np.nan

            table_rows = []
            for column, label in metrics_to_display:
                values = [
                    f'{subset_endpoints[column].median():.1f}',
                    f'{subset_control_endpoints[column].median():.1f}',
                    f'{subset_active_endpoints[column].median():.1f}',
                    f'{healthy_endpoints[column].median():.1f}',
                    f'{established_endpoints[column].median():.1f}',
                ]
                p_value_fmt = _format_p_value(anova_results.get(column, np.nan))
                table_rows.append([label, *values, p_value_fmt])

            ax_table.axis('off')
            table = ax_table.table(
                cellText=table_rows,
                colLabels=['Metric', 'Group', 'Control', 'Active', 'Healthy', 'Established', 'ANOVA'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.15)

            fig.text(0.5, 0.965, "Median CGM Endpoints", ha='center', va='top', fontsize=11, fontweight='bold')

            ax_plot.plot(xh, group_agp['median'], label='Median', color='#f57c00')
            ax_plot.plot(xh, ref_he['value'], label='Healthy Reference', color="#00f514")
            ax_plot.plot(xh, ref_e['value'], label='Established Reference', color="#f50000")
            if not subset_ref_c.empty:
                ax_plot.plot(xh, subset_ref_c['value'], label='Control Median', color="#0019f5")
            elif not ref_c.empty:
                ax_plot.plot(xh, ref_c['value'], label='Control Median', color="#0019f5")
            if not subset_ref_a.empty:
                ax_plot.plot(xh, subset_ref_a['value'], label='Active Median', color="#e5f500")
            elif not ref_a.empty:
                ax_plot.plot(xh, ref_a['value'], label='Active Median', color="#e5f500")
            ax_plot.fill_between(xh, group_agp['q1'], group_agp['q3'], color='#08306b', alpha=0.7, label='25-75th Centile')
            ax_plot.fill_between(xh, group_agp['10th Centile'], group_agp['90th Centile'], color='#6baed6', alpha=0.5, label='10-90th Centile')
            ax_plot.set_xlim(0, 24)
            ax_plot.set_xticks(
                np.arange(0, 25, 4),
                [f"{int(h):02d}:00" for h in np.arange(0, 25, 4)]
            )
            low, high = ranges[quantile]
            ax_plot.set_title(f"Group-level AGP; {FEATURE_LABELS_WITH_UNITS[quantile_feature]}: {low:.1f}-{high:.1f}", pad=15)
            ax_plot.set_xlabel("Time of Day")
            ax_plot.set_ylabel('Glucose (mmol/l)')
            ax_plot.set_ylim((4,12))

            legend_handles, legend_labels = ax_plot.get_legend_handles_labels()
            # for entry in summary_info:
            #     line = ax_plot.axhline(
            #         entry['value'],
            #         color=entry['color'],
            #         linestyle=entry['linestyle'],
            #         linewidth=1.2
            #     )
            #     legend_handles.append(line)
            #     legend_labels.append(f"{entry['label']} (Median)")

            ax_plot.legend(
                legend_handles,
                legend_labels,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                frameon=False
            )
            fig.savefig(path+f'{quantile_feature}/{quantile}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)

        return None

    def plot_AGP_quartiles_curves_only(
        self,
        df_group: pd.DataFrame,
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        quantile_feature: str,
        path: str
    ) -> None:
        """
        Plot quartile panels with only four median curves: Healthy, Established T1D, AID, and Standard Care.

        Removes tables and percentile shading; saves one PNG per quantile bin.
        """
        df_group = df_group.copy()
        df_group['treatment_arm'] = df_group['treatment_arm'].str.lower()
        df_control = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_1)]
        df_active = df_group[df_group['treatment_arm'].isin(TREATMENT_GROUP_2)]
        healthy_reference = healthy_reference.copy()
        established_reference = established_reference.copy()

        df_group['tod'] = df_group['timestamp'].dt.floor('15min').dt.time
        df_control['tod'] = df_control['timestamp'].dt.floor('15min').dt.time
        df_active['tod'] = df_active['timestamp'].dt.floor('15min').dt.time
        healthy_reference['tod'] = healthy_reference['timestamp'].dt.floor('15min').dt.time
        established_reference['tod'] = established_reference['timestamp'].dt.floor('15min').dt.time

        group_endpoints = self.get_cgm_core_endpoints_general(df_group)

        if quantile_feature in CGM_CORE_ENDPOINTS:
            quantile_df = group_endpoints[['id', quantile_feature]].dropna()
        else:
            quantile_df = (
                df_group[['id', quantile_feature]]
                .dropna(subset=[quantile_feature])
                .groupby('id', as_index=False)[quantile_feature]
                .median()
            )

        vals = quantile_df[quantile_feature]
        base_labels = ['Q1', 'Q2', 'Q3', 'Q4']

        if quantile_feature == 'cpep_auc':
            edges = [0, 0.25, 0.5, 0.75, 1]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'cpep_auc_preservation':
            edges = [0, 25, 50, 75, 100]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'beta2_score':
            edges = [0, 5, 10, 15]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        else:
            lo, hi = vals.quantile([0.05, 0.95])
            clipped_vals = vals.clip(lo, hi)
            quantile_probs = [0, 0.25, 0.5, 0.75, 1]
            bins = pd.qcut(
                clipped_vals,
                q=quantile_probs,
                labels=None,
                duplicates='drop'
            )

        intervals = list(bins.cat.categories)
        num_bins = len(intervals)
        bin_labels = base_labels[:num_bins]
        if len(bin_labels) < num_bins:
            bin_labels.extend([f'Q{i+1}' for i in range(len(bin_labels), num_bins)])
        ranges = {label: (interval.left, interval.right) for label, interval in zip(bin_labels, intervals)}
        bins = bins.cat.rename_categories(bin_labels)
        ids_by_bin = {label: quantile_df.loc[bins == label, 'id'] for label in bin_labels}

        def median_curve(df: pd.DataFrame) -> pd.Series:
            pr = df.groupby(['id','tod'])['glucose mmol/l'].median().reset_index()
            pr = pr.rename(columns={'glucose mmol/l': 'median'})
            return pr.groupby('tod')['median'].median()

        ref_h = median_curve(healthy_reference)
        ref_e = median_curve(established_reference)
        ref_c = median_curve(df_control)
        ref_a = median_curve(df_active)

        colors = {
            'Healthy': "#00f514",
            'Established T1D': "#f50000",
            'Standard Care': "#0019f5",
            'AID': "#e5f500"
        }

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for quantile in bin_labels:
            ids = ids_by_bin[quantile]
            if ids.empty:
                continue

            subset_control = df_control[df_control['id'].isin(ids)]
            subset_active = df_active[df_active['id'].isin(ids)]

            curve_h = ref_h
            curve_e = ref_e
            curve_c = median_curve(subset_control) if not subset_control.empty else pd.Series(dtype=float)
            curve_a = median_curve(subset_active) if not subset_active.empty else pd.Series(dtype=float)

            fig, ax = plt.subplots(figsize=(6, 4))

            def _plot_curve(curve: pd.Series, label: str, color: str):
                if curve.empty:
                    return
                xh = np.array([t.hour + t.minute/60 + t.second/3600 for t in curve.index], dtype=float)
                ax.plot(xh, curve.values, label=label, color=color, linewidth=2.0)

            _plot_curve(curve_h, 'Healthy', colors['Healthy'])
            _plot_curve(curve_e, 'Established T1D', colors['Established T1D'])
            _plot_curve(curve_c, 'Standard Care', colors['Standard Care'])
            _plot_curve(curve_a, 'AID', colors['AID'])

            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 4))
            ax.set_xticklabels([f"{int(h):02d}:00" for h in np.arange(0, 25, 4)], rotation=0)
            ax.set_ylim(4, 12)
            low, high = ranges[quantile]
            ax.set_title(f"{FEATURE_LABELS_WITH_UNITS.get(quantile_feature, quantile_feature)}: {low:.2f}–{high:.2f}", fontsize=11, fontweight='bold')
            ax.set_ylabel('Glucose (mmol/L)')
            ax.set_xlabel('Time of Day')
            ax.grid(False)
            ax.set_facecolor('white')

            handles = [
                plt.Line2D([0], [0], color=colors['Healthy'], label='Healthy', linewidth=2.0),
                plt.Line2D([0], [0], color=colors['Established T1D'], label='Established T1D', linewidth=2.0),
                plt.Line2D([0], [0], color=colors['Standard Care'], label='Standard Care', linewidth=2.0),
                plt.Line2D([0], [0], color=colors['AID'], label='AID', linewidth=2.0),
            ]
            ax.legend(handles=handles, loc='upper right', frameon=False)

            fig.tight_layout()
            fig.savefig(out_dir / f'curves_only_{quantile}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        return None

    def plot_AGP_quartiles_curves_only_by_arm_combined(
        self,
        df_group: pd.DataFrame,
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        quantile_feature: str,
        arm_label: str,
        arm_values: list[str],
        arm_color: str,
        path: str
    ) -> None:
        """
        Plot one figure per arm showing all quartile median curves plus healthy/established references.
        """
        df_group = df_group.copy()
        df_group['treatment_arm'] = df_group['treatment_arm'].str.lower()
        df_arm = df_group[df_group['treatment_arm'].isin(arm_values)]
        if df_arm.empty:
            return None

        healthy_reference = healthy_reference.copy()
        established_reference = established_reference.copy()

        df_group['tod'] = df_group['timestamp'].dt.floor('15min').dt.time
        df_arm['tod'] = df_arm['timestamp'].dt.floor('15min').dt.time
        healthy_reference['tod'] = healthy_reference['timestamp'].dt.floor('15min').dt.time
        established_reference['tod'] = established_reference['timestamp'].dt.floor('15min').dt.time

        group_endpoints = self.get_cgm_core_endpoints_general(df_group)

        if quantile_feature in CGM_CORE_ENDPOINTS:
            quantile_df = group_endpoints[['id', quantile_feature]].dropna()
        else:
            quantile_df = (
                df_group[['id', quantile_feature]]
                .dropna(subset=[quantile_feature])
                .groupby('id', as_index=False)[quantile_feature]
                .median()
            )

        if quantile_df.empty:
            return None

        vals = quantile_df[quantile_feature]
        base_labels = ['Q1', 'Q2', 'Q3', 'Q4']

        if quantile_feature == 'cpep_auc':
            edges = [0, 0.25, 0.5, 0.75, 1]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'cpep_auc_preservation':
            edges = [0, 25, 50, 75, 100]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        elif quantile_feature == 'beta2_score':
            edges = [0, 5, 10, 15]
            clipped_vals = vals.clip(edges[0], edges[-1])
            bins = pd.cut(clipped_vals, bins=edges, labels=None, include_lowest=True, duplicates='drop')
        else:
            lo, hi = vals.quantile([0.05, 0.95])
            clipped_vals = vals.clip(lo, hi)
            quantile_probs = [0, 0.25, 0.5, 0.75, 1]
            bins = pd.qcut(
                clipped_vals,
                q=quantile_probs,
                labels=None,
                duplicates='drop'
            )

        intervals = list(bins.cat.categories)
        num_bins = len(intervals)
        bin_labels = base_labels[:num_bins]
        if len(bin_labels) < num_bins:
            bin_labels.extend([f'Q{i+1}' for i in range(len(bin_labels), num_bins)])
        ranges = {label: (interval.left, interval.right) for label, interval in zip(bin_labels, intervals)}
        bins = bins.cat.rename_categories(bin_labels)
        ids_by_bin = {label: quantile_df.loc[bins == label, 'id'] for label in bin_labels}

        def median_curve(df: pd.DataFrame) -> pd.Series:
            pr = df.groupby(['id', 'tod'])['glucose mmol/l'].median().reset_index()
            pr = pr.rename(columns={'glucose mmol/l': 'median'})
            return pr.groupby('tod')['median'].median()

        ref_h = median_curve(healthy_reference)
        ref_e = median_curve(established_reference)

        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
        quartile_colors = {
            label: palette[idx % len(palette)]
            for idx, label in enumerate(bin_labels)
        }

        fig, ax = plt.subplots(figsize=(7.5, 5.5))

        for label in bin_labels:
            ids = ids_by_bin[label]
            if ids.empty:
                continue
            subset_arm = df_arm[df_arm['id'].isin(ids)]
            if subset_arm.empty:
                continue
            curve = median_curve(subset_arm)
            if curve.empty:
                continue
            xh = np.array([t.hour + t.minute/60 + t.second/3600 for t in curve.index], dtype=float)
            low, high = ranges[label]
            ax.plot(
                xh,
                curve.values,
                label=f"{arm_label} {label} ({low:.2f}–{high:.2f})",
                color=quartile_colors[label],
                linewidth=2.0
            )

        if not ref_h.empty:
            xh_h = np.array([t.hour + t.minute/60 + t.second/3600 for t in ref_h.index], dtype=float)
            ax.plot(xh_h, ref_h.values, label='Healthy', color="#00f514", linewidth=2.0)
        if not ref_e.empty:
            xh_e = np.array([t.hour + t.minute/60 + t.second/3600 for t in ref_e.index], dtype=float)
            ax.plot(xh_e, ref_e.values, label='Established T1D', color="#f50000", linewidth=2.0)

        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xticklabels([f"{int(h):02d}:00" for h in np.arange(0, 25, 4)])
        ax.set_ylim(4, 12)
        ax.set_title(f"{arm_label} AGP Quartiles; {FEATURE_LABELS_WITH_UNITS.get(quantile_feature, quantile_feature)}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time of Day")
        ax.set_ylabel('Glucose (mmol/L)')
        ax.grid(False)
        ax.set_facecolor('white')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return None

    def plot_AGP_random_subject_all_windows(
        self,
        studies: list[str] | None = None,
        wear_days_options: list[int] | None = None,
        wear_prct: int = 70,
        seed: int = 42,
        output_dir: str = './data/graphs/feature_analysis/AGP/random_subject_wear_windows'
    ) -> pd.DataFrame:
        """
        For each requested study, randomly select one control and one treatment subject
        from the same time bin, then overlay AGP median curves for multiple wear windows.

        Selection rule:
            - Candidate subjects must have a qualifying window for the largest wear day
              option (default 14 days) at the requested wear percentage threshold.

        Plot content (per study):
            - Control subject lines: wear windows [14, 10, 7, 5, 3] (if available)
            - Treatment subject lines: wear windows [14, 10, 7, 5, 3] (if available)
        """
        allowed_studies = ['cloud', 'clvr', 'diagnode', 'jaeb_t1d', 'hupa_ucm']
        if studies is None:
            studies = allowed_studies.copy()
        else:
            studies = [s for s in studies if s in allowed_studies]
        if wear_days_options is None:
            wear_days_options = [14, 10, 7, 5, 3]

        wear_days_options = sorted(set(wear_days_options), reverse=True)
        ref_days = wear_days_options[0]
        rng = random.Random(seed)
        selections = []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _to_pandas(df_like):
            return df_like.compute() if hasattr(df_like, 'compute') else df_like.copy()

        def _prepare_study_df(study_name: str) -> pd.DataFrame:
            df = _to_pandas(self.datasets[study_name]).copy()
            if 'treatment_arm' not in df.columns and 'insulin_delivery' in df.columns:
                df = df.rename(columns={'insulin_delivery': 'treatment_arm'})
            required = {'id', 'timestamp', 'time_bin', 'glucose mmol/l', 'treatment_arm'}
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"{study_name} missing required columns: {sorted(missing)}")

            df = df.dropna(subset=['id', 'timestamp', 'time_bin', 'glucose mmol/l', 'treatment_arm']).copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['id'] = df['id'].astype(str).str.strip()
            df['time_bin'] = df['time_bin'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
            df['treatment_arm'] = df['treatment_arm'].astype(str).str.lower().str.strip()
            df['arm_type'] = np.where(
                df['treatment_arm'].isin(TREATMENT_GROUP_1),
                'control',
                np.where(df['treatment_arm'].isin(TREATMENT_GROUP_2), 'treatment', 'other')
            )
            return df[df['arm_type'].isin(['control', 'treatment'])].copy()

        def _agp_median_curve(df_window: pd.DataFrame) -> pd.DataFrame:
            if df_window.empty:
                return pd.DataFrame(columns=['tod_hour', 'median_glucose'])
            work = df_window.copy()
            work['tod'] = work['timestamp'].dt.floor('15min').dt.time
            curve = (
                work.groupby('tod', as_index=False)['glucose mmol/l']
                .median()
                .rename(columns={'glucose mmol/l': 'median_glucose'})
            )
            curve['tod_hour'] = curve['tod'].apply(lambda t: t.hour + t.minute / 60 + t.second / 3600)
            return curve.sort_values('tod_hour')

        for study_name in studies:
            if study_name not in self.datasets:
                print(f"Skipping {study_name}: study not loaded.")
                continue

            try:
                df_study = _prepare_study_df(study_name)
            except KeyError as exc:
                print(f"Skipping {study_name}: {exc}")
                continue

            if df_study.empty:
                print(f"Skipping {study_name}: no usable CGM rows after filtering.")
                continue

            time_bins = [tb for tb in df_study['time_bin'].dropna().unique().tolist() if tb and tb != 'N/A']
            rng.shuffle(time_bins)

            selected_time_bin = None
            control_id = None
            treatment_id = None

            for tb in time_bins:
                df_tb = df_study[df_study['time_bin'] == tb]
                control_rows = self.get_cgm_windows(df_tb[df_tb['arm_type'] == 'control'], ref_days, wear_prct)
                treatment_rows = self.get_cgm_windows(df_tb[df_tb['arm_type'] == 'treatment'], ref_days, wear_prct)
                control_ids = sorted(control_rows['id'].dropna().unique().tolist())
                treatment_ids = sorted(treatment_rows['id'].dropna().unique().tolist())
                if control_ids and treatment_ids:
                    selected_time_bin = tb
                    control_id = rng.choice(control_ids)
                    treatment_id = rng.choice(treatment_ids)
                    break

            if selected_time_bin is None:
                print(
                    f"Skipping {study_name}: no time bin has both arms with >= {ref_days}-day "
                    f"window at {wear_prct}% wear."
                )
                continue

            df_tb = df_study[df_study['time_bin'] == selected_time_bin]
            arm_specs = [
                ('control', control_id, '#1f77b4'),
                ('treatment', treatment_id, '#d62728')
            ]
            line_styles = {14: '-', 10: '--', 7: '-.', 5: ':', 3: (0, (3, 1, 1, 1))}

            plt.figure(figsize=(9, 5))
            plotted = 0
            for arm_type, subject_id, color in arm_specs:
                df_subject = df_tb[(df_tb['arm_type'] == arm_type) & (df_tb['id'] == subject_id)].copy()
                for wear_days in wear_days_options:
                    df_window = self.get_cgm_windows(df_subject, wear_days, wear_prct)
                    curve = _agp_median_curve(df_window)
                    if curve.empty:
                        continue
                    plt.plot(
                        curve['tod_hour'].to_numpy(),
                        curve['median_glucose'].to_numpy(),
                        linestyle=line_styles.get(wear_days, '-'),
                        linewidth=1.7,
                        color=color,
                        alpha=0.95,
                        label=f"{arm_type.title()} ({subject_id}) {wear_days}d"
                    )
                    plotted += 1

            if plotted == 0:
                plt.close()
                print(f"Skipping {study_name}: no curves could be plotted after windowing.")
                continue

            plt.xlim(0, 24)
            plt.xticks(np.arange(0, 25, 4), [f"{int(h):02d}:00" for h in np.arange(0, 25, 4)])
            plt.xlabel('Time of Day')
            plt.ylabel('Glucose (mmol/L)')
            plt.ylim(4, 18)
            plt.title(
                f"{study_name.upper()} AGP by wear window ({selected_time_bin})\n"
                f"Control={control_id} | Treatment={treatment_id} | wear>={wear_prct}%"
            )
            plt.grid(alpha=0.25, linewidth=0.5)
            plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
            plt.tight_layout()

            out_path = out_dir / f"{study_name}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

            selections.append({
                'study': study_name,
                'time_bin': selected_time_bin,
                'control_id': control_id,
                'treatment_id': treatment_id,
                'wear_prct': wear_prct,
                'wear_days_options': ','.join(str(d) for d in wear_days_options),
                'plot_path': str(out_path)
            })

        selections_df = pd.DataFrame(selections)
        selections_path = out_dir / 'selected_subjects.csv'
        selections_df.to_csv(selections_path, index=False)
        print(f"Saved study selections to {selections_path}")
        return selections_df

    def plot_analyzed_features_general(self, feature_name: str) -> None:
        """
        Plot a feature distribution (boxplot) across all groups in `self.analyzed_datasets`.

        Args:
            feature_name (str): Column name of the feature to visualize on the y-axis.

        Returns:
            None: The figure is saved to disk at
                './data/graphs/feature_analysis/general/{feature_name}_analysis.png'.

        Raises:
            ValueError: If `self.analyzed_datasets` is empty or `feature_name` is missing in any dataset.
        """
        plot_dict_df = self.analyzed_datasets.copy()
        df_plot = pd.DataFrame([])

        for df_name, df in plot_dict_df.items():
            df['group'] = df_name
            df_plot = pd.concat([df_plot, df])

        plt.figure(figsize=(8,5))
        sns.boxplot(x='group', y= feature_name, data=df_plot)
        plt.title(f'{feature_name} Comparision')
        plt.ylabel(feature_name)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./data/graphs/feature_analysis/general/{feature_name}_analysis.png')
        plt.close()

    def plot_availability(
        self,
        df: pd.DataFrame,
        out_path: str,
    ) -> None:
        """
        Create a PNG table showing Total, Missing, Present, %Missing, %Present
        for each feature, given:
        - n_missing: pandas Series (index = feature, value = missing-count)
        - n_total:   total row count (same for all features)
        - out_dir:   directory to write the PNG into (created if missing)

        Returns the path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(len(df.columns)*0.8, len(df)*0.3 + 1))
        ax.axis('off')
        ax.axis('tight')

        # Create the table
        tbl = ax.table(cellText=df.values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.2)

        ax.set_title('Completeness Table', fontsize=12, pad=20)

        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_feature_charts(self, mat, features: list, df_name: str) -> None:
        """
        Plot per-feature daily observation counts as vertical line charts and save to disk.

        Args:
            mat: DataFrame indexed by day with feature count columns.
            features: List of feature column names to plot.
            df_name: Name of the dataset for titling the figure/output file.

        Returns:
            None
        """
        n = len(features)
        fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(12, max(2.4 * n, 3)))
        if n == 1:
            axes = [axes]

        for ax, f in zip(axes, features):
            # bar-spike style using vlines on integer x positions
            x = mat.index.astype(int)
            y = mat[f].astype(float).values
            ax.vlines(x, 0, y, linewidth=2)
            ax.set_ylabel(f, rotation=0, labelpad=40, ha='right')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[-1].set_xlabel('Day since first measurement')
        fig.suptitle(f"{df_name}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        outdir = './data/graphs/feature_analysis/isi_histograms'
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(f'{outdir}/{df_name}.png')
        plt.close(fig)

        return None

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        path: str,
        bins: int = 15,
        color: str = 'blue',
        edgecolor: str = 'black'
    ) -> None:
        """
        Plot per-feature distributions for a dataset (histograms for numeric, bar charts for categorical).

        Args:
            df (pd.DataFrame): Input table of features.
            path (str): Output directory to write per-feature PNG files.
            bins (int): Number of histogram bins for numeric columns. Defaults to 15.
            color (str): Matplotlib color for bars/bins. Defaults to 'blue'.
            edgecolor (str): Edge color for bars/bins. Defaults to 'black'.

        Returns:
            None: One image per feature is saved to `path`, and plots are closed.

        Raises:
            ValueError: If `df` has no plottable columns after excluding identifiers/time fields.
        """
        features = [col for col in df.columns if col not in ['id', 'timestamp', 'timestamp_seconds']]

        for col in features:
            sub_df = df[['id', col]]
            if col not in ['timestamp_type', 'glucose mmol/l']:
                sub_df = sub_df.drop_duplicates(subset=['id', col], keep='first')

            s = sub_df[col].compute()
            plt.figure(figsize=(8, 5))

            if pd.api.types.is_numeric_dtype(s):
                plt.hist(s.dropna(), bins=bins, alpha=0.7, color=color, edgecolor=edgecolor)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            else:
                vc = s.value_counts()
                plt.bar(vc.index.astype(str), vc.values, alpha=0.7, color=color, edgecolor=edgecolor)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Value Counts of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
            
            plt.grid(axis='y', alpha=0.75)
            safe_col = re.sub(r'[\\/*?:"<>|]', "_", col)  # replace forbidden/path chars
            file_path = os.path.join(path, f"{safe_col}.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_grouped_time_bins(self, dict_df: dict[str, pd.DataFrame], feature: str, path: str) -> None:  
        """
        Plot a feature across studies and time bins with per-study Spearman correlation annotations.

        Args:
            dict_df (dict[str, pd.DataFrame]): Mapping of study name to DataFrame.
                Each DataFrame must include 'time_bin' and the `feature` column.
            feature (str): Feature/column to visualize on the y-axis.
            path (str): Output file path for the saved figure.

        Returns:
            None: The figure is saved to disk and the plot is closed.

        Raises:
            ValueError: If any DataFrame in `dict_df` lacks required columns ('time_bin' or `feature`).
        """
        df_all = []
        for study_name, df in dict_df.items():
            df = df.copy()
            df['study'] = study_name
            df_all.append(df)
        df_all = pd.concat(df_all, ignore_index=True)

        # Create numeric month column for sorting (e.g., Months 0 → 0)
        df_all['Month_num'] = (
            df_all['time_bin']
            .str.extract(r'(\d+)')
            .astype(float)
            .fillna(-1)  # use -1 for "Healthy"
        )

        # Set consistent study order
        study_order = ['jaeb_healthy'] + [s for s in dict_df.keys() if s != 'jaeb_healthy']
        df_all['study'] = pd.Categorical(df_all['study'], categories=study_order, ordered=True)

        # Combine 'study' + 'time_bin' into one column
        df_all['study_time'] = df_all['study'].astype(str) + ' | ' + df_all['time_bin'].astype(str)

        # Sort by study and month
        df_all = df_all.sort_values(by=['study', 'Month_num'])

        # Color palette
        palette = sns.color_palette("Set2", len(study_order))
        study_color_map = dict(zip(study_order, palette))
        study_time_color_map = {
            row['study_time']: study_color_map[row['study']]
            for _, row in df_all.iterrows()
        }

        # Plot
        plt.figure(figsize=(18, 6))
        ax = sns.boxplot(x='study_time', y=feature, data=df_all,
                        palette=study_time_color_map, showfliers=False)
        sns.stripplot(x='study_time', y=feature, data=df_all,
                    color='black', alpha=0.3, jitter=True)

        # Annotate Spearman correlation
        xtick_labels = ax.get_xticklabels()
        xtick_positions = ax.get_xticks()
        label_to_pos = {label.get_text(): pos for label, pos in zip(xtick_labels, xtick_positions)}

        for study in study_order:
            if study.lower().startswith('jaeb_healthy'):
                continue

            df_study = df_all[df_all['study'] == study]
            try:
                if df_study['Month_num'].nunique() > 1:
                    rho, pval = self.run_spearman_test(df_study['Month_num'], df_study[feature])
                    significant = pval < 0.05
                else:
                    rho, pval, significant = np.nan, np.nan, False

                positions = [label_to_pos[label] for label in label_to_pos if label.startswith(study)]
                if positions:
                    x_pos = sum(positions) / len(positions)
                    if not np.isnan(rho):
                        abs_rho = abs(rho)
                        if abs_rho < 0.2:
                            corr_strength = 'Very Weak'; color = '#e74c3c'
                        elif abs_rho < 0.4:
                            corr_strength = 'Weak'; color = '#e67e22'
                        elif abs_rho < 0.6:
                            corr_strength = 'Moderate'; color = '#f1c40f'
                        elif abs_rho < 0.8:
                            corr_strength = 'Strong'; color = '#2ecc71'
                        else:
                            corr_strength = 'Very Strong'; color = '#27ae60'
                        ax.text(x_pos, df_all[feature].max() * 1.1,
                                f"Spearman ρ={rho:.3f}, p={pval:.3g}",
                                ha='center', va='bottom', fontsize=8)
                        if significant:
                            ax.text(x_pos, df_all[feature].max() * 1.06,
                                    corr_strength,
                                    ha='center', va='bottom', fontsize=8,
                                    fontweight='bold',
                                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
                    # Always show study title, even if correlation is NaN
                    ax.text(x_pos, df_all[feature].max() * 1.13,
                            study.upper(),
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
            except Exception as e:
                print(f"Could not compute Spearman for {study}: {e}")

        # Clean x-tick labels
        clean_labels = [
            label.get_text().split("Month")[-1].strip() if "Month" in label.get_text() else "Healthy"
            for label in ax.get_xticklabels()
        ]
        ax.set_xticks(range(len(clean_labels)))
        ax.set_xticklabels(clean_labels)

        ax.set_xlabel("Month")
        ax.set_ylabel(feature)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_hypothesis_test_general(
        self,
        df_healthy: pd.DataFrame,
        df_t1d: pd.DataFrame,
        feature_name: str,
        test_results: dict[str, object],
        path: str
    ) -> None:
        """
        Plot a healthy vs. T1D comparison for a single feature using boxplots and annotate with test results.

        Args:
            df_healthy (pd.DataFrame): DataFrame of healthy cohort records.
            df_t1d (pd.DataFrame): DataFrame of T1D cohort records.
            feature_name (str): Column name of the feature to compare on the y-axis.
            test_results (dict[str, object]): Statistical test results with:
                - 'test_used' (str): Name of the test performed (e.g., 'Mann–Whitney U').
                - 'p_value' (float): P-value from the test.
            path (str): Output file path for the saved figure.

        Returns:
            None: The figure is saved to disk and the plot is closed.

        Raises:
            KeyError: If required keys ('test_used', 'p_value') are missing in `test_results`.
            ValueError: If `feature_name` is missing from either input DataFrame.
        """
        df_healthy_plot = df_healthy.copy()
        df_t1d_plot = df_t1d.copy()

        df_healthy_plot['group'] = 'Healthy'
        df_t1d_plot['group'] = "T1D" #Change this for the T1D group
        df_plot = pd.concat([df_healthy_plot,df_t1d_plot])

        plt.figure(figsize=(8,5))
        sns.boxplot(x='group', y=feature_name, data=df_plot)

        p_value = test_results['p_value']
        plt.title(f'{feature_name} Comparision\n{test_results['test_used']} | p = {p_value:.3g}')

        plt.ylabel(feature_name)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path) #Change this for the T1D group
        plt.close()

    def plot_spearman_heatmap_from_dfs(
        self,
        dict_dfs: dict[str, pd.DataFrame],
        clinical_feature: str,
        cgm_features: list[str],
        path: str
    ) -> None:
        """
        Build a color-coded table (heatmap-like) of Spearman correlations across datasets.

        Args:
            dict_dfs (dict[str, pd.DataFrame]): Mapping of dataset name to DataFrame.
            clinical_feature (str): Clinical variable to correlate against CGM features.
            cgm_features (list[str]): List of CGM feature column names to include.
            path (str): Output file path for the main figure; legend is saved as
                `{path.replace('.png', '_legend.png')}`.

        Returns:
            None: Figures are saved to disk and plots are closed.

        Raises:
            ValueError: If any DataFrame lacks required columns (`clinical_feature` or items in `cgm_features`).
        """
        results = []
        desired_order = cgm_features.copy()

        for study_name, df in dict_dfs.items():
            if df.empty:
                continue
            else:
                df = df.copy()
                df['Month_num'] = (
                df['time_bin']
                .str.extract(r'(\d+)')
                .astype(float)
                .fillna(-1)  # use -1 for "Healthy"
            )
                
            for feature in cgm_features:
                try:
                    rho, pval = self.run_spearman_test(df[clinical_feature], df[feature])
                    results.append({
                        'Dataset': study_name,
                        'Feature': feature,
                        'rho': rho,
                        'pval': pval,
                        'label': f"{rho:.3f}\n{pval:.3g}"
                    })
                except Exception as e:
                    print(f"Skipping {study_name} | {feature}: {e}")

        df_results = pd.DataFrame(results)
        pivot_rho = df_results.pivot(index='Dataset', columns='Feature', values='rho')
        pivot_label = df_results.pivot(index='Dataset', columns='Feature', values='label')
        pivot_rho = pivot_rho[desired_order]
        pivot_label = pivot_label[desired_order]

        # Define custom color based on rho values
        def correlation_color(val):
            """
            Map a Spearman rho value to a categorical color for table visualization.

            Args:
                val: Correlation coefficient (float).

            Returns:
                str: Hex color code representing the correlation strength.
            """
            if abs(val) >= 0.8:
                return '#00cc00'  # Green
            elif abs(val) >= 0.6:
                return '#99ff99'  # Light green
            elif abs(val) >= 0.4:
                return '#ffff66'  # Yellow
            elif abs(val) >= 0.2:
                return '#ff9933'  # Orange
            else:
                return '#ff3300'  # Red

        cell_colors = pivot_rho.copy()
        for row in cell_colors.index:
            for col in cell_colors.columns:
                cell_colors.loc[row, col] = correlation_color(cell_colors.loc[row, col])

        fig, ax = plt.subplots(figsize=(len(cgm_features) * 1.3, len(pivot_rho) * 1.0))
        table = ax.table(
            cellText=pivot_label.values,
            rowLabels=pivot_label.index,
            colLabels=pivot_label.columns,
            cellColours=cell_colors.values,
            loc='center',
            cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.7)

        # Make headers bold
        for key, cell in table.get_celld().items():
            if key[0] == 0 or key[1] == -1:
                cell.set_text_props(weight='bold')

        ax.set_title(f"Spearman correlation: {clinical_feature} vs CGM core endpoints", fontsize=16, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

        legend_elements = [
            Patch(facecolor='#00cc00', edgecolor='black', label='|ρ| ≥ 0.8'),
            Patch(facecolor='#99ff99', edgecolor='black', label='0.6 ≤ |ρ| < 0.8'),
            Patch(facecolor='#ffff66', edgecolor='black', label='0.4 ≤ |ρ| < 0.6'),
            Patch(facecolor='#ff9933', edgecolor='black', label='0.2 ≤ |ρ| < 0.4'),
            Patch(facecolor='#ff3300', edgecolor='black', label='|ρ| < 0.2'),
        ]

        fig_legend, ax_legend = plt.subplots(figsize=(10, 1.5))  # Wider and taller
        ax_legend.axis('off')

        legend = ax_legend.legend(
            handles=legend_elements,
            loc='center',
            ncol=len(legend_elements),
            frameon=False,
            title="|Spearman ρ| Strength",
            title_fontsize=12,
            fontsize=11,
            borderpad=1.2
        )

        # Instead of tight_layout(), use bbox_inches to fit content
        legend_path = path.replace('.png', '_legend.png')
        plt.savefig(legend_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close()

    def plot_taylor_metabolic_endpoints_by_group(
        self,
        metabolic_endpoint: str,
        t1d_metabolic_endpoint: pd.DataFrame,
        healthy_metabolic_endpoint: pd.DataFrame,
        established_metabolic_endpoint: pd.DataFrame,
        x_label: str,
        title_month: str,
        path: str,
        group_labels: list[str] | None = None) -> None:
        """
        Plot a boxplot of a metabolic endpoint stratified by T1D groups plus healthy/established references.

        Args:
            metabolic_endpoint: Column name of the endpoint to plot.
            t1d_metabolic_endpoint: Table containing T1D endpoints with a 'group' column.
            healthy_metabolic_endpoint: Reference table for healthy participants.
            established_metabolic_endpoint: Reference table for established cohort.
            x_label: Label for the x-axis/groups.
            path: Output file path for the saved PNG.
            group_labels: Optional explicit ordering for the group labels.

        Returns:
            None
        """
        if metabolic_endpoint not in t1d_metabolic_endpoint.columns:
            return None
        
        t1d_vals = t1d_metabolic_endpoint.drop_duplicates(["id", metabolic_endpoint])[['group', metabolic_endpoint]].dropna()
        healthy_vals = healthy_metabolic_endpoint.drop_duplicates(["id", metabolic_endpoint])[metabolic_endpoint].dropna()
        established_vals = established_metabolic_endpoint.drop_duplicates(["id", metabolic_endpoint])[metabolic_endpoint].dropna()

        observed_groups = t1d_vals['group'].dropna().astype(str).unique().tolist()
        if group_labels:
            observed_set = set(observed_groups)
            group_order = [str(g) for g in group_labels if str(g) in observed_set]
        else:
            group_order = observed_groups

        data = [t1d_vals.loc[t1d_vals['group'].astype(str) == g, metabolic_endpoint].values
                for g in group_order]
        data.append(healthy_vals.values)
        data.append(established_vals.values)

        labels = group_order + ["Healthy", "Established"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(FEATURE_LABELS_WITH_UNITS.get(metabolic_endpoint, metabolic_endpoint))

        title = f"{FEATURE_LABELS_WITH_UNITS.get(metabolic_endpoint, metabolic_endpoint)} by {x_label}"
        if title_month:
            title += f' ({title_month})'
        ax.set_title(title)

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def get_taylor_patient_endpoints(
        self,
        df: pd.DataFrame,
        metabolic_endpoints: set,
        by_time_bin: bool,
        include_cols: set | None = None
    ) -> pd.DataFrame:
        """
        Build per-patient summaries (optionally per time_bin) for Taylor-style endpoint plots.
        """
        df = df.copy()
        endpoint_cols = set(metabolic_endpoints)
        extra_cols = set(include_cols or set())
        static_cols = [
            c for c in (endpoint_cols | extra_cols)
            if c not in CGM_CORE_ENDPOINTS and c in df.columns
        ]
        static_numeric = [
            c for c in static_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        static_non_numeric = [c for c in static_cols if c not in static_numeric]

        def _first_non_null(series: pd.Series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else np.nan

        if by_time_bin:
            group_cols = ['id', 'time_bin']
            static_parts = []
            if static_numeric:
                static_parts.append(
                    df[group_cols + static_numeric]
                    .groupby(group_cols, as_index=False)
                    .median()
                )
            if static_non_numeric:
                static_parts.append(
                    df[group_cols + static_non_numeric]
                    .groupby(group_cols, as_index=False)
                    .agg(_first_non_null)
                )
            if static_parts:
                static_df = static_parts[0]
                for part in static_parts[1:]:
                    static_df = pd.merge(static_df, part, on=group_cols, how='outer')
            else:
                static_df = df[group_cols].drop_duplicates()

            cgm_records = []
            if any(c in CGM_CORE_ENDPOINTS for c in endpoint_cols):
                for (patient_id, bin_label), group in df.groupby(group_cols):
                    endpoints = self.get_cgm_core_endpoints_per_patient(group)
                    if endpoints:
                        endpoints['id'] = patient_id
                        endpoints['time_bin'] = bin_label
                        cgm_records.append(endpoints)
            cgm_df = pd.DataFrame(cgm_records)

            if cgm_df.empty:
                summary = static_df.copy()
            else:
                summary = pd.merge(cgm_df, static_df, on=group_cols, how='outer')
        else:
            static_parts = []
            if static_numeric:
                static_parts.append(
                    df[['id'] + static_numeric]
                    .groupby('id', as_index=False)
                    .median()
                )
            if static_non_numeric:
                static_parts.append(
                    df[['id'] + static_non_numeric]
                    .groupby('id', as_index=False)
                    .agg(_first_non_null)
                )
            if static_parts:
                static_df = static_parts[0]
                for part in static_parts[1:]:
                    static_df = pd.merge(static_df, part, on='id', how='outer')
            else:
                static_df = df[['id']].drop_duplicates()

            if any(c in CGM_CORE_ENDPOINTS for c in endpoint_cols):
                cgm_df = self.get_cgm_core_endpoints_general(df)
            else:
                cgm_df = pd.DataFrame()

            if cgm_df.empty:
                summary = static_df.copy()
            else:
                summary = pd.merge(cgm_df, static_df, on='id', how='outer')

        for col in endpoint_cols | extra_cols:
            if col not in summary.columns:
                summary[col] = np.nan

        ordered_cols = []
        if 'study' in summary.columns:
            ordered_cols.append('study')
        if 'id' in summary.columns:
            ordered_cols.append('id')
        if by_time_bin and 'time_bin' in summary.columns:
            ordered_cols.append('time_bin')

        include_cols_order = []
        if include_cols is not None:
            if isinstance(include_cols, (list, tuple, pd.Index)):
                include_cols_order = list(include_cols)
            else:
                include_cols_order = sorted(include_cols)
        ordered_cols.extend(
            [col for col in include_cols_order if col in summary.columns and col not in ordered_cols]
        )

        cgm_cols = [col for col in CGM_CORE_ENDPOINTS if col in summary.columns]
        ordered_cols.extend([col for col in cgm_cols if col not in ordered_cols])

        remaining_cols = [col for col in summary.columns if col not in ordered_cols]
        summary = summary[ordered_cols + remaining_cols]

        return summary

    def plot_taylor_time_bins_graph(self, dict_df: dict[str, pd.DataFrame], metrics: list[str], path: str) -> None:
        """
        Plot Taylor-style dependent metrics across time bins, stratified by treatment arm.

        Args:
            dict_df (dict[str, pd.DataFrame]): Mapping of study name to DataFrame used to
                compute dependent metrics (patient-level summaries).
            path (str): Directory path where individual metric figures will be saved.

        Returns:
            None: One PNG per metric is written to `path`.
        """
        time_order = [
            'Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12',
            'Month 15', 'Month 18', 'Month 21', 'Month 24'
        ]
        df_all = pd.concat(dict_df.values(), ignore_index=True)
        df_all['time_bin'] = pd.Categorical(df_all['time_bin'], categories=time_order, ordered=True)
        df_results = self.get_taylor_patient_endpoints(
            df_all,
            metabolic_endpoints={
                'cpep_auc',
                'beta2_score',
                'hb_a1c',
                'gmi',
                'total_ins_dose',
                'TIR',
                'TITR',
                'TBR_Lvl_1',
                'TBR_Lvl_2',
                'TAR_Lvl_1',
                'TAR_Lvl_2'
            },
            by_time_bin=True,
            include_cols={'baseline_cpep_auc', 'treatment_arm'}
        )
        df_results['time_bin'] = pd.Categorical(df_results['time_bin'], categories=time_order, ordered=True)

        df_results['treatment_arm'] = df_results['treatment_arm'].str.lower()
        df_results['treatment_arm'] = np.where(
            df_results['treatment_arm'].isin(TREATMENT_GROUP_1),
            'Control',
            np.where(df_results['treatment_arm'].isin(TREATMENT_GROUP_2), 'Treatment', 'N/A')
        )
        df_results = df_results[df_results['treatment_arm'] != 'N/A']

        df_results['median_cpep'] = df_results['cpep_auc']
        df_results['median_log_cpep'] = np.where(
            df_results['median_cpep'] > 0,
            np.log(df_results['median_cpep']),
            np.nan
        )
        df_results['median_cpep_pct_change'] = np.where(
            (df_results['baseline_cpep_auc'].notna()) & (df_results['baseline_cpep_auc'] != 0),
            100 * (df_results['median_cpep'] - df_results['baseline_cpep_auc']) / df_results['baseline_cpep_auc'],
            np.nan
        )
        df_results['median_beta2_score'] = df_results['beta2_score']
        df_results['median_a1c'] = df_results['hb_a1c']
        df_results['median_gmi'] = df_results['gmi']
        df_results['median_insulin_dose'] = df_results['total_ins_dose']
        df_results['median_tir'] = df_results['TIR']
        df_results['median_titr'] = df_results['TITR']
        df_results['median_tbr_lvl1'] = df_results['TBR_Lvl_1']
        df_results['median_tbr_lvl2'] = df_results['TBR_Lvl_2']
        df_results['median_tar_lvl1'] = df_results['TAR_Lvl_1']
        df_results['median_tar_lvl2'] = df_results['TAR_Lvl_2']
        df_results = df_results.sort_values(['treatment_arm', 'time_bin'])

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.lineplot(
                data=df_results,
                x='time_bin',
                y=metric,
                hue='treatment_arm',
                style='treatment_arm',
                markers=True,
                dashes=False,
                hue_order=['Control', 'Treatment'],
                palette={'Control': '#1f77b4', 'Treatment': '#ff7f0e'},
                errorbar=('ci', 95),
                err_style='bars',
                err_kws={'capsize': 3},
                estimator='mean',
                sort=True
            )
            plt.title(metric.replace('_', ' ').title())
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xlabel('Time (Months)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save to file
            filename = os.path.join(path, f'{metric}.png')
            plt.savefig(filename)
            plt.close()

    def plot_taylor_time_bins_scatterplot(
        self,
        dict_df: dict[str, pd.DataFrame],
        x_feature: str,
        y_features: list[str],
        path: str
    ) -> None:
        """
        Draw faceted regression scatterplots for selected features across key time bins.

        Args:
            dict_df (dict[str, pd.DataFrame]): Mapping of study name to DataFrame.
            x_feature (str): Feature to use on the x-axis for regression.
            y_features (list[str]): Features to plot on the y-axis.
            path (str): Directory path for saving output PNGs.

        Returns:
            None: One PNG per `y_feature` is saved to `path`.

        Raises:
            ValueError: If required columns are missing from the concatenated dataset.
        """
        time_bins_to_include = ['Baseline', 'Month 6', 'Month 12']

        # Preprocess and flatten patient-level data
        all_df = pd.concat(dict_df.values(), ignore_index=True).copy()
        all_df['log_cpep_auc'] = np.log1p(all_df['cpep_auc'])
        all_df['total_ins_dose'] = all_df['total_ins_dose']/all_df['weight']

        # Compute TIR per patient per time_bin
        tir_records = []
        for (pid, bin_label), group in all_df.groupby(['id', 'time_bin']):
            if bin_label not in time_bins_to_include:
                continue
            cgm = self.get_cgm_core_endpoints_per_patient(group)
            if not cgm:
                continue

            tir_records.append({
                'id': pid,
                'time_bin': bin_label,
                'TIR': cgm['TIR'],
                'TBR_Lvl_1': cgm['TBR_Lvl_1'],
                'TBR_Lvl_2': cgm['TBR_Lvl_2'],
                'TAR_Lvl_1': cgm['TAR_Lvl_1'],
                'TAR_Lvl_2': cgm['TAR_Lvl_2'],
                'cv_percent': cgm['cv_percent']
            })
        tir_df = pd.DataFrame(tir_records)

        all_df = all_df.merge(tir_df, on=['id', 'time_bin'], how='left')

        for y_feature in y_features:
            records = []
            for (pid, bin_label), group in all_df.groupby(['id', 'time_bin']): #'treatment_arm_type'
                if bin_label not in time_bins_to_include:
                    continue
                try:
                    x_val = group[x_feature].dropna().median()
                    y_val = group[y_feature].dropna().median()
                except Exception:
                    continue

                records.append({
                    'id': pid,
                    'time_bin': bin_label,
                    x_feature: x_val,
                    y_feature: y_val
                })

            plot_df = pd.DataFrame(records)
            if plot_df.empty:
                continue

            g = sns.lmplot(
                data=plot_df,
                x=x_feature,
                y=y_feature,
                col='time_bin',
                col_order=['Baseline', 'Month 6', 'Month 12'],
                height=5,
                aspect=1.2,
                scatter_kws={'s': 30, 'color': 'black'},
                line_kws={'color': 'blue'},
                ci=95,
                facet_kws=dict(sharey=False)
            )

            for (i, j), ax in np.ndenumerate(g.axes):
                bin_label = g.col_names[j]
                sub = plot_df[(plot_df['time_bin'] == bin_label)]
                valid_points = sub[[x_feature, y_feature]].dropna()
                if len(valid_points) >= 2 and valid_points[x_feature].nunique() > 1 and valid_points[y_feature].nunique() > 1:
                    linreg_result = self.run_linear_regression_test(valid_points[x_feature], valid_points[y_feature])
                    r2 = linreg_result['r_value']**2
                    p_val = linreg_result['p_value']
                    p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
                    text = f"$R^2 = {r2:.2f},\\ n = {len(valid_points)},\\ P {p_str}$"
                else:
                    text = "Insufficient or uniform data"
                ax.text(0.05, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=10)

            for ax in g.axes.flat:
                ax.set_xlabel(FEATURE_LABELS_WITH_UNITS.get(x_feature, x_feature.replace('_', ' ').title()))
                ax.set_ylabel(FEATURE_LABELS_WITH_UNITS.get(y_feature, y_feature.replace('_', ' ').title()))

            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.tight_layout()
            g.fig.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.88)
            fname = f'{x_feature}_vs_{y_feature}.png'
            g.savefig(os.path.join(path, fname), dpi=300, bbox_inches='tight')
            plt.close()

    """ Feature analysis methods """
    def check_cgm_pct_wear(self, df_names: list) -> None:
        dict_dfs = self.datasets.copy()
        dfs_to_check = {name: dict_dfs[name] for name in df_names if name in dict_dfs}
        fail_records = []

        def expected_entries(days: int, interval_minutes: float) -> float:
            """Compute expected CGM rows for a window given the sampling interval."""
            if pd.isna(interval_minutes) or interval_minutes <= 0:
                return 0
            return (days * 24 * 60) / interval_minutes

        def best_consecutive_window(day_counts: pd.Series, window_days: int) -> tuple[bool, int]:
            """
            Find the best (max rows) consecutive `window_days` inside available day counts.
            Returns a flag indicating a qualifying window exists and the max row count inside it.
            """
            if day_counts.empty:
                return False, 0

            day_counts = day_counts.sort_index()
            dates = day_counts.index.to_list()
            counts = day_counts.values

            has_window = False
            best_total = 0
            start_idx = 0

            while start_idx < len(dates):
                end_idx = start_idx + 1
                while end_idx < len(dates) and (dates[end_idx] - dates[end_idx - 1]).days == 1:
                    end_idx += 1

                run_len = end_idx - start_idx
                if run_len >= window_days:
                    has_window = True
                    run_counts = counts[start_idx:end_idx]
                    window_sum = sum(run_counts[:window_days])
                    best_total = max(best_total, window_sum)
                    rolling_sum = window_sum
                    for offset in range(1, run_len - window_days + 1):
                        rolling_sum += run_counts[offset + window_days - 1] - run_counts[offset - 1]
                        best_total = max(best_total, rolling_sum)

                start_idx = end_idx

            return has_window, int(best_total)

        for df_name, df in dfs_to_check.items():
            df_cgm = df[['id', 'timestamp', 'time_bin']].dropna(subset=['id', 'timestamp', 'time_bin']).compute()
            df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
            df_cgm = df_cgm.sort_values(['id', 'time_bin', 'timestamp'])

            total_subjects = df_cgm['id'].nunique()
            print(f"\n=== {df_name} ===")
            print(f"Subjects with CGM entries: {total_subjects}")

            for time_bin, bin_df in df_cgm.groupby('time_bin'):
                bin_ids = bin_df['id'].unique()
                print(f"\nTime bin: {time_bin} | subjects: {len(bin_ids)}")

                ids_with_14 = set()
                ids_with_10 = set()
                passed_14_wear = set()
                passed_10_fallback = set()

                for pid, pid_df in bin_df.groupby('id'):
                    pid_df = pid_df.sort_values('timestamp')

                    median_interval = pid_df['timestamp'].diff().dt.total_seconds().dropna().median()
                    interval_minutes = median_interval / 60 if not pd.isna(median_interval) else 0
                    interval_minutes = 5 if interval_minutes <= 7.5 else 15

                    day_counts = pid_df.groupby(pid_df['timestamp'].dt.floor('D')).size()
                    has_14, best_14_rows = best_consecutive_window(day_counts, 14)
                    has_10, best_10_rows = best_consecutive_window(day_counts, 10)

                    expected_14 = expected_entries(14, interval_minutes)
                    expected_10 = expected_entries(10, interval_minutes)

                    if has_14:
                        ids_with_14.add(pid)
                        wear_14 = (best_14_rows / expected_14) * 100 if expected_14 else 0
                        if wear_14 >= 68:  # 70% with 2% grace
                            passed_14_wear.add(pid)
                        else:
                            fail_records.append({
                                'study': df_name,
                                'time_bin': time_bin,
                                'id': pid,
                                'category': 'failing_14_day_wear_lt_70pct',
                                'best_rows': best_14_rows,
                                'wear_pct': round(wear_14, 2)
                            })
                    else:
                        fail_records.append({
                            'study': df_name,
                            'time_bin': time_bin,
                            'id': pid,
                            'category': 'missing_14_day_streak',
                            'best_rows': best_14_rows,
                            'wear_pct': None
                        })

                    if has_10:
                        ids_with_10.add(pid)
                    else:
                        fail_records.append({
                            'study': df_name,
                            'time_bin': time_bin,
                            'id': pid,
                            'category': 'missing_10_day_streak',
                            'best_rows': best_10_rows,
                            'wear_pct': None
                        })

                    if (not has_14) and has_10:
                        wear_10 = (best_10_rows / expected_10) * 100 if expected_10 else 0
                        if wear_10 >= 78:  # 80% with 2% grace
                            passed_10_fallback.add(pid)
                        else:
                            fail_records.append({
                                'study': df_name,
                                'time_bin': time_bin,
                                'id': pid,
                                'category': 'failing_10_day_wear_lt_80pct',
                                'best_rows': best_10_rows,
                                'wear_pct': round(wear_10, 2)
                            })

                print(f"Subjects with >=14 consecutive days: {len(ids_with_14)}")
                print(f"Subjects with >=10 consecutive days: {len(ids_with_10)}")
                print(f"Subjects meeting 70% wear over best 14-day window: {len(passed_14_wear)}")
                print(f"Subjects (no 14-day streak) meeting 80% wear over 10 days: {len(passed_10_fallback)}")

        if fail_records:
            fail_df = pd.DataFrame(fail_records)
            fail_df.sort_values(by=['study', 'time_bin', 'id'], inplace=True)
            output_path = Path("./data/cgm_wear_failures.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fail_df.to_csv(output_path, index=False)
            print(f"\nSaved CGM wear failures across all studies to {output_path}")
        else:
            print("\nNo CGM wear failures to report.")

    def length_data(self, df: pd.DataFrame) -> tuple[float, float, float]:
        """
        Calculate total time, total Euclidean distance, and normalized distance for LoL analysis.

        Approximates glucose trajectory variability by:
        1) Converting timestamps to datetime.
        2) Downsampling if median interval < 6 minutes (takes every 3rd row).
        3) Computing time and glucose differences (glucose scaled by 18: mmol/L → mg/dL).
        4) Splitting into segments where gaps > 16 minutes.
        5) Summing Euclidean distances sqrt((Δtime_hr)^2 + (Δglucose_mgdl)^2) per segment.

        Args:
            df (pd.DataFrame): Dataset with 'timestamp' and 'glucose mmol/l' columns.

        Returns:
            tuple[float, float, float]: A tuple of:
                - total_time: Total time across segments in hours.
                - total_distance: Sum of Euclidean distances across segments.
                - normalized_distance: Distance per hour (LoL metric).
        """
        pdf = df[['timestamp', 'glucose mmol/l']].copy()
        pdf = pdf.sort_values('timestamp', kind='mergesort')

        ts = pdf['timestamp']
        if np.issubdtype(ts.dtype, np.datetime64):
            t_min = (ts.view('int64') / 1e9) / 60.0  # minutes
        else:
            t_min = (pd.to_datetime(ts, errors='coerce').view('int64') / 1e9) / 60.0

        g = pd.to_numeric(pdf['glucose mmol/l'], errors='coerce').to_numpy(dtype='float64', copy=False)
        t = np.asarray(t_min, dtype='float64')
        valid = np.isfinite(t) & np.isfinite(g)
        t, g = t[valid], g[valid]

        # Downsample if median interval < 6 minutes
        dt_all = np.diff(t)
        med = np.median(dt_all[1:]) if dt_all.size > 1 else np.inf
        if med < 6.0:
            t = t[::3]
            g = g[::3]

        # Gaps
        dt_min = np.diff(t)
        dg_mmol = np.diff(g)
        keep = dt_min < 16.0
        dt_min = dt_min[keep]
        dg_mmol = dg_mmol[keep]
        dt_hr = dt_min / 60.0
        dg_mg = np.abs(dg_mmol) * 18.0

        # GVP
        L_min = np.sum(np.sqrt(dt_min**2 + dg_mg**2))
        L_hr = np.sum(np.sqrt(dt_hr**2 + dg_mg**2))
        L0_min = np.sum(dt_min)
        L0_hr = np.sum(dt_hr)
        gvp_min = ((L_min / L0_min) - 1) * 100
        gvp_hr = ((L_hr / L0_hr) - 1) * 100

        #LoL
        LoL_min = L_min / L0_min
        LoL_hr = L_hr / L0_hr

        return gvp_min, gvp_hr, LoL_min, LoL_hr

    def print_AGP(self):
        """
        Generate AGP-style plots (median/std/cv) for all T1D studies using healthy and established references.

        Args:
            None

        Returns:
            None: Saves plots under './data/graphs/feature_analysis/AGP/{study}/{feature}.png'.
        """
        dict_dfs = self.datasets.copy()
        if 'clvr' in dict_dfs:
            dict_dfs['clvr'] = (
                dict_dfs['clvr']
                .drop(columns=['treatment_arm'], errors='ignore')
                .rename(columns={'insulin_delivery': 'treatment_arm'})
            )

        combined_studies = {}
        if {'clvr', 'cloud'}.issubset(dict_dfs):
            combined_studies['clvr_cloud'] = dd.concat(
                [dict_dfs['clvr'], dict_dfs['cloud']],
                interleave_partitions=True
            )

        all_dict = {k: v for k, v in dict_dfs.items() if k in ALL_STUDIES}
        all_dict.update(combined_studies)

        healthy_reference = dict_dfs['jaeb_healthy'].assign(total_ins_dose=0)
        established_reference = dict_dfs['hupa_ucm']
        healthy_reference_pd = healthy_reference.compute()
        established_reference_pd = established_reference.compute()

        quantile_features = ['cv_percent', 'beta2_score', 'cpep_auc_preservation', 'cpep_auc', 'hb_a1c', 'TIR', 'TITR']
        # quantile_features = ['beta2_score']
        for study_name, df in all_dict.items():
            df = df.compute()

            self.plot_AGP(
                df_group=df,
                healthy_reference=healthy_reference_pd,
                established_reference=established_reference_pd,
                path=f'./data/graphs/feature_analysis/AGP/{study_name}/median.png'
            )
            
            for quantile_feature in quantile_features:
                # self.plot_AGP_quartiles(df, healthy_reference=healthy_reference_pd, established_reference=established_reference_pd,
                #                         quantile_feature=quantile_feature, path=f'./data/graphs/feature_analysis/AGP/{study_name}/')
                self.plot_AGP_quartiles_curves_only(
                    df,
                    healthy_reference=healthy_reference_pd,
                    established_reference=established_reference_pd,
                    quantile_feature=quantile_feature,
                    path=f'./data/graphs/feature_analysis/AGP/{study_name}/{quantile_feature}/curves_only'
                )
                self.combine_pngs(
                    directory=f'./data/graphs/feature_analysis/AGP/{study_name}/{quantile_feature}/curves_only/',
                    output=f'./data/graphs/feature_analysis/AGP/{study_name}/{quantile_feature}/curves_only/combined.png',
                    cols=2,
                    include_filter='curves_only_'
                )
                self.plot_AGP_quartiles_curves_only_by_arm_combined(
                    df,
                    healthy_reference=healthy_reference_pd,
                    established_reference=established_reference_pd,
                    quantile_feature=quantile_feature,
                    arm_label='Standard Care',
                    arm_values=TREATMENT_GROUP_1,
                    arm_color="#0019f5",
                    path=f'./data/graphs/feature_analysis/AGP/{study_name}/{quantile_feature}/curves_only_control/combined.png'
                )
                self.plot_AGP_quartiles_curves_only_by_arm_combined(
                    df,
                    healthy_reference=healthy_reference_pd,
                    established_reference=established_reference_pd,
                    quantile_feature=quantile_feature,
                    arm_label='AID',
                    arm_values=TREATMENT_GROUP_2,
                    arm_color="#e5f500",
                    path=f'./data/graphs/feature_analysis/AGP/{study_name}/{quantile_feature}/curves_only_aid/combined.png'
                )

    def print_consort_general(
            self,
            df_list_raw: list,
            df_list_clean: list,
            df_final: pd.DataFrame
        ) -> None:
        """
        Print a CONSORT-style participant flow summary for merged cohort data.

        Computes the total number of unique participants across all provided domain
        DataFrames, identifies the number present in every domain, and compares these
        with the final merged dataset (produced via inner merges). Reports how many
        subjects were excluded due to missing data in one or more domains, along with
        a non-exclusive per-domain breakdown of missing cases.

        Args:
            df_list (list[pd.DataFrame]): List of domain-specific DataFrames
                (e.g., df_age, df_screening, df_cgm, df_insulin), each containing an 'id' column.
            df_final (pd.DataFrame): Final merged dataset after cleaning and inner merges.

        Returns:
            None: Prints the CONSORT-style summary to the console.

        Raises:
            ValueError: If any of the provided DataFrames are empty or lack the 'id' column.
        """
        # Raw data
        domain_id_sets_raw = []
        for df in df_list_raw:
            domain_id_sets_raw.append(set(pd.Series(df["id"]).dropna().unique()))

        # Data after .dropna()
        domain_id_sets_clean = []
        for df in df_list_clean:
            domain_id_sets_clean.append(set(pd.Series(df["id"]).dropna().unique()))

        all_ids_raw = set().union(*domain_id_sets_raw)
        all_ids_clean = set().union(*domain_id_sets_clean)
        # pd.DataFrame({"all_ids": list(all_ids_raw)}).to_csv("./all_ids.csv", index=False)
        # pd.DataFrame({"all_ids": list(all_ids_clean)}).to_csv("./all_ids.csv", index=False)
        final_ids = set(pd.Series(df_final["id"]).dropna().unique())

        print("=== CONSORT (combined across all sources) ===")
        print(f"Total participants across sources (union): {len(all_ids_raw)}")
        print(f"Final analysis cohort (df_final): {len(final_ids)}")

        print("\n-- Reasons among excluded (non-exclusive; counts can overlap) --")
        for i, (s_raw, s_clean) in enumerate(zip(domain_id_sets_raw, domain_id_sets_clean), start=1):
            missing_here = len(s_raw - s_clean)
            print(f"Dataset_{i}- Before: {len(s_raw)} After:{len(s_clean)} Missing in this domain -> {missing_here}")

        print("\nAll counts are based on unique patient IDs.\n")
        return None

    def print_consort_time_bins(
            self,
            time_bins: list,
            df_list_raw: list,
            df_list_clean: list,
            df_final: pd.DataFrame,
            df_names: list[str] | None = None
        ) -> None:
        """
        Print CONSORT-style inclusion counts per time bin across raw/clean/final datasets.

        Args:
            time_bins: Ordered list of time bin labels to report.
            df_list_raw: List of raw DataFrames for each domain.
            df_list_clean: List of cleaned DataFrames (post-dropna) for each domain.
            df_final: Final merged dataset used for analysis.
            df_names: Optional names aligned to `df_list_clean` for clearer errors.

        Returns:
            None
        """

        for time_bin in time_bins:
            # Raw data
            domain_id_sets_raw = []
            for df in df_list_raw:
                domain_id_sets_raw.append(set(pd.Series(df["id"]).dropna().unique()))

            # Data after .dropna()
            domain_id_sets_clean = []
            cgm_days = None
            for i, df in enumerate(df_list_clean, start=1):
                if 'time_bin' not in df.columns:
                    df_name = df_names[i - 1] if df_names and i - 1 < len(df_names) else f"Dataset_{i}"
                    raise KeyError(
                        f"'time_bin' column missing in {df_name} passed to print_consort_time_bins. "
                        f"Available columns: {list(df.columns)}"
                    )
                df = df[df['time_bin'] == time_bin]
                domain_id_sets_clean.append(set(pd.Series(df["id"]).dropna().unique()))

            all_ids_raw = set().union(*domain_id_sets_raw)

            df_final_time_bin = df_final[df_final['time_bin'] == time_bin]
            final_ids = set(pd.Series(df_final_time_bin["id"]).dropna().unique())

            print(f"=== CONSORT {time_bin} (combined across all sources) ===")
            print(f"Total participants across sources (union): {len(all_ids_raw)}")
            print(f"Final analysis cohort (df_final): {len(final_ids)}")

            print("\n-- Reasons among excluded (non-exclusive; counts can overlap) --")
            for i, (s_raw, s_clean) in enumerate(zip(domain_id_sets_raw, domain_id_sets_clean), start=1):
                missing_here = len(s_raw - s_clean)
                print(f"Dataset_{i}- Before: {len(s_raw)} After:{len(s_clean)} Missing in this domain -> {missing_here}")
            cgm_dfs = [df for df in df_list_clean if 'timestamp' in df.columns]
            if cgm_dfs:
                cgm_df = cgm_dfs[0]
                cgm_df = cgm_df[cgm_df['time_bin'] == time_bin].dropna(subset=['timestamp']).copy()
                if not cgm_df.empty:
                    cgm_df['timestamp'] = pd.to_datetime(cgm_df['timestamp'])
                    coverage_days = (
                        cgm_df.groupby('id')['timestamp']
                        .agg(lambda s: s.dt.normalize().nunique())  # count distinct calendar days with CGM readings
                    )
                    median_days = coverage_days.median()
                    print(f"Median CGM coverage days (distinct days with data) among CGM users: {median_days:.1f}")
                else:
                    print("Median CGM coverage days (time span): no CGM data in this time bin")
            print("\nAll counts are based on unique patient IDs.\n")

        return None
    
    def print_feature_availability_table(self) -> None:
        """
        Plot feature availability per time bin across studies and save availability heatmaps.

        Args:
            None

        Returns:
            None: Saves PNGs under './data/graphs/feature_analysis/availability/'.
        """

        def _num_key(idx: pd.Index) -> pd.Series:
            """
            Extract numeric parts from an index for natural sorting (e.g., 'Month 12').

            Args:
                idx: Pandas Index to parse.

            Returns:
                pd.Series: Numeric series used as sort key.
            """
            nums = idx.to_series().astype(str).str.extract(r'(\\d+)')[0]
            return pd.to_numeric(nums, errors='coerce')
        
        dict_dfs = self.datasets.copy()

        for df_name, df in dict_dfs.items():
            exclude = {'id', 'timestamp', 'timestamp_seconds', 'timestamp_type', 'time_bin', 'baseline_cpep_auc', 'dy', 'hba1c2', 'cpep_auc_preservation'}
            features = [c for c in df.columns if c not in exclude]
            dynamic_feature = "glucose mmol/l"
            static_features = [c for c in features if c != dynamic_feature]
            n_ids_by_bin = df.groupby('time_bin')['id'].nunique().compute()  # Series
            n = n_ids_by_bin['Baseline']

            stat_first = df.groupby(['time_bin','id'])[static_features].first().compute()
            static_missing_by_bin = stat_first.notna().groupby(level=0).sum().astype('int64').sort_index(axis=1)
            static_missing_by_bin = static_missing_by_bin.sort_index(key = _num_key)
            static_missing_by_bin = static_missing_by_bin.T
            static_missing_by_bin.insert(0, 'N', n)
            self.plot_availability(static_missing_by_bin, f'./data/graphs/feature_analysis/availability/{df_name}.png')

    def print_feature_histograms(self) -> None:
        """
        Generate and save per-feature distribution plots for each study.

        Iterates through `self.datasets` and calls `plot_feature_distributions` to write
        per-feature histograms/bar charts under './data/graphs/feature_analysis/histograms/{study_name}/'.

        Args:
            None

        Returns:
            None: Images are written to disk.

        Raises:
            AttributeError: If `self.datasets` is missing.
        """
        dict_dfs = self.datasets.copy()

        for study_name, df in dict_dfs.items():
            self.plot_feature_distributions(df, f'./data/graphs/feature_analysis/histograms/{study_name}/')

    def print_feature_isi_histograms(self) -> None:
        """
        Plot daily presence counts for selected features across all datasets.

        Args:
            None

        Returns:
            None: Saves per-dataset charts under './data/graphs/feature_analysis/isi_histograms'.
        """
        dict_dfs = self.datasets.copy()
        features = ['glucose mmol/l', 'hb_a1c', 'total_ins_dose', 'weight', 'cpep_fast', 'glucose_fast']

        for df_name, df in dict_dfs.items():
            counts_per_day = {}
            df = df.compute()
            first_day = df.groupby('id')['timestamp'].transform('min').dt.floor('D')
            df['day_from_start'] = (df['timestamp'].dt.floor('D') - first_day).dt.days + 1
            df = df[df['day_from_start'].notna() & (df['day_from_start'] >= 1)]
            df['day_from_start'] = df['day_from_start'].astype(int)

            for f in features:
                sub = df[['id', 'time_bin', 'day_from_start', f]].drop_duplicates(
                    subset=['id', 'time_bin', f], keep='first'
                )
                present = sub[sub[f].notna()]
                day_counts = present.groupby('day_from_start').size().rename('count')
                counts_per_day[f] = day_counts

            all_days = sorted(set().union(*[s.index if isinstance(s, pd.Series) else s.index
                              for s in counts_per_day.values() if len(s) > 0]))
            mat = pd.DataFrame(index=pd.Index(all_days, name='day_from_start'), columns=features).fillna(0).astype(int)
            for f, cnt in counts_per_day.items():
                mat.loc[cnt.index, f] = cnt.values

            self.plot_feature_charts(mat, features, df_name)
        return None

    def print_characteristics_table_baseline(self) -> pd.DataFrame:
        available_studies = sorted([s for s in ALL_STUDIES if s in self.datasets])
        if not available_studies:
            return pd.DataFrame()
        output_dir = Path('./data/csv_results/tables/characteristics_table')
        output_dir.mkdir(parents=True, exist_ok=True)

        def _to_pandas(df_like: pd.DataFrame) -> pd.DataFrame:
            return df_like.compute() if hasattr(df_like, 'compute') else df_like.copy()

        def _norm_treatment(val) -> str:
            if pd.isna(val):
                return ''
            return str(val).strip().lower()

        def _norm_sex(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float, np.integer, np.floating)):
                if val == 0:
                    return 'male'
                if val == 1:
                    return 'female'
            s = str(val).strip().lower()
            if s in {'m', 'male', 'man'}:
                return 'male'
            if s in {'f', 'female', 'woman'}:
                return 'female'
            return np.nan

        def _fmt_count(series: pd.Series, mask: pd.Series) -> str:
            denom = int(series.notna().sum())
            num = int(mask.sum())
            if denom == 0:
                return '0 (0.0%)'
            return f'{num} ({(num / denom) * 100:.1f}%)'

        def _fmt_mean_p5_p95(values: pd.Series) -> str:
            vals = pd.to_numeric(values, errors='coerce').dropna()
            if vals.empty:
                return 'NA'
            mean = vals.mean()
            p5 = vals.quantile(0.05)
            p95 = vals.quantile(0.95)
            return f'{mean:.2f} ({p5:.2f}, {p95:.2f})'

        def _get_days_from_diagnosis(base: pd.DataFrame) -> pd.Series:
            if 'dy' in base.columns:
                dy_vals = pd.to_numeric(base['dy'], errors='coerce')
                if dy_vals.notna().any():
                    return dy_vals

            if 'timestamp' in base.columns and 'diagnose_date' in base.columns:
                timestamp = pd.to_datetime(base['timestamp'], errors='coerce')
                diagnose_date = pd.to_datetime(base['diagnose_date'], errors='coerce')
                return (timestamp - diagnose_date).dt.days

            return pd.Series(np.nan, index=base.index, dtype='float64')

        def _first_baseline_per_id(df: pd.DataFrame) -> pd.DataFrame:
            work = df.copy()
            if 'time_bin' in work.columns:
                tb = work['time_bin'].astype(str).str.strip().str.lower()
                baseline = work[tb == 'baseline'].copy()
            else:
                baseline = pd.DataFrame(columns=work.columns)

            target = baseline if not baseline.empty else work.copy()

            sort_cols = []
            if 'timestamp' in target.columns:
                target['timestamp'] = pd.to_datetime(target['timestamp'], errors='coerce')
                sort_cols.append('timestamp')
            if sort_cols:
                target = target.sort_values(sort_cols, ascending=True, na_position='last')

            return target.drop_duplicates(subset=['id'], keep='first').copy()

        def _build_baseline_df(df: pd.DataFrame, study_name: str) -> pd.DataFrame:
            if 'id' not in df.columns:
                return pd.DataFrame()
            base = _first_baseline_per_id(df)
            if base.empty:
                return base

            treatment_col = 'treatment_arm' if 'treatment_arm' in base.columns else (
                'insulin_delivery' if 'insulin_delivery' in base.columns else None
            )
            if treatment_col is None:
                base['treatment_norm'] = ''
            else:
                base['treatment_norm'] = base[treatment_col].apply(_norm_treatment)
            base['study_name'] = study_name

            # Unified care categories from insulin delivery context:
            # standard care vs intensive care.
            if 'insulin_delivery' in base.columns:
                insulin_norm = base['insulin_delivery'].apply(_norm_treatment)
            else:
                insulin_norm = pd.Series('', index=base.index, dtype='object')
            treatment_norm = base['treatment_norm'].copy()
            care = pd.Series(np.nan, index=base.index, dtype='object')
            study_key = str(study_name).strip().lower()
            if study_key == 'cloud':
                care.loc[insulin_norm == 'mdi'] = 'standard care'
                care.loc[insulin_norm == 'hybrid closed-loop'] = 'intensive care'
                care.loc[insulin_norm == 'hybrid'] = 'intensive care'
            elif study_key == 'clvr':
                care.loc[insulin_norm == 'non-hcl'] = 'standard care'
                care.loc[insulin_norm == 'hcl'] = 'intensive care'
            elif study_key == 'diagnode':
                care.loc[insulin_norm == 'no insulin pump'] = 'standard care'
                care.loc[insulin_norm == 'insulin pump'] = 'intensive care'
            elif study_key == 'jaeb_t1d':
                care.loc[treatment_norm == 'control'] = 'standard care'
                care.loc[treatment_norm == 'active'] = 'intensive care'
            base['care_category'] = care

            base['sex_norm'] = base['sex'].apply(_norm_sex) if 'sex' in base.columns else np.nan
            base['age_num'] = pd.to_numeric(base['age'], errors='coerce') if 'age' in base.columns else np.nan
            base['weight_num'] = pd.to_numeric(base['weight'], errors='coerce') if 'weight' in base.columns else np.nan
            base['days_from_diagnosis_num'] = _get_days_from_diagnosis(base)

            cpep_candidates = ['cpep_auc', 'cpep_fast', 'cpep_0_min']
            cpep_col = next((c for c in cpep_candidates if c in base.columns), None)
            if cpep_col is None:
                base['cpep_value'] = np.nan
            else:
                base['cpep_value'] = pd.to_numeric(base[cpep_col], errors='coerce')

            # Compute baseline CGM core endpoints per participant using Taylor summary helper.
            cgm_endpoints = {'TIR', 'TITR', 'TBR_Lvl_1', 'TBR_Lvl_2', 'TAR_Lvl_1', 'TAR_Lvl_2'}
            taylor_base = self.get_taylor_patient_endpoints(
                df=df,
                metabolic_endpoints=cgm_endpoints,
                by_time_bin=True,
                include_cols=set()
            )
            if not taylor_base.empty and 'time_bin' in taylor_base.columns:
                tb = taylor_base['time_bin'].astype(str).str.strip().str.lower()
                taylor_base = taylor_base[tb == 'baseline'].copy()
            else:
                taylor_base = pd.DataFrame(columns=['id'] + sorted(cgm_endpoints))
            if not taylor_base.empty:
                taylor_base = taylor_base.drop_duplicates(subset=['id'], keep='first')
                merge_cols = ['id'] + [c for c in sorted(cgm_endpoints) if c in taylor_base.columns]
                base = base.merge(taylor_base[merge_cols], on='id', how='left')
            else:
                for c in sorted(cgm_endpoints):
                    base[c] = np.nan

            # Ensure numeric types for continuous baseline metrics used in the table.
            for c in ['hb_a1c', 'gmi', 'TIR', 'TITR', 'TBR_Lvl_1', 'TBR_Lvl_2', 'TAR_Lvl_1', 'TAR_Lvl_2']:
                if c in base.columns:
                    base[c] = pd.to_numeric(base[c], errors='coerce')
                else:
                    base[c] = np.nan

            return base

        per_study = {}
        for study in available_studies:
            per_study[study] = _build_baseline_df(_to_pandas(self.datasets[study]), study)

        all_studies_df = pd.concat(per_study.values(), ignore_index=True) if per_study else pd.DataFrame()
        table_sources = {'All studies': all_studies_df}
        for study in available_studies:
            table_sources[study] = per_study[study]

        def _col_label(name: str, df: pd.DataFrame) -> str:
            n = int(df['id'].nunique()) if (not df.empty and 'id' in df.columns) else 0
            return f'{name} (n={n})'

        row_order = [
            'Control',
            'Active',
            'Insulin Delivery: Standard Care',
            'Insulin Delivery: Intensive Care',
            'Days from Diagnosis: mean (5-95 percentile)',
            'Age: mean (5-95 percentile)',
            'Age (<18)',
            'Age (>=18)',
            'Sex (Male)',
            'Sex (Female)',
            'Cpeptide: mean (5-95 percentile)',
            'HbA1c: mean (5-95 percentile)',
            'GMI: mean (5-95 percentile)',
            'TIR: mean (5-95 percentile)',
            'TITR: mean (5-95 percentile)',
            'TBR Level 1: mean (5-95 percentile)',
            'TBR Level 2: mean (5-95 percentile)',
            'TAR Level 1: mean (5-95 percentile)',
            'TAR Level 2: mean (5-95 percentile)',
            'Weight for Adults: mean (5-95 percentile)',
            'Weight for Children: mean (5-95 percentile)',
        ]

        out = pd.DataFrame(index=row_order)
        for col_name, df in table_sources.items():
            col_label = _col_label(col_name, df)
            if df.empty:
                out[col_label] = ['NA'] * len(row_order)
                continue

            treatment = df['treatment_norm']
            care_category = df['care_category'] if 'care_category' in df.columns else pd.Series(np.nan, index=df.index)
            age = df['age_num']
            days_from_diagnosis = df['days_from_diagnosis_num'] if 'days_from_diagnosis_num' in df.columns else pd.Series(np.nan, index=df.index)
            sex = df['sex_norm']

            # Control/Active must come from treatment_arm mapping.
            control_mask = treatment.isin(TREATMENT_GROUP_1)
            active_mask = treatment.isin(TREATMENT_GROUP_2)

            std_care_mask = care_category == 'standard care'
            int_care_mask = care_category == 'intensive care'
            lt18_mask = age < 18
            ge18_mask = age >= 18
            male_mask = sex == 'male'
            female_mask = sex == 'female'
            adults_weight = df.loc[ge18_mask, 'weight_num']
            children_weight = df.loc[lt18_mask, 'weight_num']

            out.loc['Control', col_label] = _fmt_count(treatment, control_mask)
            out.loc['Active', col_label] = _fmt_count(treatment, active_mask)
            out.loc['Insulin Delivery: Standard Care', col_label] = _fmt_count(care_category, std_care_mask)
            out.loc['Insulin Delivery: Intensive Care', col_label] = _fmt_count(care_category, int_care_mask)
            out.loc['Days from Diagnosis: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(days_from_diagnosis)
            out.loc['Age: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(age)
            out.loc['Age (<18)', col_label] = _fmt_count(age, lt18_mask)
            out.loc['Age (>=18)', col_label] = _fmt_count(age, ge18_mask)
            out.loc['Sex (Male)', col_label] = _fmt_count(sex, male_mask)
            out.loc['Sex (Female)', col_label] = _fmt_count(sex, female_mask)
            out.loc['Cpeptide: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['cpep_value'])
            out.loc['HbA1c: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['hb_a1c'])
            out.loc['GMI: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['gmi'])
            out.loc['TIR: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TIR'])
            out.loc['TITR: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TITR'])
            out.loc['TBR Level 1: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TBR_Lvl_1'])
            out.loc['TBR Level 2: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TBR_Lvl_2'])
            out.loc['TAR Level 1: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TAR_Lvl_1'])
            out.loc['TAR Level 2: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(df['TAR_Lvl_2'])
            out.loc['Weight for Adults: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(adults_weight)
            out.loc['Weight for Children: mean (5-95 percentile)', col_label] = _fmt_mean_p5_p95(children_weight)

        ordered_cols = [_col_label('All studies', all_studies_df)] + [_col_label(study, per_study[study]) for study in available_studies]
        out = out.reindex(columns=ordered_cols)
        out.to_csv(output_dir / 'baseline_table.csv')
        return out
    
    def print_good_days(self) -> None:
        """
        Compute per-patient counts of \"good days\" (rows without missing values) and write summaries to CSV.

        Args:
            None

        Returns:
            None: Writes './good_days_{df_name}.csv' files and prints median stats.
        """
        dict_dfs = self.datasets.copy()

        for df_name, df in dict_dfs.items():
            df['day'] = df['timestamp'].dt.floor('D')
            df['row_missing_any'] = df.isna().any(axis=1)
            missing_counts = df.groupby(['id', 'day'])['row_missing_any'].sum()
            good_flags = (missing_counts == 0).reset_index().rename(columns={"row_missing_any": "is_good_day"})
            n_good_days = good_flags.groupby('id')['is_good_day'].sum()
            total_days = df.groupby(['id'])['day'].nunique()
            good_total_ratio = ((n_good_days / total_days) * 100)
            median_total_days = total_days.median().compute()
            median_good_days = n_good_days.median().compute()
            median_ratio = good_total_ratio.median().compute()

            df_good_days = {'total_days': total_days, 'good_days': n_good_days, 'ratio': good_total_ratio}
            df_good_days = pd.DataFrame(df_good_days)
            df_good_days.to_csv(f'./good_days_{df_name}.csv', index=False)

            print('\tGood Days:\n',df_name.capitalize(), median_total_days ,median_good_days, median_ratio,'\n')

        return None
 
    def print_insulin_inventory(self) -> None:
        dict_dfs = self.datasets.copy()
        all_dict = {k: v.compute() for k, v in dict_dfs.items() if k in ALL_STUDIES}
        output_dir = Path('./data/csv_results/tables/insulin')
        output_dir.mkdir(parents=True, exist_ok=True)

        all_ins_cols = sorted(
            {col for df in all_dict.values() for col in df.columns if 'ins_dose' in col.lower()}
        )
        both_row_label = 'basal_and_bolus_ins_dose'
        summary_rows = all_ins_cols + [both_row_label]

        print('Insulin Column Non-Null Coverage by Time Bin (% with subjects shown as n=with_value/total):')
        for df_name in sorted(all_dict.keys()):
            df = all_dict[df_name]
            ins_cols = [col for col in df.columns if 'ins_dose' in col.lower()]
            basal_col = next((col for col in ins_cols if col.lower() == 'basal_ins_dose'), None)
            bolus_col = next((col for col in ins_cols if col.lower() == 'bolus_ins_dose'), None)

            def _time_bin_sort_key(value):
                value_str = str(value).strip().lower()
                if value_str == 'baseline':
                    return (-1, 0.0)
                match = re.search(r'(-?\d+(\.\d+)?)', str(value))
                if match:
                    return (0, float(match.group(1)))
                if 'screen' in value_str:
                    return (-2, 0.0)
                return (1, float('inf'))

            time_bins = sorted(df['time_bin'].dropna().unique(), key=_time_bin_sort_key)
            summary_table = pd.DataFrame(index=summary_rows, columns=time_bins, dtype='object')

            for time_bin in time_bins:
                df_bin = df[df['time_bin'] == time_bin]
                total_subjects = df_bin['id'].nunique()
                grouped = df_bin.groupby('id', dropna=False)

                for col in ins_cols:
                    subjects_with_value = grouped[col].apply(lambda s: s.notna().any()).sum()
                    pct_notna = (subjects_with_value / total_subjects * 100) if total_subjects > 0 else np.nan
                    if pd.isna(pct_notna):
                        summary_table.loc[col, time_bin] = 'NA'
                    else:
                        summary_table.loc[col, time_bin] = f'{pct_notna:.2f}% (n={subjects_with_value}/{total_subjects})'

                if basal_col and bolus_col:
                    subjects_with_both = grouped.apply(
                        lambda group: group[basal_col].notna().any() and group[bolus_col].notna().any()
                    ).sum()
                    pct_both = (subjects_with_both / total_subjects * 100) if total_subjects > 0 else np.nan
                    summary_table.loc[both_row_label, time_bin] = (
                        'NA' if pd.isna(pct_both)
                        else f'{pct_both:.2f}% (n={subjects_with_both}/{total_subjects})'
                    )
                else:
                    summary_table.loc[both_row_label, time_bin] = 'NA'

            print(f'\n{df_name.upper()}:')
            print(summary_table.to_string())
            summary_table.to_csv(output_dir / f'{df_name}_insulin_inventory.csv')

        return None

    def print_metadata(self) -> None:
        """
        Generate and save per-time-bin metadata presence tables for each study.

        For each selected time bin, builds a table showing whether each column has at least
        one non-null value for that study/time bin; writes CSV files under './data/metadata/'.

        Args:
            None

        Returns:
            None: Writes CSV files named 'metadata_{tb}m.csv' to disk.

        Raises:
            AttributeError: If `self.datasets` is missing.
            KeyError: If required columns ('time_bin', 'id') are absent in any dataset.
            OSError: If the output directory does not exist or is not writable.
        """
        REQUIRED_COLS = [
            'N', 'diagnose_date', 'glucose mmol/l','age','treatment_arm','insulin_delivery','sex','ethnicity','race',
            'height','weight', 'hb_a1c', 'hb_a1c_cap', 'hb_a1c_local','total_ins_dose',
            'basal_ins_dose','bolus_ins_dose','cpep_pre10_min','cpep_0_min','cpep_15_min',
            'cpep_30_min','cpep_60_min','cpep_90_min','cpep_120_min','cpep_auc',
            'glucose_pre10_min','glucose_0_min','glucose_15_min','glucose_30_min',
            'glucose_60_min','glucose_90_min','glucose_120_min','beta2_score',]
        dict_df = self.datasets.copy()
        TIME_BINS = [0, 3, 6, 9, 12, 24]

        for tb in TIME_BINS:
            metadata_table = []
            counts = []

            for study_name, df_study in dict_df.items():
                df_study_tb = df_study[df_study['time_bin'] == f'Month {tb}'].compute()
                
                if df_study_tb.empty:

                    cols = list(df_study.columns)  # original columns
                    row = pd.DataFrame([{c: pd.NA for c in cols}])
                    row.rename(columns={'id': 'study'}, inplace=True)
                    row['study'] = study_name.upper()
                    metadata_table.append(row)
                    counts.append((0))
                    continue

                row = df_study_tb.head(1)
                row.rename(columns={'id': 'study'}, inplace=True)
                row['study'] = study_name.upper()
                metadata_table.append(row)
                counts.append((df_study_tb['id'].nunique()))

            metadata_table = pd.concat(metadata_table, ignore_index=True)
            metadata_table = (
                metadata_table.drop(columns=["study"])
                .notna()
                .groupby(metadata_table["study"])
                .any()
            )

            metadata_table = metadata_table.replace({True: "✅", False: "❌"})
            metadata_table['N'] = counts
            metadata_table = metadata_table[REQUIRED_COLS]
            metadata_table = metadata_table.reset_index()

            # write file named by the bin
            metadata_table.to_csv(f'./data/csv_results/metadata/metadata_{tb}m.csv', index=False)

    def print_summaries(self) -> None:
        """
        Print one-line summaries for each available dataset in `self.datasets_summary`.

        Iterates over datasets and prints each summary metric and its value.

        Args:
            None

        Returns:
            None: Outputs text to stdout.

        Raises:
            AttributeError: If `self.datasets_summary` is missing.
        """
        for df_name, df_summary in self.datasets_summary.items():
            print(f'\n{df_name} Summary:')
            for summary_feature in df_summary.columns:
                print(f'\t{summary_feature}: {df_summary[summary_feature].values[0]}')

    def print_valid_durations(self) -> None:
        """
        Print data inventory statistics per study: average actual duration, valid duration, and CGM density.

        For each dataset in `self.datasets`, computes per-patient actual recording duration,
        valid coverage (≤30-minute gaps), and average CGM points/day; then prints study-level averages.

        Args:
            None

        Returns:
            None: Outputs text to stdout.

        Raises:
            AttributeError: If `self.datasets` is missing.
            KeyError: If required columns ('timestamp', 'id') are absent in any dataset.
        """
        dict_dfs = self.datasets.copy()
        
        for df_name, df in dict_dfs.items():
            df = df.compute()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_grouped = df.groupby(['id'])
            actual_durations = []
            valid_durations = []
            densities = []
            
            for pt_id, pt_df in df_grouped:
                timestamps = pt_df['timestamp'].sort_values()
                time_diffs = timestamps.diff().dropna()
                valid_diffs = time_diffs[time_diffs <= pd.Timedelta(minutes=30)]
                valid_covered_seconds = valid_diffs.dt.total_seconds().sum()
                valid_covered_days = valid_covered_seconds/ (60 * 60 * 24)
                start = pt_df['timestamp'].min()
                end = pt_df['timestamp'].max()
                actual_duration_days = (end - start).total_seconds() / (60 * 60 * 24)

                if actual_duration_days > 0:
                    cgm_points = pt_df.shape[0]
                    density = cgm_points/valid_covered_days
                    valid_durations.append(valid_covered_days)
                    actual_durations.append(actual_duration_days)
                    densities.append(density)
            
            if actual_durations:
                avg_actual_duration = sum(actual_durations) / len(actual_durations)
                avg_valid_durations = sum(valid_durations) / len(valid_durations)
                avg_density = sum(densities) / len(densities)
                print(f"📘 Study: {df_name}")
                print(f"   ├─ Avg actual duration per patient: {avg_actual_duration:.2f} days")
                print(f"   ├─ Avg valid duration per patient: {avg_valid_durations:.2f} days")
                print(f"   └─ Avg data density: {avg_density:.1f} CGM points/day\n")
            else:
                print(f"📘 Study: {df_name} — No valid patients\n")
    
    def statistical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-patient CGM statistics and hypoglycemia metrics.

        For each patient ID, calculates average glucose, total/normalized Length of Line (LoL),
        counts and percentages for hypoglycemia levels, and returns a consolidated table.

        Args:
            df (pd.DataFrame): Input dataset containing at least 'id', 'timestamp',
                and 'glucose mmol/l' columns.

        Returns:
            pd.DataFrame: Patient-level table with columns:
                ['id', 'glucose_avg', 'total_length', 'normalized_length',
                 'hypoglycemia_level_1', 'hypoglycemia_level_2', 'cl_sig_hypo',
                 'prc_hypoglycemia_level_1', 'prc_hypoglycemia_level_2', 'prc_cl_sig_hypo'].

        Raises:
            KeyError: If required columns are missing from `df`.
        """
        df = df.compute()
        grouped_patients = df.groupby('id')
    
        patients_analysis = {}
        patients_analysis['id'] = []
        patients_analysis['glucose_avg'] = []
        patients_analysis['gvp_min'] = []
        patients_analysis['gvp_hr'] = []
        patients_analysis['LoL_min'] = []
        patients_analysis['LoL_hr'] = []
        patients_analysis['hypoglycemia_level_1'] = []
        patients_analysis['hypoglycemia_level_2'] = []
        patients_analysis['cl_sig_hypo'] = []
        patients_analysis['prc_hypoglycemia_level_1'] = []
        patients_analysis['prc_hypoglycemia_level_2'] = []
        patients_analysis['prc_cl_sig_hypo'] = []

        for patient_id, patient_df in grouped_patients:
            num_cgm_data = patient_df.shape[0]
            gvp_min, gvp_hr, LoL_min, LoL_hr = self.length_data(patient_df)
            level_1, level_2, cl_sig_events = self.get_hypoglycemia_rates(patient_df)
            glucose_avg = patient_df['glucose mmol/l'].mean()
            patients_analysis['id'].append(patient_id)
            patients_analysis['glucose_avg'].append(glucose_avg)
            patients_analysis['gvp_min'].append(gvp_min)
            patients_analysis['gvp_hr'].append(gvp_hr)
            patients_analysis['LoL_min'].append(LoL_min)
            patients_analysis['LoL_hr'].append(LoL_hr)
            patients_analysis['hypoglycemia_level_1'].append(level_1)
            patients_analysis['hypoglycemia_level_2'].append(level_2)
            patients_analysis['cl_sig_hypo'].append(cl_sig_events)
            patients_analysis['prc_hypoglycemia_level_1'].append(level_1/num_cgm_data * 100) 
            patients_analysis['prc_hypoglycemia_level_2'].append(level_2/num_cgm_data * 100)
            patients_analysis['prc_cl_sig_hypo'].append(cl_sig_events/num_cgm_data * 100)

        df_patients_analysis = pd.DataFrame(patients_analysis)
        df_patients_analysis = df_patients_analysis.loc[:, ~df_patients_analysis.columns.duplicated()]

        return df_patients_analysis

    """ Testing Methods """
    # Classical methods.
    def run_shapiro_wilk_test(self, data: pd.Series) -> tuple[float, float]:
        """
        Perform the Shapiro-Wilk normality test on a numeric series.

        Args:
            data (pd.Series): Input data to test.

        Returns:
            tuple[float, float]:
                The Shapiro-Wilk test statistic and p-value.

        Raises:
            ValueError: If fewer than 3 non-missing observations are available,
                or if all values are identical.
        """
        data = pd.to_numeric(data, errors='coerce').dropna()
        if len(data) < 3:
            raise ValueError(
                "Shapiro-Wilk test requires at least 3 non-missing observations."
            )
        if data.nunique() < 2:
            raise ValueError(
                "Shapiro-Wilk test cannot be run when all values are identical."
            )

        stat, p_value = shapiro(data)
        return float(stat), float(p_value)

    def run_kruskal_test(self, groups: dict[str, pd.Series]) -> tuple[float, float]:
        """
        Perform the Kruskal-Wallis test across multiple groups.

        Args:
            groups (dict[str, pd.Series]): Mapping of group labels to numeric data.

        Returns:
            tuple[float, float]:
                The Kruskal-Wallis statistic and p-value.

        Raises:
            ValueError: If fewer than 2 non-empty groups are available, or if all
                values across groups are identical.
        """
        valid_groups = []
        for values in groups.values():
            values = pd.to_numeric(values, errors='coerce').dropna()
            if not values.empty:
                valid_groups.append(values)

        if len(valid_groups) < 2:
            raise ValueError(
                "Kruskal-Wallis test requires at least 2 non-empty groups."
            )
        combined_values = pd.concat(valid_groups, ignore_index=True)
        if combined_values.nunique() < 2:
            raise ValueError(
                "Kruskal-Wallis test cannot be run when all values across groups are identical."
            )

        stat, p_value = kruskal(*valid_groups)
        return float(stat), float(p_value)

    def run_oneway_anova_test(self, groups: dict[str, pd.Series]) -> tuple[float, float]:
        """
        Perform a one-way ANOVA across multiple groups.

        Args:
            groups (dict[str, pd.Series]): Mapping of group labels to numeric data.

        Returns:
            tuple[float, float]:
                The ANOVA F-statistic and p-value.

        Raises:
            ValueError: If fewer than 2 non-empty groups are available, if any
                valid group has fewer than 2 observations, or if all values across
                groups are identical.
        """
        valid_groups = []
        for values in groups.values():
            values = pd.to_numeric(values, errors='coerce').dropna()
            if not values.empty:
                valid_groups.append(values)

        if len(valid_groups) < 2:
            raise ValueError(
                "One-way ANOVA requires at least 2 non-empty groups."
            )
        if any(len(group) < 2 for group in valid_groups):
            raise ValueError(
                "One-way ANOVA requires at least 2 observations in each non-empty group."
            )

        combined_values = pd.concat(valid_groups, ignore_index=True)
        if combined_values.nunique() < 2:
            raise ValueError(
                "One-way ANOVA cannot be run when all values across groups are identical."
            )

        stat, p_value = f_oneway(*valid_groups)
        if pd.isna(stat) or pd.isna(p_value):
            raise ValueError(
                "One-way ANOVA produced an invalid result. Please verify the input groups."
            )

        return float(stat), float(p_value)

    def run_dunn_posthoc(self, groups: dict[str, pd.Series]) -> list[dict[str, float | str | int]]:
        """
        Perform Dunn-style pairwise post hoc comparisons across multiple groups.

        Uses rank-based pairwise z-tests with Holm correction across the valid
        pairwise comparisons.

        Args:
            groups (dict[str, pd.Series]): Mapping of group labels to numeric data.

        Returns:
            list[dict[str, float | str | int]]:
                One result dictionary per pairwise comparison.

        Raises:
            ValueError: If fewer than 2 valid non-empty groups are available, or
                if all values across groups are identical.
        """
        non_empty_groups = {
            group: pd.to_numeric(values, errors='coerce').dropna()
            for group, values in groups.items()
            if not pd.to_numeric(values, errors='coerce').dropna().empty
        }
        if len(non_empty_groups) < 2:
            raise ValueError(
                "Dunn post hoc test requires at least 2 non-empty groups."
            )

        group_order = list(non_empty_groups.keys())
        combined_values = np.concatenate([non_empty_groups[group].to_numpy() for group in group_order])
        combined_groups = np.concatenate([
            np.repeat(group, len(non_empty_groups[group]))
            for group in group_order
        ])
        n_total = len(combined_values)
        if n_total < 2:
            raise ValueError(
                "Dunn post hoc test requires at least 2 observations across groups."
            )
        if pd.Series(combined_values).nunique() < 2:
            raise ValueError(
                "Dunn post hoc test cannot be run when all values across groups are identical."
            )

        ranks = rankdata(combined_values, method='average')
        _, tie_counts = np.unique(combined_values, return_counts=True)
        tie_correction = 1 - ((tie_counts ** 3 - tie_counts).sum() / (n_total ** 3 - n_total))
        if tie_correction <= 0:
            raise ValueError(
                "Dunn post hoc test produced a non-positive tie correction, "
                "indicating an invalid or numerically unstable tie structure."
            )

        results = []
        raw_p_values = []
        for group_a, group_b in combinations(group_order, 2):
            mask_a = combined_groups == group_a
            mask_b = combined_groups == group_b
            n_a = int(mask_a.sum())
            n_b = int(mask_b.sum())
            if n_a == 0 or n_b == 0:
                continue
            mean_rank_a = float(ranks[mask_a].mean())
            mean_rank_b = float(ranks[mask_b].mean())
            se = np.sqrt((n_total * (n_total + 1) / 12) * tie_correction * ((1 / n_a) + (1 / n_b)))
            if se == 0 or np.isnan(se):
                z_stat = np.nan
                p_value = np.nan
            else:
                z_stat = float((mean_rank_a - mean_rank_b) / se)
                p_value = float(2 * norm.sf(abs(z_stat)))

            results.append({
                'group_1': group_a,
                'group_2': group_b,
                'n_1': n_a,
                'n_2': n_b,
                'mean_rank_1': mean_rank_a,
                'mean_rank_2': mean_rank_b,
                'z_stat': z_stat,
                'p_value': p_value
            })
            raw_p_values.append(p_value)

        valid_mask = [pd.notna(p) for p in raw_p_values]
        adjusted = [np.nan] * len(raw_p_values)
        if any(valid_mask):
            valid_p_values = [p for p in raw_p_values if pd.notna(p)]
            _, corrected, _, _ = multipletests(valid_p_values, method='holm')
            corrected_iter = iter(corrected)
            for idx, is_valid in enumerate(valid_mask):
                if is_valid:
                    adjusted[idx] = float(next(corrected_iter))

        for result, adjusted_p in zip(results, adjusted):
            result['p_value_holm'] = adjusted_p

        return results

    def run_two_group_test(
        self,
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        feature_name: str,
        is_independent: bool = True,
        pair_cols: list[str] | None = None,
        alpha: float = 0.05
    ) -> dict[str, object]:
        """
        Compare a feature between two groups using an adaptive classical test.

        For independent groups, applies Shapiro-Wilk to each group and uses Welch's
        t-test if both are normal, otherwise Mann-Whitney U.

        For paired/dependent groups, aligns paired values, applies Shapiro-Wilk to the
        pairwise differences, and uses a paired t-test if the differences are normal,
        otherwise Wilcoxon signed-rank.

        Args:
            df_1 (pd.DataFrame): First group DataFrame.
            df_2 (pd.DataFrame): Second group DataFrame.
            feature_name (str): Feature column to compare.
            is_independent (bool): Whether the two groups are independent.
            pair_cols (list[str] | None): Columns used to align paired observations
                when `is_independent` is False. If None, defaults to `['id']` when
                available in both DataFrames.
            alpha (float): Significance level threshold. Defaults to 0.05.

        Returns:
            dict[str, object]: Test metadata and results.

        Raises:
            KeyError: If `feature_name` is missing in either DataFrame.
        """
        if is_independent:
            values_df1 = pd.to_numeric(df_1[feature_name], errors='coerce').dropna()
            values_df2 = pd.to_numeric(df_2[feature_name], errors='coerce').dropna()

            stat_df1, p_shapiro_1 = self.run_shapiro_wilk_test(values_df1)
            stat_df2, p_shapiro_2 = self.run_shapiro_wilk_test(values_df2)
            normal_df1 = pd.notna(p_shapiro_1) and p_shapiro_1 > alpha
            normal_df2 = pd.notna(p_shapiro_2) and p_shapiro_2 > alpha

            if normal_df1 and normal_df2:
                stat, p_value = ttest_ind(values_df1, values_df2, equal_var=False)
                test_used = "T-test (Welch's correction)"
            else:
                stat, p_value = mannwhitneyu(values_df1, values_df2, alternative='two-sided')
                test_used = "Mann-Whitney U test"

            return {
                "feature": feature_name,
                "test_used": test_used,
                "is_independent": is_independent,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "normality_df1": normal_df1,
                "normality_df2": normal_df2,
                "shapiro_stat_df1": stat_df1,
                "shapiro_p_value_df1": p_shapiro_1,
                "shapiro_stat_df2": stat_df2,
                "shapiro_p_value_df2": p_shapiro_2,
                "n_df1": len(values_df1),
                "n_df2": len(values_df2),
            }
        else:
            effective_pair_cols = pair_cols
            if effective_pair_cols is None:
                effective_pair_cols = ['id'] if 'id' in df_1.columns and 'id' in df_2.columns else []
            else:
                effective_pair_cols = [
                    col for col in effective_pair_cols
                    if col in df_1.columns and col in df_2.columns
                ]

            if not effective_pair_cols:
                raise ValueError(
                    "Dependent tests require valid pairing columns. "
                    "Provide `pair_cols` explicitly or include a shared `id` column in both DataFrames."
                )

            paired_df1 = df_1[effective_pair_cols + [feature_name]].copy()
            paired_df2 = df_2[effective_pair_cols + [feature_name]].copy()
            paired_df = paired_df1.merge(
                paired_df2,
                on=effective_pair_cols,
                how='inner',
                suffixes=('_df1', '_df2')
            )
            values_df1 = pd.to_numeric(paired_df[f'{feature_name}_df1'], errors='coerce')
            values_df2 = pd.to_numeric(paired_df[f'{feature_name}_df2'], errors='coerce')

            paired_values = pd.DataFrame({
                'value_df1': values_df1,
                'value_df2': values_df2
            }).dropna()

            if paired_values.empty:
                raise ValueError(
                    f"No valid paired observations were found for feature '{feature_name}' "
                    f"using pairing columns {effective_pair_cols}."
                )

            diffs = paired_values['value_df1'] - paired_values['value_df2']
            if (diffs == 0).all():
                raise ValueError(
                    f"All paired differences are zero for feature '{feature_name}'. "
                    "Please verify that the paired inputs and pairing columns are correct."
                )

            stat_diff, p_diff = self.run_shapiro_wilk_test(diffs)
            normal_differences = pd.notna(p_diff) and p_diff > alpha

            if normal_differences:
                stat, p_value = ttest_rel(
                    paired_values['value_df1'],
                    paired_values['value_df2']
                )
                test_used = "Paired t-test"
            else:
                non_zero_diffs = diffs[diffs != 0]
                stat, p_value = wilcoxon(non_zero_diffs, zero_method='wilcox', alternative='two-sided')
                test_used = "Wilcoxon signed-rank test"

            return {
                "feature": feature_name,
                "test_used": test_used,
                "is_independent": is_independent,
                "statistic": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "significant": bool(pd.notna(p_value) and p_value < alpha),
                "normality_differences": normal_differences,
                "shapiro_stat_differences": stat_diff,
                "shapiro_p_value_differences": p_diff,
                "n_pairs": int(len(paired_values))
            }

    # Correlation methods.
    def run_spearman_test(self, x: pd.Series, y: pd.Series) -> tuple[float, float]:
        """
        Perform a Spearman rank correlation test between two variables.

        Args:
            x (pd.Series): First variable.
            y (pd.Series): Second variable.

        Returns:
            tuple[float, float]:
                Spearman correlation coefficient and p-value.

        Raises:
            ValueError: If fewer than 2 paired non-missing observations are available,
                or if either variable is constant after pairing.
        """
        paired_values = pd.DataFrame({
            'x': pd.to_numeric(x, errors='coerce'),
            'y': pd.to_numeric(y, errors='coerce')
        }).dropna()

        if len(paired_values) < 2:
            raise ValueError(
                "Spearman test requires at least 2 paired non-missing observations."
            )
        if paired_values['x'].nunique() < 2:
            raise ValueError(
                "Spearman test cannot be run when the first variable is constant."
            )
        if paired_values['y'].nunique() < 2:
            raise ValueError(
                "Spearman test cannot be run when the second variable is constant."
            )

        rho, p_value = spearmanr(paired_values['x'], paired_values['y'])
        if pd.isna(rho) or pd.isna(p_value):
            raise ValueError(
                "Spearman test produced an invalid result. Please verify the input variables."
            )

        return float(rho), float(p_value)

    def run_linear_regression_test(self, x: pd.Series, y: pd.Series) -> dict[str, float]:
        """
        Perform simple linear regression between two variables.

        Args:
            x (pd.Series): Predictor variable.
            y (pd.Series): Outcome variable.

        Returns:
            dict[str, float]:
                Regression summary with slope, intercept, correlation coefficient,
                p-value, and standard error.

        Raises:
            ValueError: If fewer than 2 paired non-missing observations are available,
                or if the predictor variable is constant.
        """
        paired_values = pd.DataFrame({
            'x': pd.to_numeric(x, errors='coerce'),
            'y': pd.to_numeric(y, errors='coerce')
        }).dropna()

        if len(paired_values) < 2:
            raise ValueError(
                "Linear regression requires at least 2 paired non-missing observations."
            )
        if paired_values['x'].nunique() < 2:
            raise ValueError(
                "Linear regression cannot be run when the predictor variable is constant."
            )

        slope, intercept, r_value, p_value, std_err = linregress(
            paired_values['x'],
            paired_values['y']
        )
        if any(pd.isna(v) for v in [slope, intercept, r_value, p_value, std_err]):
            raise ValueError(
                "Linear regression produced an invalid result. Please verify the input variables."
            )

        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'n': int(len(paired_values))
        }

    # Linear model methods.
    def run_linear_mixed_model(
        self,
        df: pd.DataFrame,
        formula: str,
        group_col: str,
        reml: bool = False
    ):
        """
        Fit a linear mixed-effects model.

        Args:
            df (pd.DataFrame): Input modeling dataset.
            formula (str): Statsmodels mixedlm formula.
            group_col (str): Column used for random-effect grouping.
            reml (bool): Whether to fit with REML. Defaults to False.

        Returns:
            statsmodels.regression.mixed_linear_model.MixedLMResults:
                Fitted mixed-effects model result object.

        Raises:
            ValueError: If the input data are empty, required columns are missing,
                the grouping column has fewer than 2 unique values, or the model fit
                produces an invalid result.
        """
        if df.empty:
            raise ValueError("Linear mixed model requires a non-empty input DataFrame.")
        if group_col not in df.columns:
            raise ValueError(
                f"Linear mixed model requires grouping column '{group_col}' in the input DataFrame."
            )
        if df[group_col].dropna().nunique() < 2:
            raise ValueError(
                f"Linear mixed model requires at least 2 unique non-missing values in grouping column '{group_col}'."
            )

        try:
            model = smf.mixedlm(
                formula,
                df,
                groups=df[group_col]
            )
            result = model.fit(reml=reml)
        except Exception as exc:
            raise ValueError(
                f"Linear mixed model fit failed for formula '{formula}': {exc}"
            ) from exc

        if pd.isna(result.llf):
            raise ValueError(
                f"Linear mixed model produced an invalid log-likelihood for formula '{formula}'."
            )

        return result

    def run_likelihood_ratio_test(self, full_result, reduced_result) -> dict[str, float]:
        """
        Compare two nested fitted models using a likelihood ratio test.

        Args:
            full_result: Fitted result for the more complex model.
            reduced_result: Fitted result for the nested simpler model.

        Returns:
            dict[str, float]:
                Likelihood ratio statistic, degrees-of-freedom difference, and p-value.

        Raises:
            ValueError: If either model is missing required attributes, if the degrees
                of freedom difference is not positive, or if the fitted log-likelihoods
                are invalid.
        """
        required_attrs = ['llf', 'fe_params']
        for attr in required_attrs:
            if not hasattr(full_result, attr):
                raise ValueError(f"Full model result is missing required attribute '{attr}'.")
            if not hasattr(reduced_result, attr):
                raise ValueError(f"Reduced model result is missing required attribute '{attr}'.")

        if pd.isna(full_result.llf) or pd.isna(reduced_result.llf):
            raise ValueError("Likelihood ratio test requires valid log-likelihood values from both models.")

        df_diff = len(full_result.fe_params) - len(reduced_result.fe_params)
        if df_diff <= 0:
            raise ValueError(
                "Likelihood ratio test requires the full model to have more fixed-effect parameters than the reduced model."
            )

        lrt_stat = 2 * (full_result.llf - reduced_result.llf)
        p_value = chi2.sf(lrt_stat, df=df_diff)
        if pd.isna(lrt_stat) or pd.isna(p_value):
            raise ValueError("Likelihood ratio test produced an invalid result.")

        return {
            'lrt_stat': float(lrt_stat),
            'df_diff': int(df_diff),
            'p_value': float(p_value)
        }

    def run_cpep_strata_time_lmm(
        self,
        df: pd.DataFrame,
        endpoints: set[str],
        *,
        full_formula: str,
        reduced_formula: str,
        trend_formula: str,
        group_col: str,
        study_col: str,
        time_col: str,
        stratum_col: str,
        strict: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run time-adjusted LMMs for C-peptide strata with strict validation.
        """
        if df.empty:
            raise ValueError("Time-adjusted LMM requires a non-empty DataFrame.")
        required_cols = {group_col, study_col, time_col, stratum_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for time-adjusted LMM: {sorted(missing)}")

        adjusted_means = []
        lrt_results = []
        trend_results = []

        for endpoint in endpoints:
            if endpoint not in df.columns:
                if strict:
                    raise ValueError(f"Missing endpoint column '{endpoint}' for time-adjusted LMM.")
                continue

            df_endpoint = df[[study_col, group_col, time_col, endpoint, stratum_col]].dropna(
                subset=[study_col, group_col, time_col, endpoint, stratum_col]
            )
            if df_endpoint.empty:
                if strict:
                    raise ValueError(f"No data available for endpoint '{endpoint}' after dropping missing values.")
                continue

            df_endpoint = df_endpoint.copy()
            for col in [study_col, stratum_col]:
                if pd.api.types.is_categorical_dtype(df_endpoint[col]):
                    df_endpoint[col] = df_endpoint[col].cat.remove_unused_categories()
                else:
                    df_endpoint[col] = df_endpoint[col].astype('category')

            n_participants = df_endpoint[group_col].nunique()
            n_studies = df_endpoint[study_col].nunique(dropna=True)
            n_time_points = df_endpoint[time_col].nunique(dropna=True)
            n_strata = df_endpoint[stratum_col].nunique(dropna=True)
            if n_participants < 2 or n_studies < 2 or n_time_points < 2 or n_strata < 2:
                msg = (
                    f"Insufficient support for endpoint '{endpoint}': "
                    f"participants={n_participants}, studies={n_studies}, "
                    f"time_points={n_time_points}, strata={n_strata}"
                )
                if strict:
                    raise ValueError(msg)
                continue

            try:
                full_model = smf.mixedlm(full_formula, df_endpoint, groups=df_endpoint[group_col])
                if np.linalg.matrix_rank(full_model.exog) < full_model.exog.shape[1]:
                    raise ValueError(f"Full model design matrix is rank deficient for endpoint '{endpoint}'.")
                full_result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=full_formula,
                    group_col=group_col,
                    reml=False
                )

                reduced_model = smf.mixedlm(reduced_formula, df_endpoint, groups=df_endpoint[group_col])
                if np.linalg.matrix_rank(reduced_model.exog) < reduced_model.exog.shape[1]:
                    raise ValueError(f"Reduced model design matrix is rank deficient for endpoint '{endpoint}'.")
                reduced_result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=reduced_formula,
                    group_col=group_col,
                    reml=False
                )

                lrt_result = self.run_likelihood_ratio_test(full_result, reduced_result)
                lrt_results.append({
                    'endpoint': endpoint,
                    'lrt_stat': lrt_result['lrt_stat'],
                    'df_diff': lrt_result['df_diff'],
                    'p_value': lrt_result['p_value']
                })

                df_endpoint['cpep_stratum_num'] = df_endpoint[stratum_col].cat.codes + 1
                trend_model = smf.mixedlm(trend_formula, df_endpoint, groups=df_endpoint[group_col])
                if np.linalg.matrix_rank(trend_model.exog) < trend_model.exog.shape[1]:
                    raise ValueError(f"Trend model design matrix is rank deficient for endpoint '{endpoint}'.")
                trend_result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=trend_formula,
                    group_col=group_col,
                    reml=False
                )
                trend_results.append({
                    'endpoint': endpoint,
                    'trend_coef': trend_result.params.get('cpep_stratum_num', np.nan),
                    'trend_p_value': trend_result.pvalues.get('cpep_stratum_num', np.nan)
                })

                design_info = full_result.model.data.design_info
                fe_params = full_result.fe_params
                cov_params = full_result.cov_params()
                if isinstance(cov_params, pd.DataFrame):
                    cov_fe = cov_params.iloc[:len(fe_params), :len(fe_params)]
                else:
                    cov_fe = cov_params[:len(fe_params), :len(fe_params)]

                study_time_weights = (
                    df_endpoint.groupby([study_col, time_col], observed=True)
                    .size()
                    .rename('n')
                    .reset_index()
                )
                if study_time_weights.empty or study_time_weights['n'].sum() <= 0:
                    raise ValueError(f"Invalid study/time weights for endpoint '{endpoint}'.")
                study_time_weights['weight'] = study_time_weights['n'] / study_time_weights['n'].sum()

                cpep_levels = df_endpoint[stratum_col].cat.categories
                for level in cpep_levels:
                    if level not in df_endpoint[stratum_col].values:
                        continue
                    design_df = pd.DataFrame({
                        stratum_col: pd.Categorical(
                            [level] * len(study_time_weights),
                            categories=cpep_levels,
                            ordered=True
                        ),
                        study_col: pd.Categorical(
                            study_time_weights[study_col].tolist(),
                            categories=df_endpoint[study_col].cat.categories
                            if hasattr(df_endpoint[study_col], 'cat') else sorted(df_endpoint[study_col].unique())
                        ),
                        time_col: study_time_weights[time_col].tolist()
                    })
                    exog = patsy.build_design_matrices(
                        [design_info],
                        design_df,
                        return_type='dataframe'
                    )[0]
                    weights = study_time_weights['weight'].to_numpy()
                    xbar = (exog.to_numpy() * weights[:, None]).sum(axis=0)
                    mean = float(xbar @ fe_params.to_numpy())
                    se = float(np.sqrt(xbar @ cov_fe.to_numpy() @ xbar.T))
                    if not np.isfinite(mean) or not np.isfinite(se):
                        raise ValueError(f"Adjusted mean/SE invalid for endpoint '{endpoint}', stratum '{level}'.")
                    ci_lower = mean - 1.96 * se
                    ci_upper = mean + 1.96 * se
                    adjusted_means.append({
                        'endpoint': endpoint,
                        'cpep_stratum': level,
                        'adjusted_mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })
            except Exception as exc:
                if strict:
                    raise
                print(f"Time-adjusted LMM failed for {endpoint}: {exc}")

        adjusted_means_df = pd.DataFrame(adjusted_means)
        lrt_df = pd.DataFrame(lrt_results)
        trend_df = pd.DataFrame(trend_results)
        return adjusted_means_df, lrt_df, trend_df

    """ Hypothesis Tests """
    def a1c_vs_gmi_treatment_arms(self) -> None:
        """
        Baseline boxplots for the full cohort: fasting glucose, GMI vs HbA1c,
        and BETA2 vs BETA3 (no arm split).

        Generates three panels:
            - Fasting glucose (glucose_0_min)
            - Side-by-side GMI and HbA1c
            - Side-by-side BETA2 and BETA3

        Saves the figure to `./data/graphs/feature_analysis/a1c_vs_gmi_treatment_arms.png`.
        """
        allowed_time_bins = ['Baseline']
        dict_dfs = self.datasets.copy()

        all_dict = {k: v.compute() for k, v in dict_dfs.items() if k in ALL_STUDIES}
        all_dict = {k: df[df['time_bin'].isin(allowed_time_bins)] for k, df in all_dict.items()}
        df_t1d = pd.concat(all_dict.values(), ignore_index=True)

        features = {
            'Fasting Glucose': ['glucose_0_min'],
            'GMI vs HbA1c': ['gmi', 'hb_a1c'],
            'BETA2 vs BETA3': ['beta2_score', 'beta3_score']
        }
        paired_map = {
            'GMI vs HbA1c': ('hb_a1c', 'gmi'),
            'BETA2 vs BETA3': ('beta2_score', 'beta3_score')
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        for ax, (title, cols) in zip(axes, features.items()):
            cols_present = [c for c in cols if c in df_t1d.columns]
            if not cols_present:
                ax.set_visible(False)
                continue

            df_plot = df_t1d[['id', 'time_bin'] + cols_present].drop_duplicates()
            df_long = df_plot.melt(id_vars=['id', 'time_bin'], value_vars=cols_present,
                                   var_name='metric', value_name='value')
            df_long = df_long.dropna(subset=['value'])

            sns.boxplot(
                data=df_long,
                x='metric',
                y='value',
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend_.remove() if ax.get_legend() else None

            # Annotate paired test results on applicable panels.
            if title in paired_map:
                a, b = paired_map[title]
                paired = df_t1d[[a, b]].dropna()
                if len(paired) >= 2:
                    diffs = paired[a] - paired[b]
                    med_diff = np.median(diffs)
                    # Wilcoxon is more robust for skew/outliers; fall back if all diffs are zero.
                    diffs_nonzero = diffs[diffs != 0]
                    if len(diffs_nonzero) > 0:
                        try:
                            w_stat, w_p = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
                            annotation = f"Wilcoxon p={w_p:.3g}, Δmedian={med_diff:.2f}, n={len(paired)}"
                        except ValueError:
                            annotation = f"Wilcoxon not computed (ties), Δmedian={med_diff:.2f}, n={len(paired)}"
                    else:
                        annotation = f"No difference (all ties), n={len(paired)}"
                else:
                    annotation = "Insufficient paired data"
                ax.text(0.5, 0.9, annotation, ha='center', va='center', transform=ax.transAxes)

        os.makedirs('./data/graphs/feature_analysis', exist_ok=True)
        plt.savefig('./data/graphs/feature_analysis/a1c_vs_gmi_treatment_arms.png', dpi=300)
        plt.close(fig)

        # Paired comparisons (Wilcoxon preferred for skew/outliers).
        paired_tests = [
            ('hb_a1c', 'gmi'),
            ('beta2_score', 'beta3_score')
        ]

        for a, b in paired_tests:
            if a not in df_t1d.columns or b not in df_t1d.columns:
                print(f"Skipping paired test for {a} vs {b}: missing columns.")
                continue
            paired = df_t1d[[a, b]].dropna()
            if paired.empty or len(paired) < 2:
                print(f"Skipping paired test for {a} vs {b}: insufficient paired data.")
                continue

            diffs = paired[a] - paired[b]
            med_diff = np.median(diffs)
            diffs_nonzero = diffs[diffs != 0]
            if len(diffs_nonzero) > 0:
                try:
                    w_stat, w_p = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
                    print(f"Wilcoxon {a} vs {b}: W={w_stat:.3f}, p={w_p:.3g}, n={len(paired)}, Δmedian={med_diff:.3f}")
                except ValueError as e:
                    print(f"Wilcoxon {a} vs {b} not computed (ties): {e}")
            else:
                print(f"{a} vs {b}: all paired differences are zero, n={len(paired)}")

    def best_cgm_pct_wear_and_days(self) -> tuple[int, int] | None:
        """
        Compare CGM core endpoints across wear-day windows using 14 days / 70% as reference.
        Returns the best (wear_prct, wear_days) pair across all studies.
        """
        wear_days_options = [10, 7, 5, 3]
        wear_prct_options = [70]
        ref_days = 14
        ref_prct = 70
        all_wear_days = sorted(set(wear_days_options + [ref_days]), reverse=True)
        all_wear_prct = sorted(set(wear_prct_options + [ref_prct]))

        dict_dfs = self.datasets.copy()
        # studies = [name for name in ALL_STUDIES if name in dict_dfs]
        studies = ['cloud','clvr','diagnode','jaeb_t1d','hupa_ucm']

        endpoints_records = []
        endpoints_by_key = {}

        use_time_bin = True
        merge_keys = ['id', 'time_bin'] if use_time_bin else ['id']

        for study_name in studies:
            df = dict_dfs[study_name]
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            if df.empty or not {'id', 'timestamp', 'time_bin'}.issubset(df.columns):
                print(f"Skipping {study_name}: missing CGM columns.")
                continue

            df = df.copy()
            if 'study' not in df.columns:
                df['study'] = study_name

            for wear_prct in all_wear_prct:
                for wear_days in all_wear_days:
                    window_df = self.get_cgm_windows(df, wear_days, wear_prct)
                    if window_df.empty:
                        continue

                    self.get_gmi(window_df)
                    endpoints = self.get_taylor_patient_endpoints(
                        window_df,
                        set(CGM_CORE_ENDPOINTS) | {'gmi'},
                        by_time_bin=use_time_bin,
                        include_cols={'study', 'gmi'}
                    )
                    endpoints['wear_days'] = wear_days
                    endpoints['wear_prct'] = wear_prct
                    endpoints['study'] = study_name

                    endpoints_records.append(endpoints)
                    endpoints_by_key[(study_name, wear_days, wear_prct)] = endpoints

        if endpoints_records:
            endpoints_df = pd.concat(endpoints_records, ignore_index=True)
            output_path = Path("./data/cgm_wear_window_endpoints.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            endpoints_df.to_csv(output_path, index=False)
            print(f"Saved CGM wear window endpoints to {output_path}")
        else:
            print("No CGM wear window endpoints to report.")
            return

        overlap_records = []
        for study_name in studies:
            ref_df = endpoints_by_key.get((study_name, ref_days, ref_prct))
            if ref_df is None or ref_df.empty:
                continue
            ref_keys = set(tuple(row) for row in ref_df[merge_keys].itertuples(index=False, name=None))
            for wear_prct in wear_prct_options:
                for wear_days in wear_days_options:
                    comp_df = endpoints_by_key.get((study_name, wear_days, wear_prct))
                    if comp_df is None or comp_df.empty:
                        continue
                    comp_keys = set(tuple(row) for row in comp_df[merge_keys].itertuples(index=False, name=None))
                    overlap_records.append({
                        'study': study_name,
                        'wear_days': wear_days,
                        'wear_prct': wear_prct,
                        'ref_wear_days': ref_days,
                        'ref_wear_prct': ref_prct,
                        'n_ref': len(ref_keys),
                        'n_comp': len(comp_keys),
                        'n_overlap': len(ref_keys & comp_keys)
                    })

        if overlap_records:
            overlap_df = pd.DataFrame(overlap_records)
            overlap_path = Path("./data/cgm_wear_window_overlap.csv")
            overlap_path.parent.mkdir(parents=True, exist_ok=True)
            overlap_df.to_csv(overlap_path, index=False)
            print(f"Saved CGM wear window overlap counts to {overlap_path}")

        priority_metrics = set(CGM_CORE_ENDPOINTS) | {'gmi'}

        stats_records = []
        tost_records = []
        agreement_records = []
        tost_margins = {
            'gmi': 0.2,
            'TIR': 2.0,
            'TITR': 2.0,
            'TAR_Lvl_1': 2.0,
            'median_glucose': 0.5,
            'GVP': 2.0,
        }
        tost_alpha = 0.05
        ba_output_dir = Path("./data/graphs/cgm_wear_window_ba")
        ba_output_dir.mkdir(parents=True, exist_ok=True)
        mard_eps = 1e-6

        def _icc_2_1(ratings: np.ndarray) -> float:
            """
            ICC(2,1): two-way random effects, absolute agreement, single measurement.
            ratings shape: (n_subjects, k_raters)
            """
            n, k = ratings.shape
            if n < 2 or k < 2:
                return np.nan
            mean_row = ratings.mean(axis=1, keepdims=True)
            mean_col = ratings.mean(axis=0, keepdims=True)
            grand_mean = ratings.mean()
            ssr = k * np.sum((mean_row - grand_mean) ** 2)
            ssc = n * np.sum((mean_col - grand_mean) ** 2)
            sse = np.sum((ratings - mean_row - mean_col + grand_mean) ** 2)
            msr = ssr / (n - 1) if n > 1 else np.nan
            msc = ssc / (k - 1) if k > 1 else np.nan
            mse = sse / ((n - 1) * (k - 1)) if n > 1 and k > 1 else np.nan
            if np.isnan(msr) or np.isnan(msc) or np.isnan(mse):
                return np.nan
            denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
            if denom == 0:
                return np.nan
            return (msr - mse) / denom

        for study_name in studies:
            ref_df = endpoints_by_key.get((study_name, ref_days, ref_prct))
            if ref_df is None or ref_df.empty:
                continue

            for wear_days in wear_days_options:
                comp_df = endpoints_by_key.get((study_name, wear_days, ref_prct))
                if comp_df is None or comp_df.empty:
                    continue

                for metric in sorted(set(CGM_CORE_ENDPOINTS) | {'gmi'}):
                    if metric not in ref_df.columns or metric not in comp_df.columns:
                        continue
                    merged = ref_df[merge_keys + [metric]].merge(
                        comp_df[merge_keys + [metric]],
                        on=merge_keys,
                        suffixes=('_ref', '_comp')
                    ).dropna()

                    n_pairs = len(merged)
                    if n_pairs == 0:
                        continue

                    diffs = merged[f"{metric}_comp"] - merged[f"{metric}_ref"]
                    mean_diff = float(diffs.mean()) if n_pairs > 0 else np.nan
                    w_stat = np.nan
                    p_val = np.nan
                    test = "insufficient"

                    if n_pairs >= 2:
                        diffs_nonzero = diffs[diffs != 0]
                        if len(diffs_nonzero) > 0:
                            try:
                                w_stat, p_val = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
                                test = "wilcoxon"
                            except ValueError:
                                test = "wilcoxon_failed"
                        else:
                            test = "all_ties"

                    stats_records.append({
                        'study': study_name,
                        'metric': metric,
                        'wear_days': wear_days,
                        'wear_prct': wear_prct,
                        'ref_wear_days': ref_days,
                        'ref_wear_prct': ref_prct,
                        'n_pairs': n_pairs,
                        'p_value': p_val,
                        'test': test
                    })

                    if n_pairs >= 2:
                        ref_vals = merged[f"{metric}_ref"].to_numpy()
                        comp_vals = merged[f"{metric}_comp"].to_numpy()
                        abs_diffs = np.abs(comp_vals - ref_vals)
                        mean_abs_diff = float(abs_diffs.mean())
                        median_abs_diff = float(np.median(abs_diffs))
                        mard = float(np.mean(abs_diffs / (np.abs(ref_vals) + mard_eps)))
                        ratings = np.column_stack([ref_vals, comp_vals])
                        icc_2_1 = float(_icc_2_1(ratings))

                        means = (ref_vals + comp_vals) / 2.0
                        bias = float(diffs.mean())
                        sd_diff = float(np.std(diffs, ddof=1))
                        loa_lower = bias - 1.96 * sd_diff
                        loa_upper = bias + 1.96 * sd_diff

                        agreement_records.append({
                            'study': study_name,
                            'metric': metric,
                            'wear_days': wear_days,
                            'wear_prct': wear_prct,
                            'ref_wear_days': ref_days,
                            'ref_wear_prct': ref_prct,
                            'n_pairs': n_pairs,
                            'mean_abs_diff': mean_abs_diff,
                            'median_abs_diff': median_abs_diff,
                            'mard': mard,
                            'icc_2_1': icc_2_1,
                            'ba_bias': bias,
                            'ba_sd_diff': sd_diff,
                            'ba_loa_lower': loa_lower,
                            'ba_loa_upper': loa_upper
                        })

                        plot_path = (
                            ba_output_dir
                            / study_name
                            / metric
                            / f"wear{wear_days}_prct{wear_prct}_ref{ref_prct}.png"
                        )
                        plot_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.figure(figsize=(6, 4))
                        plt.scatter(means, diffs, s=12, alpha=0.6)
                        plt.axhline(bias, color='red', linestyle='--', linewidth=1)
                        plt.axhline(loa_lower, color='gray', linestyle='--', linewidth=1)
                        plt.axhline(loa_upper, color='gray', linestyle='--', linewidth=1)
                        plt.title(f"{study_name} {metric} BA (wear {wear_days} vs ref {ref_prct}%)")
                        plt.xlabel("(Comp + Ref)/2")
                        plt.ylabel("Comp - Ref")
                        plt.tight_layout()
                        plt.savefig(plot_path, dpi=200)
                        plt.close()

                    if metric in tost_margins and n_pairs >= 2:
                        margin = tost_margins[metric]
                        std_diff = float(diffs.std(ddof=1))
                        tost_p_lower = np.nan
                        tost_p_upper = np.nan
                        equivalent = False

                        if std_diff == 0.0:
                            if abs(mean_diff) <= margin:
                                tost_p_lower = 0.0
                                tost_p_upper = 0.0
                                equivalent = True
                            else:
                                tost_p_lower = 1.0
                                tost_p_upper = 1.0
                        else:
                            se_diff = std_diff / np.sqrt(n_pairs)
                            t_lower = (mean_diff + margin) / se_diff
                            t_upper = (mean_diff - margin) / se_diff
                            df = n_pairs - 1
                            tost_p_lower = 1.0 - t.cdf(t_lower, df)
                            tost_p_upper = t.cdf(t_upper, df)
                            equivalent = (tost_p_lower < tost_alpha) and (tost_p_upper < tost_alpha)

                        tost_records.append({
                            'study': study_name,
                            'metric': metric,
                            'wear_days': wear_days,
                            'wear_prct': wear_prct,
                            'ref_wear_days': ref_days,
                            'ref_wear_prct': ref_prct,
                            'n_pairs': n_pairs,
                            'mean_diff': mean_diff,
                            'margin': margin,
                            'tost_p_lower': tost_p_lower,
                            'tost_p_upper': tost_p_upper,
                            'equivalent': equivalent,
                            'alpha': tost_alpha
                        })

        if stats_records:
            stats_df = pd.DataFrame(stats_records)
            stats_path = Path("./data/cgm_wear_window_stats.csv")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(stats_path, index=False)
            print(f"Saved CGM wear window stats to {stats_path}")
        else:
            print("No CGM wear window stats to report.")
            return None

        if tost_records:
            tost_df = pd.DataFrame(tost_records)
            tost_path = Path("./data/cgm_wear_window_tost.csv")
            tost_path.parent.mkdir(parents=True, exist_ok=True)
            tost_df.to_csv(tost_path, index=False)
            print(f"Saved CGM wear window TOST results to {tost_path}")
        else:
            print("No CGM wear window TOST results to report.")

        if agreement_records:
            agreement_df = pd.DataFrame(agreement_records)
            agreement_path = Path("./data/cgm_wear_window_agreement.csv")
            agreement_path.parent.mkdir(parents=True, exist_ok=True)
            agreement_df.to_csv(agreement_path, index=False)
            print(f"Saved CGM wear window agreement metrics to {agreement_path}")
        else:
            print("No CGM wear window agreement metrics to report.")

        if agreement_records:
            agreement_df = pd.DataFrame(agreement_records)
            agreement_grouped = agreement_df.groupby(
                ['wear_days', 'wear_prct', 'ref_wear_days', 'ref_wear_prct', 'metric'],
                as_index=False
            ).agg(
                n_rows=('mean_abs_diff', 'size'),
                mean_abs_diff=('mean_abs_diff', 'mean'),
                median_abs_diff=('median_abs_diff', 'mean'),
                mean_mard=('mard', 'mean'),
                mean_icc=('icc_2_1', 'mean'),
                ba_bias=('ba_bias', 'mean'),
                ba_sd_diff=('ba_sd_diff', 'mean'),
                ba_loa_lower=('ba_loa_lower', 'mean'),
                ba_loa_upper=('ba_loa_upper', 'mean')
            )

            if tost_records:
                tost_df = pd.DataFrame(tost_records)
                tost_df['equivalent'] = tost_df['equivalent'].astype(str).str.lower() == 'true'
                tost_grouped = tost_df.groupby(
                    ['wear_days', 'wear_prct', 'ref_wear_days', 'ref_wear_prct', 'metric'],
                    as_index=False
                ).agg(
                    tost_equiv_rate=('equivalent', 'mean'),
                    tost_p_lower_mean=('tost_p_lower', 'mean'),
                    tost_p_upper_mean=('tost_p_upper', 'mean'),
                    tost_alpha=('alpha', 'max')
                )
                agreement_grouped = agreement_grouped.merge(
                    tost_grouped,
                    on=['wear_days', 'wear_prct', 'ref_wear_days', 'ref_wear_prct', 'metric'],
                    how='left'
                )

            summary_dir = Path("./data/csv_results/cgm_wear_window_metric_summaries")
            summary_dir.mkdir(parents=True, exist_ok=True)
            for (wear_days, wear_prct, ref_days, ref_prct), sub in agreement_grouped.groupby(
                ['wear_days', 'wear_prct', 'ref_wear_days', 'ref_wear_prct']
            ):
                out_path = summary_dir / f"wear{wear_days}_prct{wear_prct}_ref{ref_prct}.csv"
                sub.drop(columns=['wear_days', 'wear_prct', 'ref_wear_days', 'ref_wear_prct']).to_csv(
                    out_path,
                    index=False
                )
            print(f"Saved per-wear-day metric summaries to {summary_dir}")

        valid_stats = stats_df.dropna(subset=['p_value'])
        if valid_stats.empty:
            print("No valid p-values to determine best wear window.")
            return None

        valid_stats = valid_stats[valid_stats['metric'].isin(priority_metrics)]
        if valid_stats.empty:
            print("No valid p-values for available metrics.")
            return None

        summary = (
            valid_stats
            .groupby(['wear_prct', 'wear_days'], as_index=False)
            .agg(
                total_tests=('p_value', 'size'),
                nonsig_rate=('p_value', lambda s: float((s >= 0.05).mean())),
                mean_p=('p_value', 'mean')
            )
        )
        summary = summary.sort_values(
            by=['nonsig_rate', 'mean_p', 'wear_days'],
            ascending=[False, False, False]
        )
        best_row = summary.iloc[0]
        best_prct = int(best_row['wear_prct'])
        best_days = int(best_row['wear_days'])
        print("Wear window summary across all studies (non-sig rate, mean p):")
        for _, row in summary.iterrows():
            print(
                f"- {int(row['wear_prct'])}% for {int(row['wear_days'])} days: "
                f"non-sig rate={row['nonsig_rate']:.3f}, mean p={row['mean_p']:.3f}, "
                f"tests={int(row['total_tests'])}"
            )
        print(
            f"Best wear window across all studies: "
            f"{best_prct}% for {best_days} days "
            f"(non-sig rate={best_row['nonsig_rate']:.3f}, mean p={best_row['mean_p']:.3f})."
        )
        return best_prct, best_days

    def ht_CGM_daytime_type(self) -> None:
        """
        Run hypothesis tests on CGM endpoints over time bins, split by daytime vs. nocturnal periods.

        Computes per-time-bin endpoints for day/night subsets, generates grouped time-bin plots,
        and saves Spearman correlation heatmaps for each subset.

        Args:
            None

        Returns:
            None: Plots and heatmaps are saved to disk.

        Raises:
            KeyError: If 'jaeb_healthy' is missing from `self.datasets`.
            ValueError: If required columns ('time_bin', 'timestamp_type', 'id') are missing.
        """
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy'].copy().compute()
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)
        jaeb_healthy_core_endpoints['time_bin'] = 'Healthy'
        jaeb_healthy_core_endpoints['id'] = 'Healthy_' + jaeb_healthy_core_endpoints['id'].astype('str')

        df_daytime_grouped_time_bins = {}
        df_nocturnal_grouped_time_bins = {}

        for df_name, df_t1d in dict_df_t1d.items():
            df_t1d = df_t1d.compute()
            df_t1d = df_t1d[~(df_t1d['time_bin'] == 'N/A')]
            df_t1d['timestamp_type'] = df_t1d['timestamp_type'].str.lower()
            df_daytime_time_bins = df_t1d[df_t1d['timestamp_type'] == 'daytime']
            df_nocturnal_time_bins = df_t1d[df_t1d['timestamp_type'] == 'nocturnal']

            df_daytime_time_bins = df_daytime_time_bins.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_daytime_time_bins = df_daytime_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_nocturnal_time_bins = df_nocturnal_time_bins.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_nocturnal_time_bins = df_nocturnal_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)

            df_daytime_grouped_time_bins[df_name] = df_daytime_time_bins
            df_nocturnal_grouped_time_bins[df_name] = df_nocturnal_time_bins

        daytime_group_1_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_daytime_grouped_time_bins['hupa_ucm'],
                                     'defend': df_daytime_grouped_time_bins['defend'], 'jaeb_t1d': df_daytime_grouped_time_bins['jaeb_t1d'],
                                     'gskalb': df_daytime_grouped_time_bins['gskalb']}
        
        daytime_group_2_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_daytime_grouped_time_bins['hupa_ucm'],
                                     'diagnode': df_daytime_grouped_time_bins['diagnode'], 'cloud': df_daytime_grouped_time_bins['cloud'],
                                     'clvr': df_daytime_grouped_time_bins['clvr']}
        
        nocturnal_group_1_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_nocturnal_grouped_time_bins['hupa_ucm'],
                                       'defend': df_nocturnal_grouped_time_bins['defend'], 'jaeb_t1d': df_nocturnal_grouped_time_bins['jaeb_t1d'],
                                       'gskalb': df_nocturnal_grouped_time_bins['gskalb']}
        
        nocturnal_group_2_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_nocturnal_grouped_time_bins['hupa_ucm'],
                                       'diagnode': df_nocturnal_grouped_time_bins['diagnode'], 'cloud': df_nocturnal_grouped_time_bins['cloud'],
                                       'clvr': df_nocturnal_grouped_time_bins['clvr']}
        
        all_cgm_feats = [col for col in df_daytime_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time', 'level_1']]
        for core_endpoints_feature in all_cgm_feats:
            self.plot_grouped_time_bins(daytime_group_1_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/daytime_djg.png')
            self.plot_grouped_time_bins(daytime_group_2_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/daytime_dcc.png')          
            self.plot_grouped_time_bins(nocturnal_group_1_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/nocturnal_djg.png')
            self.plot_grouped_time_bins(nocturnal_group_2_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/nocturnal_dcc.png')          

        self.plot_spearman_heatmap_from_dfs(df_daytime_grouped_time_bins, 'Month_num', all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/daytime_heatmap_timebins.png')
        self.plot_spearman_heatmap_from_dfs(df_nocturnal_grouped_time_bins, 'Month_num', all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/nocturnal_heatmap_timebins.png')

    def ht_CGM_general(self) -> None:
        """
        Run hypothesis tests on CGM core endpoints comparing 'jaeb_healthy' to each T1D dataset.

        Computes CGM endpoints for healthy and T1D studies, performs tests per endpoint,
        and saves comparison plots via `plot_hypothesis_test_general`.

        Args:
            None

        Returns:
            None: Comparison plots are saved to disk.

        Raises:
            KeyError: If 'jaeb_healthy' is missing from `self.datasets`.
        """
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy']
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)

        for df_name, df_t1d in dict_df_t1d.items():
            t1d_core_endpoints = self.get_cgm_core_endpoints_general(df_t1d)
            for core_endpoints_feature in jaeb_healthy_core_endpoints.columns:
                hypo_test_results = self.run_two_group_test(jaeb_healthy_core_endpoints, t1d_core_endpoints, core_endpoints_feature)
                self.plot_hypothesis_test_general(jaeb_healthy_core_endpoints, t1d_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/graphs/feature_analysis/general/{core_endpoints_feature}/jh_vs_{df_name}.png')

    def ht_CGM_time_bins(self) -> None:
        """
        Run hypothesis tests on CGM core endpoints over time bins across studies.

        Computes per-time-bin CGM endpoints for healthy and T1D datasets, filters bins
        with sufficient sample size, generates grouped time-bin plots, and saves a
        Spearman correlation heatmap across datasets.

        Args:
            None

        Returns:
            None: Plots and heatmaps are saved to disk.

        Raises:
            KeyError: If 'jaeb_healthy' is missing from `self.datasets`.
            ValueError: If required columns ('time_bin', 'id') are missing in datasets.
        """
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy'].copy().compute()
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)
        jaeb_healthy_core_endpoints['time_bin'] = 'Healthy'
        jaeb_healthy_core_endpoints['id'] = 'Healthy_' + jaeb_healthy_core_endpoints['id'].astype('str')
        df_grouped_time_bins = {}

        # for df_version, percentile_method in percentile_methods.items():
        for df_name, df_t1d_original in dict_df_t1d.items():
            df_t1d_original = df_t1d_original.compute()
            df_t1d_original = df_t1d_original[~(df_t1d_original['time_bin'] == 'N/A')]
            # df_t1d = percentile_method(df_t1d_original)
            df_t1d = df_t1d_original.copy()
            df_time_bins = df_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_time_bins = df_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            if df_time_bins['id'].nunique() > 10:
                df_grouped_time_bins[df_name] = df_time_bins
            else:
                continue  # skip this dataset in current percentile version

        group_1_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_grouped_time_bins['hupa_ucm'],
                             'defend': df_grouped_time_bins['defend'], 'jaeb_t1d': df_grouped_time_bins['jaeb_t1d'],
                             'gskalb': df_grouped_time_bins['gskalb']}
        group_2_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_grouped_time_bins['hupa_ucm'],
                             'diagnode': df_grouped_time_bins['diagnode'], 'cloud': df_grouped_time_bins['cloud'],
                             'clvr': df_grouped_time_bins['clvr']}
        
        all_cgm_feats = [col for col in df_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time', 'level_1']]
        for core_endpoints_feature in all_cgm_feats:
            self.plot_grouped_time_bins(group_1_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/djg.png')
            self.plot_grouped_time_bins(group_2_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/dcc.png') 
        
        self.plot_spearman_heatmap_from_dfs(df_grouped_time_bins, 'Month_num', all_cgm_feats, path= f'./data/graphs/feature_analysis/time_bins/heatmaps/heatmap_timebins.png')  
  
    def ht_CGM_treatment_arms(self) -> None:
        """
        Run hypothesis tests on CGM endpoints over time bins, stratified by treatment arms.

        Splits each T1D dataset by predefined treatment groups, computes per-time-bin endpoints,
        generates grouped time-bin plots for each arm, and saves Spearman heatmaps.

        Args:
            None

        Returns:
            None: Plots and heatmaps are saved to disk.

        Raises:
            KeyError: If 'jaeb_healthy' is missing or if treatment group constants are undefined.
            ValueError: If required columns ('time_bin', 'treatment_arm', 'id') are missing.
        """
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy'].copy().compute()
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)
        jaeb_healthy_core_endpoints['time_bin'] = 'Healthy'
        jaeb_healthy_core_endpoints['id'] = 'Healthy_' + jaeb_healthy_core_endpoints['id'].astype('str')

        df_t1_grouped_time_bins = {}
        df_t2_grouped_time_bins = {}

        for df_name, df_t1d in dict_df_t1d.items():
            df_t1d = df_t1d.compute()
            df_t1d = df_t1d[~(df_t1d['time_bin'] == 'N/A')]
            df_t1d['treatment_arm'] = df_t1d['treatment_arm'].str.lower()
            df_t1_time_bins = df_t1d[df_t1d['treatment_arm'].isin(TREATMENT_GROUP_1)]
            df_t2_time_bins = df_t1d[df_t1d['treatment_arm'].isin(TREATMENT_GROUP_2)]

            df_t1_time_bins = df_t1_time_bins.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_t1_time_bins = df_t1_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_t2_time_bins = df_t2_time_bins.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_t2_time_bins = df_t2_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)

            df_t1_grouped_time_bins[df_name] = df_t1_time_bins
            df_t2_grouped_time_bins[df_name] = df_t2_time_bins

        t1_group_1_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints,'hupa_ucm': df_t1_grouped_time_bins['hupa_ucm'],
                                'defend': df_t1_grouped_time_bins['defend'], 'jaeb_t1d': df_t1_grouped_time_bins['jaeb_t1d'],
                                'gskalb': df_t1_grouped_time_bins['gskalb']}
        
        t1_group_2_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_t1_grouped_time_bins['hupa_ucm'],
                                'diagnode': df_t1_grouped_time_bins['diagnode'], 'cloud': df_t1_grouped_time_bins['cloud'],
                                'clvr': df_t1_grouped_time_bins['clvr']}
        
        t2_group_1_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints, 'hupa_ucm': df_t2_grouped_time_bins['hupa_ucm'],
                                'defend': df_t2_grouped_time_bins['defend'], 'jaeb_t1d': df_t2_grouped_time_bins['jaeb_t1d'],
                                'gskalb': df_t2_grouped_time_bins['gskalb']}
        
        t2_group_2_time_bins = {'jaeb_healthy' : jaeb_healthy_core_endpoints,  'hupa_ucm': df_t2_grouped_time_bins['hupa_ucm'],
                                'diagnode': df_t2_grouped_time_bins['diagnode'], 'cloud': df_t2_grouped_time_bins['cloud'],
                                'clvr': df_t2_grouped_time_bins['clvr']}
        
        all_cgm_feats = [col for col in df_t1_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time', 'level_1']]
        for core_endpoints_feature in all_cgm_feats:
            self.plot_grouped_time_bins(t1_group_1_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/t1_djg.png')
            self.plot_grouped_time_bins(t1_group_2_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/t1_dcc.png')          
            self.plot_grouped_time_bins(t2_group_1_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/t2_djg.png')
            self.plot_grouped_time_bins(t2_group_2_time_bins, core_endpoints_feature, path=f'./data/graphs/feature_analysis/time_bins/{core_endpoints_feature}/t2_dcc.png')          

        self.plot_spearman_heatmap_from_dfs(df_t1_grouped_time_bins, 'Month_num', all_cgm_feats, path= f'./data/graphs/feature_analysis/time_bins/heatmaps/t1_heatmap_timebins.png')  
        self.plot_spearman_heatmap_from_dfs(df_t2_grouped_time_bins, 'Month_num', all_cgm_feats, path= f'./data/graphs/feature_analysis/time_bins/heatmaps/t2_heatmap_timebins.png')  
   
    def ht_clinical_features(self) -> None:
        """
        Evaluate correlations between clinical features and CGM core endpoints across time bins.

        For each dataset, computes per-time-bin endpoints, merges with clinical metrics
        (C-peptide AUC, BETA2), and saves Spearman correlation heatmaps.

        Args:
            None

        Returns:
            None: Heatmap images are saved to disk.

        Raises:
            ValueError: If required columns are missing from datasets.
        """
        dict_df_test = self.datasets.copy()
        dict_dfs_for_plot = {}

        for df_name, df_t1d in dict_df_test.items():
            df_t1d = df_t1d[~(df_t1d['time_bin'] == 'N/A')].compute()
            df_time_bins = df_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general).reset_index()
            df_time_bins = df_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_time_bins.drop(columns = ['level_1'], inplace=True)

            df_cpep_auc = df_t1d[['id', 'time_bin', 'cpep_auc', 'beta2_score']]
            df_cpep_auc = df_cpep_auc.drop_duplicates(subset=['id', 'time_bin'])
            df_cpep_auc = df_cpep_auc.merge(df_time_bins, on=['id', 'time_bin'], how='inner')

            dict_dfs_for_plot[df_name] = df_cpep_auc

        clinical_feats = ['cpep_auc', 'beta2_score']
        all_cgm_feats = [col for col in df_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time']]

        for clinical_feat in clinical_feats:
            self.plot_spearman_heatmap_from_dfs(dict_dfs_for_plot, clinical_feature=clinical_feat, cgm_features=all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/heatmap_{clinical_feat}.png')

    def ht_clinical_features_daytime_type(self) -> None:
        """
        Evaluate correlations between clinical features and CGM endpoints by daytime vs. nocturnal periods.

        Splits each dataset into daytime and nocturnal subsets, merges clinical metrics with
        per-time-bin endpoints, and saves separate Spearman heatmaps for each subset.

        Args:
            None

        Returns:
            None: Heatmap images are saved to disk.

        Raises:
            ValueError: If required columns ('timestamp_type', 'time_bin', 'id') are missing.
        """
        dict_df_test = self.datasets.copy()
        dict_daytime_dfs_for_plot = {}
        dict_nocturnal_dfs_for_plot = {}

        for df_name, df_t1d in dict_df_test.items():
            df_t1d = df_t1d[~(df_t1d['time_bin'] == 'N/A')].compute()
            df_t1d['timestamp_type'] = df_t1d['timestamp_type'].str.lower()
            df_daytime_t1d = df_t1d[df_t1d['timestamp_type'] == 'daytime']
            df_nocturnal_t1d = df_t1d[df_t1d['timestamp_type'] == 'nocturnal']

            df_daytime_time_bins = df_daytime_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_daytime_time_bins = df_daytime_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_daytime_time_bins.drop(columns = ['level_1'], inplace=True)
            df_nocturnal_time_bins = df_nocturnal_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_nocturnal_time_bins = df_nocturnal_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_nocturnal_time_bins.drop(columns = ['level_1'], inplace=True)

            df_daytime_cpep_auc = df_daytime_t1d[['id', 'time_bin', 'cpep_auc', 'beta2_score']]
            df_daytime_cpep_auc = df_daytime_cpep_auc.drop_duplicates(subset=['id', 'time_bin'])
            df_daytime_cpep_auc = df_daytime_cpep_auc.merge(df_daytime_time_bins, on=['id', 'time_bin'], how='inner')
            df_nocturnal_cpep_auc = df_nocturnal_t1d[['id', 'time_bin', 'cpep_auc', 'beta2_score']]
            df_nocturnal_cpep_auc = df_nocturnal_cpep_auc.drop_duplicates(subset=['id', 'time_bin'])
            df_nocturnal_cpep_auc = df_nocturnal_cpep_auc.merge(df_nocturnal_time_bins, on=['id', 'time_bin'], how='inner')

            dict_daytime_dfs_for_plot[df_name] = df_daytime_cpep_auc
            dict_nocturnal_dfs_for_plot[df_name] = df_nocturnal_cpep_auc

        clinical_feats = ['cpep_auc', 'beta2_score']
        all_cgm_feats = [col for col in df_daytime_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time']]

        for clinical_feat in clinical_feats:
            self.plot_spearman_heatmap_from_dfs(dict_daytime_dfs_for_plot, clinical_feature=clinical_feat, cgm_features=all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/daytime_heatmap_{clinical_feat}.png')
            self.plot_spearman_heatmap_from_dfs(dict_nocturnal_dfs_for_plot, clinical_feature=clinical_feat, cgm_features=all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/nocturnal_heatmap_{clinical_feat}.png')

    def ht_clinical_features_treatment_arms(self) -> None:
        """
        Evaluate correlations between clinical features and CGM endpoints by treatment arm.

        Splits each dataset into predefined treatment groups, merges clinical metrics with
        per-time-bin endpoints, and saves separate Spearman heatmaps for each arm.

        Args:
            None

        Returns:
            None: Heatmap images are saved to disk.

        Raises:
            ValueError: If required columns ('treatment_arm', 'time_bin', 'id') are missing.
        """
        dict_df_test = self.datasets.copy()
        dict_t1_dfs_for_plot = {}
        dict_t2_dfs_for_plot = {}

        for df_name, df_t1d in dict_df_test.items():
            df_t1d = df_t1d[~(df_t1d['time_bin'] == 'N/A')].compute()
            df_t1d['treatment_arm'] = df_t1d['treatment_arm'].str.lower()
            df_t1_t1d = df_t1d[df_t1d['treatment_arm'].isin(TREATMENT_GROUP_1)]
            df_t2_t1d = df_t1d[df_t1d['treatment_arm'].isin(TREATMENT_GROUP_2)]

            df_t1_time_bins = df_t1_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_t1_time_bins = df_t1_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_t1_time_bins.drop(columns = ['level_1'], inplace=True)
            df_t2_time_bins = df_t2_t1d.groupby(['time_bin']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_t2_time_bins = df_t2_time_bins.groupby('time_bin').filter(lambda g: len(g) >= 10)
            df_t2_time_bins.drop(columns = ['level_1'], inplace=True)

            df_t1_cpep_auc = df_t1_t1d[['id', 'time_bin', 'cpep_auc', 'beta2_score']]
            df_t1_cpep_auc = df_t1_cpep_auc.drop_duplicates(subset=['id', 'time_bin'])
            df_t1_cpep_auc = df_t1_cpep_auc.merge(df_t1_time_bins, on=['id', 'time_bin'], how='inner')
            df_t2_cpep_auc = df_t2_t1d[['id', 'time_bin', 'cpep_auc', 'beta2_score']]
            df_t2_cpep_auc = df_t2_cpep_auc.drop_duplicates(subset=['id', 'time_bin'])
            df_t2_cpep_auc = df_t2_cpep_auc.merge(df_t2_time_bins, on=['id', 'time_bin'], how='inner')

            dict_t1_dfs_for_plot[df_name] = df_t1_cpep_auc
            dict_t2_dfs_for_plot[df_name] = df_t2_cpep_auc

        clinical_feats = ['cpep_auc', 'beta2_score']
        all_cgm_feats = [col for col in df_t1_time_bins.columns if col not in ['id', 'time_bin', 'percent_wear_time']]

        for clinical_feat in clinical_feats:
            self.plot_spearman_heatmap_from_dfs(dict_t1_dfs_for_plot, clinical_feature=clinical_feat, cgm_features=all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/t1_heatmap_{clinical_feat}.png')
            self.plot_spearman_heatmap_from_dfs(dict_t2_dfs_for_plot, clinical_feature=clinical_feat, cgm_features=all_cgm_feats, path=f'./data/graphs/feature_analysis/time_bins/heatmaps/t2_heatmap_{clinical_feat}.png')

    def ht_cpep_strata(self) -> None:
        """
        Run raw descriptive metabolic-endpoint analyses across C-peptide strata.

        For each dataset (`cloud`, `clvr`, and pooled `combined`) the function:
            - builds patient-level summaries by visit/time bin
            - assigns C-peptide AUC strata
            - separates participants into control and active treatment arms
            - compares endpoint distributions across strata within each arm using
              the Kruskal-Wallis test
            - exports descriptive summary tables and arm-specific figure panels

        Args:
            None

        Returns:
            None: Raw tables and figures are saved to disk.
        """
        df_cloud = self.get_dataset('cloud').compute()
        df_clvr = self.get_dataset('clvr').compute()
        if 'insulin_delivery' in df_clvr.columns:
            df_clvr = df_clvr.drop(columns=['treatment_arm'], errors='ignore')
            df_clvr = df_clvr.rename(columns={'insulin_delivery': 'treatment_arm'})
            df_clvr.dropna(subset=['treatment_arm'], inplace=True)
        df_cloud = df_cloud.assign(study='cloud')
        df_clvr = df_clvr.assign(study='clvr')
        df_cloud_clvr = pd.concat([df_cloud, df_clvr], ignore_index=True)

        datasets = [
            ('cloud', df_cloud),
            ('clvr', df_clvr),
            ('combined', df_cloud_clvr)
        ]

        cpep_bins = [0.0, 0.2, 0.5, 0.8, np.inf]
        cpep_labels = ['0.0-<0.2', '0.2-<0.5', '0.5-<0.8', '>=0.8']
        preferred_endpoint_order = [
            endpoint for endpoint in CGM_ENDPOINTS
            if endpoint in FEATURE_LABELS_WITH_UNITS or endpoint in {'GVP'}
        ]
        output_root = './data/graphs/feature_analysis/cpep_strata_raw'
        csv_output_root = './data/csv_results/hypothesis_tests/cpep_strata_raw'
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(csv_output_root, exist_ok=True)

        long_rows = []
        summary_rows = []
        posthoc_rows = []

        def _format_summary(series: pd.Series, n_ids: int) -> str:
            if series.empty:
                return 'NA'
            q1 = series.quantile(0.25)
            median = series.median()
            q3 = series.quantile(0.75)
            return (
                f"{median:.2f} [{q1:.2f}, {q3:.2f}] "
                f"(obs={len(series)}, ids={n_ids})"
            )

        for dataset_name, df in datasets:
            endpoints = self.get_taylor_patient_endpoints(
                df,
                metabolic_endpoints=set(CGM_CORE_ENDPOINTS),
                by_time_bin=True,
                include_cols={'cpep_auc', 'treatment_arm', 'gmi', 'hb_a1c', 'total_ins_dose'}
            )
            if endpoints.empty:
                continue

            endpoints['treatment_arm'] = (
                endpoints['treatment_arm']
                .astype('string')
                .str.strip()
                .str.lower()
            )
            endpoints['treatment_arm'] = np.where(
                endpoints['treatment_arm'].isin(TREATMENT_GROUP_1),
                'Control',
                np.where(
                    endpoints['treatment_arm'].isin(TREATMENT_GROUP_2),
                    'Active',
                    pd.NA
                )
            )
            endpoints = endpoints.dropna(subset=['treatment_arm', 'cpep_auc']).copy()
            endpoints['cpep_stratum'] = pd.cut(
                endpoints['cpep_auc'],
                bins=cpep_bins,
                labels=cpep_labels,
                include_lowest=True,
                right=False
            )
            endpoints = endpoints.dropna(subset=['cpep_stratum']).copy()
            endpoints['cpep_stratum'] = pd.Categorical(
                endpoints['cpep_stratum'],
                categories=cpep_labels,
                ordered=True
            )

            dataset_dir = os.path.join(output_root, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            available_endpoints = [col for col in preferred_endpoint_order if col in endpoints.columns]

            for arm_name in ['Control', 'Active']:
                arm_df = endpoints[endpoints['treatment_arm'] == arm_name].copy()
                if arm_df.empty:
                    continue

                arm_slug = arm_name.lower()
                plot_endpoints = []

                for endpoint in available_endpoints:
                    endpoint_df = arm_df[['id', 'time_bin', 'cpep_stratum', endpoint]].dropna(subset=[endpoint]).copy()
                    if endpoint_df.empty:
                        continue

                    strata_present = []
                    values_by_group = {}
                    for stratum in cpep_labels:
                        stratum_df = endpoint_df[endpoint_df['cpep_stratum'] == stratum]
                        values = stratum_df[endpoint].dropna()
                        if not values.empty:
                            strata_present.append(stratum)
                            values_by_group[stratum] = values

                    kw_stat, kw_p = self.run_kruskal_test(values_by_group)
                    if len(strata_present) >= 2 and pd.notna(kw_p) and kw_p < 0.05:
                        for posthoc_result in self.run_dunn_posthoc(values_by_group):
                            posthoc_rows.append({
                                'dataset': dataset_name,
                                'treatment_arm': arm_name,
                                'endpoint': endpoint,
                                'kruskal_wallis_stat': kw_stat,
                                'kruskal_wallis_p_value': kw_p,
                                **posthoc_result
                            })

                    summary_row = {
                        'dataset': dataset_name,
                        'treatment_arm': arm_name,
                        'endpoint': endpoint,
                        'kruskal_wallis_stat': kw_stat,
                        'kruskal_wallis_p_value': kw_p
                    }

                    for stratum in cpep_labels:
                        stratum_df = endpoint_df[endpoint_df['cpep_stratum'] == stratum]
                        values = stratum_df[endpoint].dropna()
                        n_ids = stratum_df['id'].nunique()
                        summary = _format_summary(values, n_ids)
                        summary_row[f'{stratum}_summary'] = summary
                        long_rows.append({
                            'dataset': dataset_name,
                            'treatment_arm': arm_name,
                            'endpoint': endpoint,
                            'cpep_stratum': stratum,
                            'participants': n_ids,
                            'observations': len(values),
                            'median': values.median() if not values.empty else np.nan,
                            'q1': values.quantile(0.25) if not values.empty else np.nan,
                            'q3': values.quantile(0.75) if not values.empty else np.nan,
                            'kruskal_wallis_stat': kw_stat,
                            'kruskal_wallis_p_value': kw_p
                        })
                    summary_rows.append(summary_row)
                    plot_endpoints.append(endpoint)

                if not plot_endpoints:
                    continue

                ncols = 2
                nrows = math.ceil(len(plot_endpoints) / ncols)
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows))
                axes = np.atleast_1d(axes).ravel()

                palette = {
                    '0.0-<0.2': '#d8b365',
                    '0.2-<0.5': '#f5f5f5',
                    '0.5-<0.8': '#5ab4ac',
                    '>=0.8': '#01665e'
                }

                for ax, endpoint in zip(axes, plot_endpoints):
                    endpoint_df = arm_df[['id', 'cpep_stratum', endpoint]].dropna(subset=[endpoint]).copy()
                    if endpoint_df.empty:
                        ax.set_visible(False)
                        continue

                    sns.boxplot(
                        data=endpoint_df,
                        x='cpep_stratum',
                        y=endpoint,
                        order=cpep_labels,
                        palette=palette,
                        showfliers=False,
                        ax=ax
                    )
                    sns.stripplot(
                        data=endpoint_df,
                        x='cpep_stratum',
                        y=endpoint,
                        order=cpep_labels,
                        color='black',
                        size=3,
                        alpha=0.45,
                        jitter=0.2,
                        ax=ax
                    )

                    kw_row = next(
                        (
                            row for row in summary_rows
                            if row['dataset'] == dataset_name
                            and row['treatment_arm'] == arm_name
                            and row['endpoint'] == endpoint
                        ),
                        None
                    )
                    kw_p = kw_row['kruskal_wallis_p_value'] if kw_row is not None else np.nan
                    title = FEATURE_LABELS_WITH_UNITS.get(endpoint, endpoint)
                    if pd.notna(kw_p):
                        title += f"\nKruskal-Wallis p={kw_p:.3f}"
                    else:
                        title += "\nKruskal-Wallis p=NA"
                    ax.set_title(title, fontsize=11)
                    ax.set_xlabel('C-peptide stratum')
                    ax.set_ylabel(FEATURE_LABELS_WITH_UNITS.get(endpoint, endpoint))
                    ax.tick_params(axis='x', rotation=20)

                for ax in axes[len(plot_endpoints):]:
                    ax.set_visible(False)

                fig.suptitle(f'{dataset_name.upper()} {arm_name} arm', fontsize=15)
                fig.tight_layout(rect=[0, 0, 1, 0.98])
                fig.savefig(
                    os.path.join(dataset_dir, f'{arm_slug}_cpep_strata_metabolic_endpoints.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close(fig)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            numeric_cols = summary_df.select_dtypes(include='number').columns
            summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
            summary_df = summary_df.sort_values(['dataset', 'treatment_arm', 'endpoint'])
            summary_df.to_csv(
                os.path.join(csv_output_root, 'cpep_strata_descriptive_summary.csv'),
                index=False
            )

        if long_rows:
            long_df = pd.DataFrame(long_rows)
            numeric_cols = long_df.select_dtypes(include='number').columns
            long_df[numeric_cols] = long_df[numeric_cols].round(3)
            long_df = long_df.sort_values(
                [col for col in ['dataset', 'treatment_arm', 'endpoint', 'cpep_stratum'] if col in long_df.columns]
            )
            long_df.to_csv(
                os.path.join(csv_output_root, 'cpep_strata_descriptive_long.csv'),
                index=False
            )

        if posthoc_rows:
            posthoc_df = pd.DataFrame(posthoc_rows)
            numeric_cols = posthoc_df.select_dtypes(include='number').columns
            posthoc_df[numeric_cols] = posthoc_df[numeric_cols].round(3)
            posthoc_df = posthoc_df.sort_values(
                ['dataset', 'treatment_arm', 'endpoint', 'group_1', 'group_2']
            )
            posthoc_df.to_csv(
                os.path.join(csv_output_root, 'cpep_strata_posthoc_dunn.csv'),
                index=False
            )

        return None

    def taylor_analysis(self) -> None:
        """
        Runs the full Taylor-style time bin analysis across all loaded datasets.

        Workflow:
            - Splits datasets into positive and negative study groups.
            - Generates scatterplots of relationships between:
                1. log(C-peptide AUC) vs clinical/CGM metrics
                2. Beta2 score vs clinical/CGM metrics

        Returns:
            None. Saves scatterplots to ./data/graphs/taylor_analysis/time_bins/.
        """
        dict_dfs = self.datasets.copy()
        allowed_time_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12', 'Month 18', 'Month 24']
        all_dict = {k: v.compute() for k, v in dict_dfs.items() if k in ALL_STUDIES}
        positive_dict = {k: v.compute() for k, v in dict_dfs.items() if k in POSITIVE_STUDIES}
        negative_dict = {k: v.compute() for k, v in dict_dfs.items() if k in NEGATIVE_STUDIES}
        healthy_reference = dict_dfs['jaeb_healthy'].compute()
        healthy_reference['total_ins_dose'] = 0
        established_reference = dict_dfs['hupa_ucm'].compute()

        all_dict = {k: df[df['time_bin'].isin(allowed_time_bins)] for k, df in all_dict.items()}
        positive_dict = {k: df[df['time_bin'].isin(allowed_time_bins)] for k, df in positive_dict.items()}
        negative_dict = {k: df[df['time_bin'].isin(allowed_time_bins)] for k, df in negative_dict.items()}

        ''' Part A'''
        # Time analysis
        order = ['median_cpep', 'median_beta2_score','median_insulin_dose','median_a1c', 'median_gmi', 'median_tir',
                 'median_titr','median_tbr_lvl1','median_tbr_lvl2','median_tar_lvl1','median_tar_lvl2']
        metrics = order
        
        self.plot_taylor_time_bins_graph(dict_df=all_dict, metrics=metrics, path='./data/graphs/taylor_analysis/time_bins/all/')
        self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/all/',
                          output='./data/graphs/taylor_analysis/time_bins/all/combined.png',
                          cols=4,
                          ordered_labels=order)
        
        self.plot_taylor_time_bins_graph(dict_df=positive_dict, metrics=metrics, path='./data/graphs/taylor_analysis/time_bins/positive/')
        self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/positive/',
                          output='./data/graphs/taylor_analysis/time_bins/positive/combined.png',
                          cols=4,
                          ordered_labels=order)
        
        self.plot_taylor_time_bins_graph(dict_df=negative_dict, metrics=metrics, path='./data/graphs/taylor_analysis/time_bins/negative/')
        self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/negative/',
                          output='./data/graphs/taylor_analysis/time_bins/negative/combined.png',
                          cols=4,
                          ordered_labels=order)

        for df_name, df in all_dict.items():
            self.plot_taylor_time_bins_graph(dict_df={df_name: df}, metrics=metrics, path=f'./data/graphs/taylor_analysis/time_bins/{df_name}/')
            self.combine_pngs(directory=f'./data/graphs/taylor_analysis/time_bins/{df_name}/',
                            output=f'./data/graphs/taylor_analysis/time_bins/{df_name}/combined.png',
                            cols=4,
                            ordered_labels=order)

        # # C-Pep AUC as Independent Value
        # order = None
        # x_feature = 'log_cpep_auc'
        # y_features = ['total_ins_dose', 'beta2_score', 'hb_a1c',
        #               'TIR', 'TBR_Lvl_1', 'TBR_Lvl_2', 'TAR_Lvl_1', 'TAR_Lvl_2', 'cv_percent']

        # self.plot_taylor_time_bins_scatterplot(all_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/all/scatterplot/cpep_auc/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/all/scatterplot/cpep_auc/',
        #                 output=f'./data/graphs/taylor_analysis/time_bins/all/scatterplot/cpep_auc/combined.png',
        #                 cols=4)

        # self.plot_taylor_time_bins_scatterplot(positive_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/cpep_auc/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/cpep_auc/',
        #                 output='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/cpep_auc/combined.png',
        #                 cols=4)

        # self.plot_taylor_time_bins_scatterplot(negative_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/cpep_auc/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/cpep_auc/',
        #                 output='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/cpep_auc/combined.png',
        #                 cols=4)

        # for df_name, df in all_dict.items():
        #     self.plot_taylor_time_bins_scatterplot(dict_df={df_name: df},
        #                                            x_feature=x_feature, y_features=y_features,
        #                                            path=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/cpep_auc/')
        #     self.combine_pngs(directory=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/cpep_auc/',
        #                     output=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/cpep_auc/combined.png',
        #                     cols=4)

        # # BETA2 Score as Independent Value
        # order = None
        # x_feature = 'beta2_score'
        # y_features = ['total_ins_dose', 'hb_a1c',
        #               'TIR', 'TBR_Lvl_1', 'TBR_Lvl_2', 'TAR_Lvl_1', 'TAR_Lvl_2', 'cv_percent']

        # self.plot_taylor_time_bins_scatterplot(all_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/all/scatterplot/beta2_score/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/all/scatterplot/beta2_score/',
        #                 output='./data/graphs/taylor_analysis/time_bins/all/scatterplot/beta2_score/combined.png',
        #                 cols=4)

        # self.plot_taylor_time_bins_scatterplot(positive_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/beta2_score/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/beta2_score/',
        #                 output='./data/graphs/taylor_analysis/time_bins/positive/scatterplot/beta2_score/combined.png',
        #                 cols=4)

        # self.plot_taylor_time_bins_scatterplot(negative_dict,
        #                                        x_feature=x_feature, y_features=y_features,
        #                                        path='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/beta2_score/')
        # self.combine_pngs(directory='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/beta2_score/',
        #                 output='./data/graphs/taylor_analysis/time_bins/negative/scatterplot/beta2_score/combined.png',
        #                 cols=4)

        # for df_name, df in all_dict.items():
        #     self.plot_taylor_time_bins_scatterplot(dict_df={df_name: df},
        #                                            x_feature=x_feature, y_features=y_features,
        #                                            path=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/beta2_score/')
        #     self.combine_pngs(directory=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/beta2_score/',
        #                     output=f'./data/graphs/taylor_analysis/time_bins/{df_name}/scatterplot/beta2_score/combined.png',
        #                     cols=4)

        ''' Part B'''
        metabolic_endpoints = {'hb_a1c', 'gmi', 'TIR', 'TITR', 'total_ins_dose', 'TBR_Lvl_1', 'TBR_Lvl_2', 'TAR_Lvl_1', 'TAR_Lvl_2',
                               'cv_percent', 'GVP'}   
        
        # # Stratify by absolute Cpeptide
        # self.taylor_metabolic_endpoints_by_group_total(
        #     dict_df=all_dict,
        #     group_by='cpep_auc',
        #     ranges=[-float('inf'),0.2,0.5,0.8,float('inf')],
        #     labels=['0-0.2','0.2-0.5','0.5-0.8','0.8<'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

        # self.taylor_metabolic_endpoints_by_group_time_bins(
        #     dict_df=all_dict,
        #     group_by='cpep_auc',
        #     ranges=[-float('inf'),0.2,0.5,0.8,float('inf')],
        #     labels=['0-0.2','0.2-0.5','0.5-0.8','0.8<'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

        # # Stratify by %Cpeptide preservation
        # self.taylor_metabolic_endpoints_by_group_total(
        #     dict_df=all_dict,
        #     group_by='cpep_auc_preservation',
        #     ranges=[-float('inf'),20,40,60,80,float('inf')],
        #     labels=['0-20%','20-40%','40-60%','60-80%','80-100%'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

        # self.taylor_metabolic_endpoints_by_group_time_bins(
        #     dict_df=all_dict,
        #     group_by='cpep_auc_preservation',
        #     ranges=[-float('inf'),20,40,60,80,float('inf')],
        #     labels=['0-20%','20-40%','40-60%','60-80%','80-100%'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

        # # Stratify by Beta2 Score
        # self.taylor_metabolic_endpoints_by_group_total(
        #     dict_df=all_dict,
        #     group_by='beta2_score',
        #     ranges = [-float('inf'),5,10,15,20],
        #     labels = ['0-5','6-10','11-14','15-20'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

        # self.taylor_metabolic_endpoints_by_group_time_bins(
        #     dict_df=all_dict,
        #     group_by='beta2_score',
        #     ranges = [-float('inf'),5,10,15,20],
        #     labels = ['0-5','6-10','11-14','15-20'],
        #     healthy_reference=healthy_reference,
        #     established_reference=established_reference,
        #     metabolic_endpoints=metabolic_endpoints
        # )

    def taylor_metabolic_endpoints_by_group_time_bins(
        self,
        group_by: str,
        ranges: list,
        labels: list,
        dict_df: dict[str, pd.DataFrame],
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        metabolic_endpoints: set) -> None:
        """
        Plot metabolic endpoints by group and time bin, comparing to healthy/established references.

        Args:
            group_by: Column used to create grouping bins.
            ranges: Bin edges for grouping.
            labels: Labels for the group bins.
            dict_df: Mapping of study name to DataFrame with metabolic endpoints.
            healthy_reference: Reference DataFrame for healthy cohort CGM endpoints.
            established_reference: Reference DataFrame for established cohort CGM endpoints.
            metabolic_endpoints: Set of endpoint column names to plot.

        Returns:
            None
        """

        df_t1d = pd.concat(dict_df.values(), ignore_index=True)
        t1d_patient = self.get_taylor_patient_endpoints(
            df_t1d,
            metabolic_endpoints,
            by_time_bin=True,
            include_cols={group_by}
        )
        healthy_patient = self.get_taylor_patient_endpoints(healthy_reference, metabolic_endpoints, by_time_bin=False)
        established_patient = self.get_taylor_patient_endpoints(established_reference, metabolic_endpoints, by_time_bin=False)

        for time_bin, df_time_bin in t1d_patient.groupby('time_bin'):
            df_time_bin = df_time_bin.copy()
            df_time_bin['group'] = pd.cut(df_time_bin[group_by], bins=ranges, labels=labels, ordered=True)
        
            for endpoint in sorted(metabolic_endpoints):
                self.plot_taylor_metabolic_endpoints_by_group(
                    metabolic_endpoint=endpoint,
                    t1d_metabolic_endpoint=df_time_bin,
                    healthy_metabolic_endpoint=healthy_patient,
                    established_metabolic_endpoint=established_patient,
                    x_label=group_by,
                    title_month=time_bin,
                    path=f'./data/graphs/taylor_analysis/{group_by}/{endpoint}/{time_bin}.png',
                    group_labels=labels
                )
                order = ['Baseline','Month 3','Month 6','Month 9','Month 12','Month 18','Month 24']
                self.combine_pngs(directory=f'./data/graphs/taylor_analysis/{group_by}/{endpoint}/',
                                output=f'./data/graphs/taylor_analysis/{group_by}/{endpoint}/combine.png',
                                cols=3,
                                ordered_labels=order)

    def taylor_metabolic_endpoints_by_group_total(
        self,
        group_by: str,
        ranges: list,
        labels: list,
        dict_df: dict[str, pd.DataFrame],
        healthy_reference: pd.DataFrame,
        established_reference: pd.DataFrame,
        metabolic_endpoints: set) -> None:
        """
        Plot metabolic endpoints grouped by a clinical variable (e.g., beta2 ranges) against healthy/established refs.

        Args:
            group_by: Column used to create grouping bins.
            ranges: Bin edges for grouping.
            labels: Labels for the group bins.
            dict_df: Mapping of study name to DataFrame with metabolic endpoints.
            healthy_reference: Reference DataFrame for healthy cohort CGM endpoints.
            established_reference: Reference DataFrame for established cohort CGM endpoints.
            metabolic_endpoints: Set of endpoint column names to plot.

        Returns:
            None
        """

        df_t1d = pd.concat(dict_df.values(), ignore_index=True)
        t1d_patient = self.get_taylor_patient_endpoints(
            df_t1d,
            metabolic_endpoints,
            by_time_bin=False,
            include_cols={group_by}
        )
        t1d_patient['group'] = pd.cut(t1d_patient[group_by], bins=ranges, labels=labels, ordered=True)

        healthy_patient = self.get_taylor_patient_endpoints(healthy_reference, metabolic_endpoints, by_time_bin=False)
        established_patient = self.get_taylor_patient_endpoints(established_reference, metabolic_endpoints, by_time_bin=False)

        for endpoint in sorted(metabolic_endpoints):
            self.plot_taylor_metabolic_endpoints_by_group(
                metabolic_endpoint=endpoint,
                t1d_metabolic_endpoint=t1d_patient,
                healthy_metabolic_endpoint=healthy_patient,
                established_metabolic_endpoint=established_patient,
                x_label=group_by,
                title_month=None,
                path=f'./data/graphs/taylor_analysis/{group_by}/{endpoint}.png',
                group_labels=labels
            )
        order = ['hb_a1c', 'gmi', 'TIR', 'TITR', 'total_ins_dose', 'TAR_Lvl_1', 'TAR_Lvl_2', 'TBR_Lvl_1', 'TBR_Lvl_2', 'cv_percent', 'GVP']
        self.combine_pngs(directory=f'./data/graphs/taylor_analysis/{group_by}/',
                              output=f'./data/graphs/taylor_analysis/{group_by}/combine.png',
                              cols=3,
                              ordered_labels=order)

    """ Linear Models """
    def lmm_cpep_between_strata(self) -> None:
        """
        Fit mixed-effects models comparing metabolic endpoints across C-peptide strata.

        Uses pooled T1D study summaries, adjusts for study, and writes adjusted means plus
        omnibus and linear-trend test results for each endpoint.

        LMM formulas per endpoint:
            - Full fixed-effects formula: `endpoint ~ C(cpep_stratum) + C(study)`
            - Reduced fixed-effects formula: `endpoint ~ C(study)`
            - Trend fixed-effects formula: `endpoint ~ cpep_stratum_num + C(study)`
            - Random intercept: supplied separately via `group_col='id'`
            - Equivalent mixed-model notation:
              `endpoint ~ ... + (1 | id)`

        Args:
            None

        Returns:
            None
        """
        #TODO
        dict_dfs = self.datasets.copy()

        all_dict = {k: v.compute() for k, v in dict_dfs.items() if k in ALL_STUDIES}
        study_split = re.compile(r"[-_/]")
        df_t1d = pd.concat(
            [
                df.assign(study=study_split.split(study_name, maxsplit=1)[0])
                for study_name, df in all_dict.items()
            ],
            ignore_index=True
        )
        t1d_endpoints = set(CGM_CORE_ENDPOINTS) | {'gmi', 'hb_a1c'}
        df_t1d_time_bins = self.get_taylor_patient_endpoints(
            df_t1d,
            CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'cpep_auc', 'study', 'gmi', 'hb_a1c'}
        )
        if 'study' in df_t1d_time_bins.columns:
            df_t1d_time_bins['study'] = df_t1d_time_bins['study'].astype('category')

        cpep_bins = [0.0, 0.2, 0.5, 0.8, np.inf]
        df_t1d_time_bins['cpep_stratum'] = pd.cut(
            df_t1d_time_bins['cpep_auc'],
            bins=cpep_bins,
            labels=[1, 2, 3, 4],
            include_lowest=True,
            right=False
        ).astype('Int64')
            
        df_t1d_time_bins['cpep_stratum'] = pd.Categorical(
            df_t1d_time_bins['cpep_stratum'],
            categories=[1, 2, 3, 4],
            ordered=True
        )

        desc_df = df_t1d_time_bins.dropna(subset=['cpep_stratum'])
        if not desc_df.empty:
            participants_per_stratum = (
                desc_df.groupby('cpep_stratum')['id']
                .nunique()
            )
            observations_per_stratum = (
                desc_df.groupby('cpep_stratum')
                .size()
            )
            print("Unique participants per C-peptide stratum:")
            print(participants_per_stratum.to_string())
            print("\nCGM observations per C-peptide stratum:")
            print(observations_per_stratum.to_string())

        full_models = {}
        reduced_models = {}
        adjusted_means = []
        lrt_results = []
        trend_results = []
        for endpoint in t1d_endpoints:
            df_endpoint = df_t1d_time_bins[['study', 'id', 'time_bin', endpoint, 'cpep_stratum']]
            df_endpoint = df_endpoint.dropna(subset=['cpep_stratum', endpoint, 'study'])
            try:
                result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=f"{endpoint} ~ C(cpep_stratum) + C(study)",
                    group_col='id',
                    reml=False
                )
                full_models[endpoint] = result

                reduced_result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=f"{endpoint} ~ C(study)",
                    group_col='id',
                    reml=False
                )
                reduced_models[endpoint] = reduced_result
                lrt_result = self.run_likelihood_ratio_test(result, reduced_result)
                lrt_results.append({
                    'endpoint': endpoint,
                    'lrt_stat': lrt_result['lrt_stat'],
                    'df_diff': lrt_result['df_diff'],
                    'p_value': lrt_result['p_value']
                })

                df_endpoint = df_endpoint.copy()
                df_endpoint['cpep_stratum_num'] = df_endpoint['cpep_stratum'].cat.codes + 1
                trend_result = self.run_linear_mixed_model(
                    df=df_endpoint,
                    formula=f"{endpoint} ~ cpep_stratum_num + C(study)",
                    group_col='id',
                    reml=False
                )
                trend_results.append({
                    'endpoint': endpoint,
                    'trend_coef': trend_result.params.get('cpep_stratum_num', np.nan),
                    'trend_p_value': trend_result.pvalues.get('cpep_stratum_num', np.nan)
                })
                design_info = result.model.data.design_info
                fe_params = result.fe_params
                cov_params = result.cov_params()
                if isinstance(cov_params, pd.DataFrame):
                    cov_fe = cov_params.iloc[:len(fe_params), :len(fe_params)]
                else:
                    cov_fe = cov_params[:len(fe_params), :len(fe_params)]

                study_weights = (
                    df_endpoint['study']
                    .value_counts(normalize=True, sort=False)
                    .to_dict()
                )
                cpep_levels = df_endpoint['cpep_stratum'].cat.categories
                for level in cpep_levels:
                    if level not in df_endpoint['cpep_stratum'].values:
                        continue
                    design_df = pd.DataFrame({
                        'cpep_stratum': pd.Categorical(
                            [level] * len(study_weights),
                            categories=cpep_levels,
                            ordered=True
                        ),
                        'study': list(study_weights.keys())
                    })
                    exog = patsy.build_design_matrices(
                        [design_info],
                        design_df,
                        return_type='dataframe'
                    )[0]
                    weights = np.array([study_weights[s] for s in design_df['study']])
                    xbar = (exog.to_numpy() * weights[:, None]).sum(axis=0)
                    mean = float(xbar @ fe_params.to_numpy())
                    se = float(np.sqrt(xbar @ cov_fe.to_numpy() @ xbar.T))
                    ci_lower = mean - 1.96 * se
                    ci_upper = mean + 1.96 * se
                    adjusted_means.append({
                        'endpoint': endpoint,
                        'cpep_stratum': level,
                        'adjusted_mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })
            except Exception as exc:
                print(f"Full LMM failed for {endpoint}: {exc}")

        if adjusted_means:
            os.makedirs('./data/feature_analysis', exist_ok=True)
            adjusted_means_df = pd.DataFrame(adjusted_means)
            lrt_map = {row['endpoint']: row['p_value'] for row in lrt_results}
            trend_map = {row['endpoint']: row['trend_p_value'] for row in trend_results}
            adjusted_means_df['p_value_omnibus'] = adjusted_means_df['endpoint'].map(lrt_map)
            adjusted_means_df['p_value_trend'] = adjusted_means_df['endpoint'].map(trend_map)
            numeric_cols = adjusted_means_df.select_dtypes(include='number').columns
            adjusted_means_df[numeric_cols] = adjusted_means_df[numeric_cols].round(2)
            for col in ['p_value_omnibus', 'p_value_trend']:
                if col in adjusted_means_df.columns:
                    adjusted_means_df[col] = adjusted_means_df[col].apply(
                        lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
                    )
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = adjusted_means_df['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            adjusted_means_df['endpoint'] = pd.Categorical(
                adjusted_means_df['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            adjusted_means_df = adjusted_means_df.sort_values(['endpoint', 'cpep_stratum'])
            adjusted_means_df.to_csv(
                './data/csv_results/lmm_results/cpepstrata_study_adjusted_means.csv',
                index=False
            )
        if lrt_results:
            lrt_df = pd.DataFrame(lrt_results)
            print("\nLikelihood ratio tests (full vs reduced):")
            print(lrt_df.to_string(index=False))
        if trend_results:
            trend_df = pd.DataFrame(trend_results)
            print("\nLinear trend tests (cpep_stratum_num):")
            print(trend_df.to_string(index=False))

        return None

    def lmm_cpep_cont_vs_strata(self) -> None:
        """
        Compare continuous versus stratified C-peptide mixed models by treatment arm.

        Builds endpoint summaries for `cloud`, `clvr`, and their combined dataset, then fits:
        1. A model using continuous log-transformed C-peptide with treatment interaction.
        2. A model using categorical C-peptide strata with treatment interaction.
        The function exports fixed effects, stratum-level contrasts, emmeans, and count tables.

        LMM formulas per endpoint:
            - Model 1 fixed-effects formula:
              `endpoint ~ C(treatment_arm) + log_cpep_auc + C(treatment_arm):log_cpep_auc`
            - Model 2 fixed-effects formula:
              `endpoint ~ C(treatment_arm) + C(cpep_stratum) + C(treatment_arm):C(cpep_stratum)`
            - If multiple studies are present, both models add `+ C(study)`
            - Random intercept: supplied separately via `group_col='id'`
            - Equivalent mixed-model notation:
              `endpoint ~ ... + (1 | id)`

        Args:
            None

        Returns:
            None
        """
        df_cloud = self.get_dataset('cloud')
        df_clvr = self.get_dataset('clvr')
        if 'insulin_delivery' in df_clvr.columns:
            df_clvr = df_clvr.drop(columns=['treatment_arm'], errors='ignore')
            df_clvr = df_clvr.rename(columns={'insulin_delivery': 'treatment_arm'})
            df_clvr.dropna(subset=['treatment_arm'], inplace=True)
        df_cloud = df_cloud.assign(study='cloud')
        df_clvr = df_clvr.assign(study='clvr')
        df_cloud_clvr = pd.concat([df_cloud, df_clvr], ignore_index=True)

        '''Part 1'''

        datasets = [
            ('cloud', df_cloud),
            ('clvr', df_clvr),
            ('cloud_clvr', df_cloud_clvr)
        ]
        model1_results = {}
        model2_results = {}
        model1_summary_rows = []
        model2_summary_rows = []
        model2_diff_rows = []
        model2_emmeans_rows = []
        counts_rows = []

        agp_endpoints = set(CGM_CORE_ENDPOINTS) | {'gmi', 'hb_a1c'}
        for name, df in datasets:
            endpoints = self.get_taylor_patient_endpoints(
                df,
                metabolic_endpoints=CGM_CORE_ENDPOINTS,
                by_time_bin=True,
                include_cols={'study', 'cpep_auc', 'treatment_arm', 'insulin_delivery', 'gmi', 'hb_a1c'})
            treatment_source = 'treatment_arm'
            endpoints['treatment_arm'] = (
                endpoints[treatment_source]
                .astype('string')
                .str.strip()
                .str.lower()
            )
            endpoints.loc[
                endpoints['treatment_arm'].isin(TREATMENT_GROUP_1),
                'treatment_arm'
            ] = 'control'
            endpoints.loc[
                endpoints['treatment_arm'].isin(TREATMENT_GROUP_2),
                'treatment_arm'
            ] = 'active'
            endpoints['treatment_arm'] = pd.Categorical(
                endpoints['treatment_arm'],
                categories=['control', 'active']
            )
            endpoints = endpoints.dropna(subset=['cpep_auc'])
            endpoints['log_cpep_auc'] = np.log(endpoints['cpep_auc'] + 1)
            cpep_bins = [0.0, 0.2, 0.5, 0.8, 1.0]
            endpoints['cpep_stratum'] = pd.cut(
                endpoints['cpep_auc'],
                bins=cpep_bins,
                labels=[1, 2, 3, 4],
                include_lowest=True,
                right=False
            ).astype('Int64')
            endpoints['cpep_stratum'] = pd.Categorical(
                endpoints['cpep_stratum'],
                categories=[1, 2, 3, 4],
                ordered=True
            )

            counts_df = endpoints.dropna(subset=['cpep_stratum', 'treatment_arm'])
            if not counts_df.empty:
                participants_counts = (
                    counts_df.groupby(['cpep_stratum', 'treatment_arm'])['id']
                    .nunique()
                    .rename('participants')
                    .reset_index()
                )
                observation_counts = (
                    counts_df.groupby(['cpep_stratum', 'treatment_arm'])
                    .size()
                    .rename('observations')
                    .reset_index()
                )
                counts_summary = pd.merge(
                    participants_counts,
                    observation_counts,
                    on=['cpep_stratum', 'treatment_arm'],
                    how='outer'
                )
                counts_summary['dataset'] = name
                counts_rows.append(counts_summary)

            include_study = endpoints['study'].nunique(dropna=True) > 1
            for endpoint in agp_endpoints:
                df_endpoint = endpoints[
                    ['id', 'study', 'treatment_arm', 'log_cpep_auc', 'cpep_stratum', endpoint]
                ].dropna(subset=['treatment_arm', 'log_cpep_auc', endpoint])
                if df_endpoint.empty:
                    continue

                model1_formula = (
                    f"{endpoint} ~ C(treatment_arm) + log_cpep_auc + "
                    "C(treatment_arm):log_cpep_auc"
                )
                if include_study:
                    model1_formula += " + C(study)"
                try:
                    model1 = self.run_linear_mixed_model(
                        df=df_endpoint,
                        formula=model1_formula,
                        group_col='id',
                        reml=False
                    )
                    model1_results[(name, endpoint)] = model1
                    fe_params = model1.fe_params
                    conf_int = model1.conf_int().loc[fe_params.index]
                    pvals = model1.pvalues.loc[fe_params.index]
                    model1_table = pd.DataFrame({
                        'term': fe_params.index,
                        'estimate': fe_params.values,
                        'ci_lower': conf_int[0].values,
                        'ci_upper': conf_int[1].values,
                        'p_value': pvals.values
                    })
                    model1_table['dataset'] = name
                    model1_table['endpoint'] = endpoint
                    model1_summary_rows.append(model1_table)
                except Exception as exc:
                    print(f"Model 1 failed for {name} {endpoint}: {exc}")

                df_endpoint_strata = df_endpoint.dropna(subset=['cpep_stratum'])
                if df_endpoint_strata.empty:
                    continue
                model2_formula = (
                    f"{endpoint} ~ C(treatment_arm) + C(cpep_stratum) + "
                    "C(treatment_arm):C(cpep_stratum)"
                )
                if include_study:
                    model2_formula += " + C(study)"
                try:
                    model2 = self.run_linear_mixed_model(
                        df=df_endpoint_strata,
                        formula=model2_formula,
                        group_col='id',
                        reml=False
                    )
                    model2_results[(name, endpoint)] = model2
                    fe_params = model2.fe_params
                    conf_int = model2.conf_int().loc[fe_params.index]
                    pvals = model2.pvalues.loc[fe_params.index]
                    model2_table = pd.DataFrame({
                        'term': fe_params.index,
                        'estimate': fe_params.values,
                        'ci_lower': conf_int[0].values,
                        'ci_upper': conf_int[1].values,
                        'p_value': pvals.values
                    })
                    model2_table['dataset'] = name
                    model2_table['endpoint'] = endpoint
                    model2_summary_rows.append(model2_table)

                    design_info = model2.model.data.design_info
                    cov_params = model2.cov_params()
                    if isinstance(cov_params, pd.DataFrame):
                        cov_fe = cov_params.loc[fe_params.index, fe_params.index]
                    else:
                        cov_fe = cov_params[:len(fe_params), :len(fe_params)]
                    if include_study:
                        study_weights = (
                            df_endpoint_strata['study']
                            .value_counts(normalize=True, sort=False)
                            .to_dict()
                        )
                    else:
                        study_weights = {df_endpoint_strata['study'].iloc[0]: 1.0}

                    stratum_levels = df_endpoint_strata['cpep_stratum'].cat.categories
                    diff_rows = []
                    for level in stratum_levels:
                        if level not in df_endpoint_strata['cpep_stratum'].values:
                            continue
                        design_df = pd.DataFrame({
                            'treatment_arm': pd.Categorical(
                                ['control', 'active'] * len(study_weights),
                                categories=['control', 'active']
                            ),
                            'cpep_stratum': pd.Categorical(
                                [level] * (2 * len(study_weights)),
                                categories=stratum_levels,
                                ordered=True
                            ),
                            'study': list(study_weights.keys()) * 2
                        })
                        exog = patsy.build_design_matrices(
                            [design_info],
                            design_df,
                            return_type='dataframe'
                        )[0]
                        active_mask = design_df['treatment_arm'] == 'active'
                        control_mask = design_df['treatment_arm'] == 'control'
                        weights = np.array([study_weights[s] for s in design_df['study']])
                        weights_active = weights[active_mask]
                        weights_control = weights[control_mask]
                        weights_active = weights_active / weights_active.sum()
                        weights_control = weights_control / weights_control.sum()
                        x_active = (exog.to_numpy()[active_mask] * weights_active[:, None]).sum(axis=0)
                        x_control = (exog.to_numpy()[control_mask] * weights_control[:, None]).sum(axis=0)
                        mean_control = float(x_control @ fe_params.to_numpy())
                        mean_active = float(x_active @ fe_params.to_numpy())
                        se_control = float(np.sqrt(x_control @ cov_fe.to_numpy() @ x_control.T))
                        se_active = float(np.sqrt(x_active @ cov_fe.to_numpy() @ x_active.T))
                        model2_emmeans_rows.append({
                            'dataset': name,
                            'endpoint': endpoint,
                            'cpep_stratum': level,
                            'treatment_arm': 'control',
                            'emmeans': mean_control,
                            'ci_lower': mean_control - 1.96 * se_control,
                            'ci_upper': mean_control + 1.96 * se_control
                        })
                        model2_emmeans_rows.append({
                            'dataset': name,
                            'endpoint': endpoint,
                            'cpep_stratum': level,
                            'treatment_arm': 'active',
                            'emmeans': mean_active,
                            'ci_lower': mean_active - 1.96 * se_active,
                            'ci_upper': mean_active + 1.96 * se_active
                        })
                        diff = x_active - x_control
                        diff_est = float(diff @ fe_params.to_numpy())
                        diff_se = float(np.sqrt(diff @ cov_fe.to_numpy() @ diff.T))
                        z_stat = diff_est / diff_se if diff_se > 0 else np.nan
                        diff_p = 2 * norm.sf(abs(z_stat)) if np.isfinite(z_stat) else np.nan
                        diff_rows.append({
                            'cpep_stratum': level,
                            'treatment_minus_control': diff_est,
                            'ci_lower': diff_est - 1.96 * diff_se,
                            'ci_upper': diff_est + 1.96 * diff_se,
                            'p_value': diff_p
                        })
                    if diff_rows:
                        diff_df = pd.DataFrame(diff_rows)
                        diff_df['dataset'] = name
                        diff_df['endpoint'] = endpoint
                        model2_diff_rows.append(diff_df)
                except Exception as exc:
                    print(f"Model 2 failed for {name} {endpoint}: {exc}")

        if counts_rows:
            counts_all = pd.concat(counts_rows, ignore_index=True)
            counts_numeric = counts_all.select_dtypes(include='number').columns
            counts_all[counts_numeric] = counts_all[counts_numeric].round(2)
            counts_all = counts_all.sort_values(['dataset', 'cpep_stratum', 'treatment_arm'])
            for dataset_name in counts_all['dataset'].unique():
                counts_all[counts_all['dataset'] == dataset_name].to_csv(
                    f'./data/csv_results/lmm_results/cont_vs_strata/counts_by_stratum_{dataset_name}.csv',
                    index=False
                )
        if model1_summary_rows:
            model1_all = pd.concat(model1_summary_rows, ignore_index=True)
            model1_cols = ['dataset', 'endpoint', 'term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']
            model1_all = model1_all[[c for c in model1_cols if c in model1_all.columns]]
            model1_pvals = model1_all['p_value'] if 'p_value' in model1_all.columns else None
            model1_numeric = [c for c in model1_all.select_dtypes(include='number').columns if c != 'p_value']
            model1_all[model1_numeric] = model1_all[model1_numeric].round(2)
            if model1_pvals is not None:
                model1_all['p_value'] = model1_pvals.apply(
                    lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
                )
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = model1_all['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            model1_all['endpoint'] = pd.Categorical(
                model1_all['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            model1_all = model1_all.sort_values(['dataset', 'endpoint', 'term'])
            for dataset_name in model1_all['dataset'].unique():
                model1_all[model1_all['dataset'] == dataset_name].to_csv(
                    f'./data/csv_results/lmm_results/cont_vs_strata/model1_fixed_effects_{dataset_name}.csv',
                    index=False
                )
        if model2_summary_rows:
            model2_all = pd.concat(model2_summary_rows, ignore_index=True)
            model2_cols = ['dataset', 'endpoint', 'term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']
            model2_all = model2_all[[c for c in model2_cols if c in model2_all.columns]]
            model2_pvals = model2_all['p_value'] if 'p_value' in model2_all.columns else None
            model2_numeric = [c for c in model2_all.select_dtypes(include='number').columns if c != 'p_value']
            model2_all[model2_numeric] = model2_all[model2_numeric].round(2)
            if model2_pvals is not None:
                model2_all['p_value'] = model2_pvals.apply(
                    lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
                )
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = model2_all['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            model2_all['endpoint'] = pd.Categorical(
                model2_all['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            model2_all = model2_all.sort_values(['dataset', 'endpoint', 'term'])
            for dataset_name in model2_all['dataset'].unique():
                model2_all[model2_all['dataset'] == dataset_name].to_csv(
                    f'./data/csv_results/lmm_results/cont_vs_strata/model2_fixed_effects_{dataset_name}.csv',
                    index=False
                )
        if model2_diff_rows:
            model2_diffs_all = pd.concat(model2_diff_rows, ignore_index=True)
            diff_pvals = model2_diffs_all['p_value'] if 'p_value' in model2_diffs_all.columns else None
            diff_numeric = [c for c in model2_diffs_all.select_dtypes(include='number').columns if c != 'p_value']
            model2_diffs_all[diff_numeric] = model2_diffs_all[diff_numeric].round(2)
            if diff_pvals is not None:
                model2_diffs_all['p_value'] = diff_pvals.apply(
                    lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
                )
            diff_cols = [
                'dataset',
                'endpoint',
                'cpep_stratum',
                'treatment_minus_control',
                'ci_lower',
                'ci_upper',
                'p_value'
            ]
            model2_diffs_all = model2_diffs_all[[c for c in diff_cols if c in model2_diffs_all.columns]]
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = model2_diffs_all['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            model2_diffs_all['endpoint'] = pd.Categorical(
                model2_diffs_all['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            model2_diffs_all = model2_diffs_all.sort_values(['dataset', 'endpoint', 'cpep_stratum'])
            for dataset_name in model2_diffs_all['dataset'].unique():
                model2_diffs_all[model2_diffs_all['dataset'] == dataset_name].to_csv(
                    f'./data/csv_results/lmm_results/cont_vs_strata/model2_stratum_diffs_{dataset_name}.csv',
                    index=False
                )
        if model2_emmeans_rows:
            model2_emmeans_all = pd.DataFrame(model2_emmeans_rows)
            emmeans_numeric = model2_emmeans_all.select_dtypes(include='number').columns
            model2_emmeans_all[emmeans_numeric] = model2_emmeans_all[emmeans_numeric].round(2)
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = model2_emmeans_all['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            model2_emmeans_all['endpoint'] = pd.Categorical(
                model2_emmeans_all['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            model2_emmeans_all = model2_emmeans_all.sort_values(
                ['dataset', 'endpoint', 'cpep_stratum', 'treatment_arm']
            )
            for dataset_name in model2_emmeans_all['dataset'].unique():
                model2_emmeans_all[model2_emmeans_all['dataset'] == dataset_name].to_csv(
                    f'./data/csv_results/lmm_results/cont_vs_strata/model2_emmeans_{dataset_name}.csv',
                    index=False
                )

    def lmm_cpep_strata_arms_vs_reference(self) -> None:
        """
        Compare treatment-arm endpoint estimates within each C-peptide stratum against references.

        For each T1D C-peptide stratum, the function combines active/control observations with
        healthy and established reference cohorts, fits mixed-effects models, and exports group
        counts plus predefined contrasts against the reference groups.

        LMM formula per endpoint within each stratum:
            - Fixed-effects formula: `endpoint ~ C(group)`
            - Random intercept: supplied separately via `group_col='id'`
            - Equivalent mixed-model notation:
              `endpoint ~ C(group) + (1 | id)`

        Contrasts are reported for:
            - `active - established`
            - `control - established`
            - `active - healthy`
            - `control - healthy`

        Args:
            None

        Returns:
            None
        """
        dict_dfs = self.datasets.copy()

        all_dict = {k: v.compute() for k, v in dict_dfs.items() if k in ALL_STUDIES}
        df_t1d = pd.concat(all_dict.values(), ignore_index=True)
        df_established = self.get_dataset('hupa_ucm')
        df_healthy = self.get_dataset('jaeb_healthy')

        t1d_endpoints = self.get_taylor_patient_endpoints(
            df_t1d,
            metabolic_endpoints=CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'cpep_auc', 'treatment_arm', 'insulin_delivery', 'gmi', 'hb_a1c'})
        healthy_endpoints = self.get_taylor_patient_endpoints(
            df_healthy,
            metabolic_endpoints=CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'cpep_auc', 'treatment_arm', 'insulin_delivery', 'gmi', 'hb_a1c'})
        established_endpoints = self.get_taylor_patient_endpoints(
            df_established,
            metabolic_endpoints=CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'cpep_auc', 'treatment_arm', 'insulin_delivery', 'gmi', 'hb_a1c'})
        
        treatment_source = 'treatment_arm'
        t1d_endpoints['treatment_arm'] = (
            t1d_endpoints[treatment_source]
            .astype('string')
            .str.strip()
            .str.lower()
        )
        t1d_endpoints.loc[
            t1d_endpoints['treatment_arm'].isin(TREATMENT_GROUP_1),
            'treatment_arm'
        ] = 'control'
        t1d_endpoints.loc[
            t1d_endpoints['treatment_arm'].isin(TREATMENT_GROUP_2),
            'treatment_arm'
        ] = 'active'
        t1d_endpoints['group'] = t1d_endpoints['treatment_arm']
        healthy_endpoints['group'] = 'healthy'
        established_endpoints['group'] = 'established'

        cpep_bins = [0.0, 0.2, 0.5, 0.8, 1.0]
        t1d_endpoints['cpep_stratum'] = pd.cut(
            t1d_endpoints['cpep_auc'],
            bins=cpep_bins,
            labels=[1, 2, 3, 4],
            include_lowest=True,
            right=False
        ).astype('Int64')
        t1d_endpoints.dropna(subset=['cpep_stratum'], inplace=True)
        group_categories = ['control', 'active', 'healthy', 'established']
        contrasts_rows = []
        group_counts_rows = []
        endpoints = set(CGM_CORE_ENDPOINTS) | {'gmi', 'hb_a1c'}
        for cpep_stratum in t1d_endpoints['cpep_stratum'].unique():
            stratum_df = t1d_endpoints[t1d_endpoints['cpep_stratum'] == cpep_stratum]
            stratum_df = pd.concat([stratum_df, healthy_endpoints], ignore_index=True)
            stratum_df = pd.concat([stratum_df, established_endpoints], ignore_index=True)
            stratum_df['group'] = pd.Categorical(
                stratum_df['group'],
                categories=group_categories
            )

            counts_df = stratum_df.dropna(subset=['group'])
            if not counts_df.empty:
                participants_counts = (
                    counts_df.groupby('group')['id']
                    .nunique()
                    .rename('participants')
                    .reset_index()
                )
                observations_counts = (
                    counts_df.groupby('group')
                    .size()
                    .rename('observations')
                    .reset_index()
                )
                counts_summary = pd.merge(
                    participants_counts,
                    observations_counts,
                    on='group',
                    how='outer'
                )
                counts_summary['cpep_stratum'] = int(cpep_stratum)
                group_counts_rows.append(counts_summary)

            for endpoint in endpoints:
                df_endpoint = stratum_df[['id', 'group', endpoint]].dropna(subset=['group', endpoint])
                if df_endpoint.empty:
                    continue
                df_endpoint['group'] = pd.Categorical(
                    df_endpoint['group'],
                    categories=group_categories
                )
                present_groups = df_endpoint['group'].dropna().unique().tolist()
                if len(present_groups) < 2:
                    continue
                try:
                    model = self.run_linear_mixed_model(
                        df=df_endpoint,
                        formula=f"{endpoint} ~ C(group)",
                        group_col='id',
                        reml=False
                    )
                except Exception as exc:
                    print(f"Reference model failed (stratum {cpep_stratum}, {endpoint}): {exc}")
                    continue

                fe_params = model.fe_params
                cov_params = model.cov_params()
                if isinstance(cov_params, pd.DataFrame):
                    cov_fe = cov_params.loc[fe_params.index, fe_params.index]
                else:
                    cov_fe = cov_params[:len(fe_params), :len(fe_params)]
                design_info = model.model.data.design_info

                def _design_row(group_name: str) -> np.ndarray:
                    design_df = pd.DataFrame({
                        'group': pd.Categorical([group_name], categories=group_categories)
                    })
                    exog = patsy.build_design_matrices(
                        [design_info],
                        design_df,
                        return_type='dataframe'
                    )[0]
                    return exog.to_numpy().ravel()

                comparisons = [
                    ('active', 'established'),
                    ('control', 'established'),
                    ('active', 'healthy'),
                    ('control', 'healthy')
                ]
                for group_a, group_b in comparisons:
                    if group_a not in present_groups or group_b not in present_groups:
                        continue
                    x_a = _design_row(group_a)
                    x_b = _design_row(group_b)
                    contrast = x_a - x_b
                    estimate = float(contrast @ fe_params.to_numpy())
                    se = float(np.sqrt(contrast @ cov_fe.to_numpy() @ contrast.T))
                    if se == 0 or np.isnan(se):
                        p_value = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                    else:
                        z_score = estimate / se
                        p_value = 2 * norm.sf(abs(z_score))
                        ci_lower = estimate - 1.96 * se
                        ci_upper = estimate + 1.96 * se
                    contrasts_rows.append({
                        'cpep_stratum': int(cpep_stratum),
                        'endpoint': endpoint,
                        'contrast': f"{group_a} - {group_b}",
                        'estimate': estimate,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_value': p_value
                    })

            stratum_df.to_csv(f'./data/csv_results/lmm_results/reference_group_comparision/df_stratum{cpep_stratum}.csv', index=False)

        if group_counts_rows:
            counts_out = pd.concat(group_counts_rows, ignore_index=True)
            counts_out[['participants', 'observations']] = counts_out[['participants', 'observations']].fillna(0).astype(int)
            counts_out = counts_out[['cpep_stratum', 'group', 'participants', 'observations']]
            counts_out = counts_out.sort_values(['cpep_stratum', 'group'])
            counts_out.to_csv(
                './data/csv_results/lmm_results/reference_group_comparision/group_counts_by_stratum.csv',
                index=False
            )

        if contrasts_rows:
            contrasts_df = pd.DataFrame(contrasts_rows)
            numeric_cols = ['estimate', 'ci_lower', 'ci_upper', 'p_value']
            contrasts_df[numeric_cols] = contrasts_df[numeric_cols].round(2)
            contrasts_df['p_value'] = contrasts_df['p_value'].apply(
                lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
            )
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = contrasts_df['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            contrasts_df['endpoint'] = pd.Categorical(
                contrasts_df['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            contrasts_df = contrasts_df.sort_values(['endpoint', 'cpep_stratum', 'contrast'])
            contrasts_df.to_csv(
                './data/csv_results/lmm_results/reference_group_comparision/group_contrasts_by_stratum.csv',
                index=False
            )

        return None
 
    def lmm_cpep_strata_time_difference(
        self,
        df_t1d: pd.DataFrame | dd.DataFrame,
        *,
        full_formula: str,
        reduced_formula: str,
        trend_formula: str,
        group_col: str
    ) -> None:
        """
        Fit mixed-effects models for C-peptide strata while adjusting for study and follow-up time.

        Uses pooled T1D study summaries, includes both study and continuous follow-up time in the fixed
        effects, adds a participant-level random intercept via `group_col`, and exports adjusted endpoint
        means together with omnibus and linear-trend tests across C-peptide strata.

        Args:
            df_t1d: Pooled study-level dataframe for the entire cohort. Must include
                subject-level records and a `study` column.
            full_formula: Full fixed-effects formula evaluated per endpoint.
            reduced_formula: Reduced fixed-effects formula evaluated per endpoint.
            trend_formula: Trend fixed-effects formula evaluated per endpoint.
            group_col: Column used for the participant-level random intercept grouping structure.

        Returns:
            None
        """
        if df_t1d is None:
            raise ValueError("`df_t1d` cannot be None for the pooled time-difference LMM.")

        if isinstance(df_t1d, dd.DataFrame):
            df_t1d = df_t1d.compute()
        else:
            df_t1d = df_t1d.copy()

        if df_t1d.empty:
            raise ValueError("`df_t1d` must be a non-empty pooled dataframe.")

        if 'study' not in df_t1d.columns:
            raise ValueError("`df_t1d` must include a 'study' column.")

        for formula_name, formula in {
            'full_formula': full_formula,
            'reduced_formula': reduced_formula,
            'trend_formula': trend_formula
        }.items():
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError(f"`{formula_name}` must be a non-empty formula string.")

        if not isinstance(group_col, str) or not group_col.strip():
            raise ValueError("`group_col` must be a non-empty string.")

        t1d_endpoints = set(CGM_CORE_ENDPOINTS) | {'gmi', 'hb_a1c'}
        df_t1d_time_bins = self.get_taylor_patient_endpoints(
            df_t1d,
            CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'cpep_auc', 'study', 'gmi', 'hb_a1c'}
        )
        if 'study' in df_t1d_time_bins.columns:
            df_t1d_time_bins['study'] = df_t1d_time_bins['study'].astype('category')
            
        time_bin_to_months = {
            'Baseline': 0.0,
            'Week 6': 1.5,
            'Month 3': 3.0,
            'Month 6': 6.0,
            'Month 9': 9.0,
            'Month 12': 12.0,
            'Month 15': 15.0,
            'Month 18': 18.0,
            'Month 21': 21.0,
            'Month 24': 24.0
        }
        if 'time_bin' in df_t1d_time_bins.columns:
            df_t1d_time_bins['time_months'] = df_t1d_time_bins['time_bin'].map(time_bin_to_months)
        cpep_bins = [0.0, 0.2, 0.5, 0.8, np.inf]
        df_t1d_time_bins['cpep_stratum'] = pd.cut(
            df_t1d_time_bins['cpep_auc'],
            bins=cpep_bins,
            labels=[1, 2, 3, 4],
            include_lowest=True,
            right=False
        ).astype('Int64')
            
        df_t1d_time_bins['cpep_stratum'] = pd.Categorical(
            df_t1d_time_bins['cpep_stratum'],
            categories=[1, 2, 3, 4],
            ordered=True
        )

        desc_df = df_t1d_time_bins.dropna(subset=['cpep_stratum'])
        if not desc_df.empty:
            participants_per_stratum = (
                desc_df.groupby('cpep_stratum')['id']
                .nunique()
            )
            observations_per_stratum = (
                desc_df.groupby('cpep_stratum')
                .size()
            )
            print("Unique participants per C-peptide stratum:")
            print(participants_per_stratum.to_string())
            print("\nCGM observations per C-peptide stratum:")
            print(observations_per_stratum.to_string())

        adjusted_means_df, lrt_df, trend_df = self.run_cpep_strata_time_lmm(
            df=df_t1d_time_bins,
            endpoints=t1d_endpoints,
            full_formula=full_formula,
            reduced_formula=reduced_formula,
            trend_formula=trend_formula,
            group_col=group_col,
            study_col='study',
            time_col='time_months',
            stratum_col='cpep_stratum',
            strict=True
        )

        if not adjusted_means_df.empty:
            lrt_map = dict(zip(lrt_df['endpoint'], lrt_df['p_value'])) if not lrt_df.empty else {}
            trend_map = dict(zip(trend_df['endpoint'], trend_df['trend_p_value'])) if not trend_df.empty else {}
            adjusted_means_df['p_value_omnibus'] = adjusted_means_df['endpoint'].map(lrt_map)
            adjusted_means_df['p_value_trend'] = adjusted_means_df['endpoint'].map(trend_map)
            numeric_cols = adjusted_means_df.select_dtypes(include='number').columns
            adjusted_means_df[numeric_cols] = adjusted_means_df[numeric_cols].round(2)
            for col in ['p_value_omnibus', 'p_value_trend']:
                if col in adjusted_means_df.columns:
                    adjusted_means_df[col] = adjusted_means_df[col].apply(
                        lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (f"{p:.3f}" if pd.notna(p) else p)
                    )
            preferred_endpoints = ['TIR', 'TITR', 'hb_a1c', 'gmi']
            endpoints_present = adjusted_means_df['endpoint'].dropna().unique().tolist()
            ordered_endpoints = (
                [e for e in preferred_endpoints if e in endpoints_present] +
                sorted([e for e in endpoints_present if e not in preferred_endpoints])
            )
            adjusted_means_df['endpoint'] = pd.Categorical(
                adjusted_means_df['endpoint'],
                categories=ordered_endpoints,
                ordered=True
            )
            adjusted_means_df = adjusted_means_df.sort_values(['endpoint', 'cpep_stratum'])
            adjusted_means_df.to_csv(
                './data/csv_results/lmm_results/cpepstrata_study_time_adjusted_means.csv',
                index=False
            )
        if not lrt_df.empty:
            print("\nLikelihood ratio tests (full vs reduced):")
            print(lrt_df.to_string(index=False))
        if not trend_df.empty:
            print("\nLinear trend tests (cpep_stratum_num):")
            print(trend_df.to_string(index=False))

        return None

    def lmm_cpep_x_treat(self, df_name, df) -> None:
        """
        Estimate simple C-peptide slopes within treatment arms from Model 1 mixed models.

        For each CGM endpoint in the supplied dataset, this function is intended to fit
        or reuse the continuous C-peptide treatment-interaction model, then extract the
        C-peptide slope separately within each treatment arm.

        Model 1 fixed-effects formula per endpoint:
            - `endpoint ~ C(treatment_arm) + log_cpep_auc + C(treatment_arm):log_cpep_auc`
            - If multiple studies are present, add `+ C(study)`
            - Random intercept: supplied separately via `group_col='id'`
            - Equivalent mixed-model notation:
              `endpoint ~ ... + (1 | id)`

        Simple slopes:
            - Control arm slope: `beta(log_cpep_auc)`
            - Active arm slope:
              `beta(log_cpep_auc) + beta(C(treatment_arm)[T.1]:log_cpep_auc)`

        Treatment arms are coded as:
            - 0: control, using values in `TREATMENT_GROUP_1`
            - 1: active, using values in `TREATMENT_GROUP_2`

        Report for each dataset, endpoint, and treatment arm:
            - Slope estimate
            - 95% Wald confidence interval
            - Wald p-value

        Args:
            df_name: Dataset label used in outputs, such as `cloud`, `clvr`, or
                `cloud_clvr`.
            df: Input dataset containing CGM data and model covariates.

        Returns:
            None
        """
        df = df.copy()
        treatment_arm = df['treatment_arm'].astype('string').str.strip().str.lower()
        df['treatment_arm'] = pd.Series(
            np.select(
                [
                    treatment_arm.isin(TREATMENT_GROUP_1),
                    treatment_arm.isin(TREATMENT_GROUP_2)
                ],
                [0, 1],
                default=np.nan
            ),
            index=df.index
        ).astype('Int64')
        if df['treatment_arm'].dropna().nunique() < 2:
            raise ValueError(
                f"{df_name} requires both control and active treatment arms after recoding."
            )

        endpoints = self.get_taylor_patient_endpoints(
            df,
            metabolic_endpoints=CGM_CORE_ENDPOINTS,
            by_time_bin=True,
            include_cols={'study', 'cpep_auc', 'treatment_arm', 'insulin_delivery', 'gmi', 'hb_a1c'})
        endpoints['treatment_arm'] = pd.to_numeric(
            endpoints['treatment_arm'],
            errors='coerce'
        ).astype('Int64')
        endpoints = endpoints.dropna(subset=['cpep_auc'])
        if endpoints.empty:
            raise ValueError(f"{df_name} has no endpoint rows with non-missing cpep_auc.")
        if endpoints['treatment_arm'].dropna().nunique() < 2:
            raise ValueError(
                f"{df_name} endpoints require both control and active treatment arms."
            )
        endpoints['log_cpep_auc'] = np.log(endpoints['cpep_auc'] + 1)

        model1_results = {}
        model1_summary_rows = []
        simple_slope_rows = []
        agp_endpoints = sorted(set(CGM_CORE_ENDPOINTS) | {'gmi', 'hb_a1c'})
        include_study = (
            'study' in endpoints.columns and
            endpoints['study'].nunique(dropna=True) > 1
        )

        for endpoint in agp_endpoints:
            if endpoint not in endpoints.columns:
                raise ValueError(f"{df_name} endpoints are missing required endpoint '{endpoint}'.")

            model_cols = ['id', 'treatment_arm', 'log_cpep_auc', endpoint]
            if include_study:
                model_cols.append('study')
            df_endpoint = endpoints[model_cols].dropna(
                subset=['id', 'treatment_arm', 'log_cpep_auc', endpoint]
            ).copy()
            if df_endpoint.empty:
                raise ValueError(
                    f"{df_name} {endpoint} has no complete rows for Model 1."
                )
            if df_endpoint['treatment_arm'].nunique(dropna=True) < 2:
                raise ValueError(
                    f"{df_name} {endpoint} requires both treatment arms for Model 1."
                )
            df_endpoint['treatment_arm'] = df_endpoint['treatment_arm'].astype(int)

            model1_formula = (
                f"{endpoint} ~ C(treatment_arm) + log_cpep_auc + "
                "C(treatment_arm):log_cpep_auc"
            )
            if include_study:
                model1_formula += " + C(study)"

            model1 = self.run_linear_mixed_model(
                df=df_endpoint,
                formula=model1_formula,
                group_col='id',
                reml=False
            )
            model1_results[(df_name, endpoint)] = model1
            fe_params = model1.fe_params
            conf_int = model1.conf_int().loc[fe_params.index]
            pvals = model1.pvalues.loc[fe_params.index]
            model1_table = pd.DataFrame({
                'term': fe_params.index,
                'estimate': fe_params.values,
                'ci_lower': conf_int[0].values,
                'ci_upper': conf_int[1].values,
                'p_value': pvals.values
            })
            model1_table['dataset'] = df_name
            model1_table['endpoint'] = endpoint
            model1_summary_rows.append(model1_table)

            cov_params = model1.cov_params()
            if isinstance(cov_params, pd.DataFrame):
                cov_fe = cov_params.loc[fe_params.index, fe_params.index]
            else:
                cov_fe = cov_params[:len(fe_params), :len(fe_params)]

            cpep_term = 'log_cpep_auc'
            interaction_terms = [
                'C(treatment_arm)[T.1]:log_cpep_auc',
                'log_cpep_auc:C(treatment_arm)[T.1]'
            ]
            interaction_term = next(
                (term for term in interaction_terms if term in fe_params.index),
                None
            )
            if cpep_term not in fe_params.index:
                raise ValueError(
                    f"{df_name} {endpoint} Model 1 is missing '{cpep_term}'."
                )
            if interaction_term is None:
                raise ValueError(
                    f"{df_name} {endpoint} Model 1 is missing the treatment-by-C-peptide interaction."
                )

            for treatment_arm, contrast_terms in [
                ('control', {cpep_term: 1.0}),
                ('active', {cpep_term: 1.0, interaction_term: 1.0})
            ]:
                contrast = np.zeros(len(fe_params))
                for term, weight in contrast_terms.items():
                    contrast[fe_params.index.get_loc(term)] = weight

                estimate = float(contrast @ fe_params.to_numpy())
                variance = float(contrast @ cov_fe.to_numpy() @ contrast.T)
                if variance <= 0 or np.isnan(variance):
                    se = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan
                    p_value = np.nan
                    status = 'invalid_standard_error'
                else:
                    se = float(np.sqrt(variance))
                    z_score = estimate / se
                    p_value = 2 * norm.sf(abs(z_score))
                    ci_lower = estimate - 1.96 * se
                    ci_upper = estimate + 1.96 * se
                    status = 'ok'

                simple_slope_rows.append({
                    'dataset': df_name,
                    'endpoint': endpoint,
                    'treatment_arm': treatment_arm,
                    'estimate': estimate,
                    'standard_error': se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'status': status
                })

        if not model1_summary_rows:
            raise ValueError(f"{df_name} produced no Model 1 fixed-effect summaries.")
        if not simple_slope_rows:
            raise ValueError(f"{df_name} produced no simple-slope estimates.")

        output_dir = Path('./data/csv_results/lmm_results/cpep_slopes_by_treatment')
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

        model1_summary_df = pd.concat(model1_summary_rows, ignore_index=True)
        model1_summary_df.to_csv(
            output_dir / f'{df_name}_model1_fixed_effects.csv',
            index=False
        )

        simple_slope_df = pd.DataFrame(simple_slope_rows)
        simple_slope_df.to_csv(
            output_dir / f'{df_name}_simple_slopes.csv',
            index=False
        )

        return None

if __name__ == '__main__':
    data_analysis = DataAnalysis()

    ''' Loading datasets '''
    # df_bandit = data_analysis.get_dataset('bandit')
    df_cloud = data_analysis.get_dataset('cloud')
    df_clvr = data_analysis.get_dataset('clvr')
    # df_defend = data_analysis.get_dataset('defend')
    # df_diagnode = data_analysis.get_dataset('diagnode')
    # df_gskalb = data_analysis.get_dataset('gskalb')
    # df_itx = data_analysis.get_dataset('itx')
    # df_jaeb_healthy = data_analysis.get_dataset('jaeb_healthy')
    # df_jaeb_t1d = data_analysis.get_dataset('jaeb_t1d')

    # df_entire_data = pd.concat(
    #     [
    #         df_bandit.assign(study='bandit'),
    #         df_cloud.assign(study='cloud'),
    #         df_clvr.assign(study='clvr'),
    #         df_defend.assign(study='defend'),
    #         df_diagnode.assign(study='diagnode'),
    #         df_gskalb.assign(study='gskalb'),
    #         df_jaeb_t1d.assign(study='jaeb_t1d')
    #     ],
    #     ignore_index=True
    # )

    df_cloud_clvr = pd.concat(
        [
            df_cloud.assign(study='cloud'),
            df_clvr.assign(study='clvr')
        ],
        ignore_index=True
    )

    ''' Printing dataset summaries '''
    # data_analysis.print_summaries()

    '''Printing data inventory and metadata: '''
    # data_analysis.best_cgm_pct_wear_and_days()
    # data_analysis.check_cgm_pct_wear(ALL_STUDIES)
    # data_analysis.print_AGP()
    # data_analysis.plot_AGP_random_subject_all_windows()
    # data_analysis.print_feature_availability_table()
    # data_analysis.print_valid_durations()
    # data_analysis.print_feature_histograms()
    # data_analysis.print_feature_isi_histograms()
    # data_analysis.print_characteristics_table_baseline()
    # data_analysis.print_good_days()
    # data_analysis.print_insulin_inventory()
    # data_analysis.print_metadata()

    '''Hypoglycemia rates statistics '''
    # # print(hypo_rate_statistics)

    '''Hypothesis test: Length of Line'''
    # data_analysis.hypothesis_test_LoL()

    '''Hypothesis test: CGM core endpoints GENERAL'''
    # data_analysis.ht_CGM_general()

    '''Hypothesis test: CGM core endpoints TIME BINS'''
    # data_analysis.ht_CGM_time_bins()

    '''Hypothesis test: CGM core endpoints vs Time (seperated by treatment arms)'''
    # data_analysis.ht_CGM_treatment_arms()

    '''Hypothesis test: CGM core endpoints vs Time (seperated by daytime type)'''
    # data_analysis.ht_CGM_daytime_type()

    '''Hypothesis test: Correlation CGM and CPEP AUC, CGM and Beta2 Score'''
    # data_analysis.ht_clinical_features()

    '''Hypothesis test: Correlation CGM and CPEP AUC, CGM and Beta2 Score (seperated by treatment arms)'''
    # data_analysis.ht_clinical_features_treatment_arms()

    '''Hypothesis test: Correlation CGM and CPEP AUC, CGM and Beta2 Score (seperated by daytime type)'''
    # data_analysis.ht_clinical_features_daytime_type()

    '''Hypothesis test: CGM core endpoints difference among CPEP strata'''
    # data_analysis.ht_cpep_strata()

    '''Boxplot hypoglycemia rates'''
    # data_analysis.plot_analyzed_features_general('prc_hypoglycemia_level_1')
    # data_analysis.plot_analyzed_features_general('prc_hypoglycemia_level_2')
    # data_analysis.plot_analyzed_features_general('prc_cl_sig_hypo')

    '''Taylor Analysis: '''
    # data_analysis.taylor_analysis()
    # data_analysis.combine_pngs('./data/graphs/feature_analysis/AGP/cloud/beta2_score/', './data/graphs/feature_analysis/AGP/cloud/beta2_score/combined.png', cols=2)

    '''Entire Cohor: Control vs Treatment (Fasting Glucose, GMI, HbA1C, Beta2, Beta3)'''
    # data_analysis.a1c_vs_gmi_treatment_arms()

    '''
    Linear mixed effect modelling for c-peptide strata comparisons:
        - Full fixed-effects formula: `endpoint ~ C(cpep_stratum) + C(study)`
        - Reduced fixed-effects formula: `endpoint ~ C(study)`
        - Trend fixed-effects formula: `endpoint ~ cpep_stratum_num + C(study)`
        - Random intercept: supplied separately via `group_col='id'`
        - Equivalent mixed-model notation: `endpoint ~ ... + (1 | id)`
    '''
    # data_analysis.lmm_cpep_between_strata()

    '''
    LMM formulas per endpoint:
        - Model 1 fixed-effects formula:
          `endpoint ~ C(treatment_arm) + log_cpep_auc + C(treatment_arm):log_cpep_auc`
        - Model 2 fixed-effects formula:
          `endpoint ~ C(treatment_arm) + C(cpep_stratum) + C(treatment_arm):C(cpep_stratum)`
        - If multiple studies are present, both models add `+ C(study)`
        - Random intercept: supplied separately via `group_col='id'`
        - Equivalent mixed-model notation: `endpoint ~ ... + (1 | id)`
    '''
    # data_analysis.lmm_cpep_cont_vs_strata()

    '''
    LMM formula per endpoint within each stratum:
        - Fixed-effects formula: `endpoint ~ C(group)`
        - Random intercept: supplied separately via `group_col='id'`
        - Equivalent mixed-model notation: `endpoint ~ C(group) + (1 | id)`

    Contrasts are reported for:
        - `active - established`
        - `control - established`
        - `active - healthy`
        - `control - healthy`
    '''
    # data_analysis.lmm_cpep_strata_arms_vs_reference()

    '''
    LMM formulas per endpoint for entire data:
        - Full fixed-effects formula: `endpoint ~ C(cpep_stratum) + C(study) + time_months`
        - Reduced fixed-effects formula: `endpoint ~ C(study) + time_months`
        - Trend fixed-effects formula: `endpoint ~ cpep_stratum_num + C(study) + time_months`
        - Random intercept: supplied separately via `group_col='id'`
        - Equivalent mixed-model notation: `endpoint ~ ... + (1 | id)`
    '''
    # data_analysis.lmm_cpep_strata_time_difference(
    #     df_entire_data,
    #     full_formula="endpoint ~ C(cpep_stratum) + C(study) + time_months",
    #     reduced_formula="endpoint ~ C(study) + time_months",
    #     trend_formula="endpoint ~ cpep_stratum_num + C(study) + time_months",
    #     group_col='id'
    # )

    '''
    Simple C-peptide slopes within treatment arms for CLOUD, CLVR, and pooled CLOUD+CLVR:
        - Model 1 fixed-effects formula:
          `endpoint ~ C(treatment_arm) + log_cpep_auc + C(treatment_arm):log_cpep_auc`
        - Pooled CLOUD+CLVR models add `+ C(study)`
        - Random intercept: supplied separately via `group_col='id'`
        - Treatment coding: control = 0, active = 1
        - Control slope: `beta(log_cpep_auc)`
        - Active slope: `beta(log_cpep_auc) + beta(C(treatment_arm)[T.1]:log_cpep_auc)`
    '''
    data_analysis.lmm_cpep_x_treat('cloud', df_cloud)
    data_analysis.lmm_cpep_x_treat('clvr', df_clvr)
    data_analysis.lmm_cpep_x_treat('cloud_clvr', df_cloud_clvr)

    '''
    LMM formulas per endpoint for CLOUD/CLVR:
        - Full fixed-effects formula: `endpoint ~ C(cpep_stratum) + C(study) + time_months`
        - Reduced fixed-effects formula: `endpoint ~ C(study) + time_months`
        - Trend fixed-effects formula: `endpoint ~ cpep_stratum_num + C(study) + time_months`
        - Random intercept: supplied separately via `group_col='id'`
        - Equivalent mixed-model notation: `endpoint ~ ... + (1 | id)`
    '''
    # data_analysis.lmm_cpep_strata_time_difference(
    #     df_cloud_clvr,
    #     full_formula="endpoint ~ C(cpep_stratum) + C(study) + time_months",
    #     reduced_formula="endpoint ~ C(study) + time_months",
    #     trend_formula="endpoint ~ cpep_stratum_num + C(study) + time_months",
    #     group_col='id'
    # )
