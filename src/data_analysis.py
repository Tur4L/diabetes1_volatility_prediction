import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, kruskal, spearmanr
from itertools import combinations
import seaborn as sns
import random
import warnings
import matplotlib.pyplot as plt
import importlib
import os
from itx_package import ITXData

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
scaler = MinMaxScaler()
itx_dataloader = ITXData()

DATASET_CONFIG = {
    "cloud": {"csv_folder": "./data/cloud", "preprocess": "preprocess_cloud"},
    "clvr": {"csv_folder": "./data/clvr", "preprocess": "preprocess_clvr"},
    "defend": {'csv_folder': './data/defend', "preprocess": 'preprocess_defend'},
    "diagnode": {"csv_folder": "./data/diagnode", "preprocess": "preprocess_diagnode"},
    "gskalb": {'csv_folder': './data/gskalb', 'preprocess': 'preprocess_gskalb'},
    "itx": {"csv_folder": "./data/itx", "preprocess": "preprocess_itx"},
    "jaeb_healthy": {"csv_folder": "./data/jaeb_healthy", "preprocess": "preprocess_jaeb_healthy"},
    "jaeb_t1d": {"csv_folder": "./data/jaeb_t1d", "preprocess": "preprocess_jaeb_t1d"},
}

class DataAnalysis:
    def __init__(self):
        """Initializes and loads all datasets dynamically."""
        self.datasets = {}
        self.datasets_summary = {}
        self.analyzed_datasets = {}
        self.load_data()

    def load_data(self): 
        """Loads, cleans, and normalizes all datasets from CSV files."""

        print('\n\tLoading, summarizing, and analyzing datasets...')
        for dataset_name, config in DATASET_CONFIG.items():
            csv_folder = config["csv_folder"]
            if not os.path.exists(csv_folder):
                print(f"Warning: {csv_folder} not found. Skipping {dataset_name}.")
                continue

            print(f"Loading {dataset_name} from {csv_folder}...")

            # if os.path.exists(f'{csv_folder}/df_final.csv') and os.path.exists(f'{csv_folder}/df_summary.csv'):
            #     df = pd.read_csv(f'{csv_folder}/df_final.csv')
            #     df_summary = pd.read_csv(f'{csv_folder}/df_summary.csv')

            # else: 
            df, df_summary = self._preprocess_data(csv_folder, config["preprocess"])
            df.to_csv(f'{csv_folder}/df_final.csv', index=False)
            df_summary.to_csv(f'{csv_folder}/df_summary.csv', index=False)
                

            self.datasets[dataset_name] = df
            self.datasets_summary[dataset_name] = df_summary
            print('Loading and Summary Done.')

            if os.path.exists(f'{csv_folder}/df_analysis.csv'):
                df_analysis = pd.read_csv(f'{csv_folder}/df_analysis.csv')

            else:
                df_analysis = self.statistical_analysis(csv_folder, df)

            self.analyzed_datasets[dataset_name] = df_analysis
            print('Analysis Done.\n')

        print('\tLoading Done.\n')
        
    def _preprocess_data(self, csv_folder, preprocess_func_name):
        """
        Dynamically loads the preprocessing function from either the current class
        or the ITX package.
        """
        preprocess_func = getattr(self, preprocess_func_name, None)

        if preprocess_func:
            return preprocess_func(csv_folder)

    def preprocess_cloud(self, csv_folder):

        #CGM Data
        """Custom preprocessing for CLOUD dataset."""
        df_cgm = pd.read_csv(f'{csv_folder}/original/CloudAbbottCGM.txt', sep="|")
        df_cgm.rename(columns={'PtID':'id', 'DeviceDtTm':'timestamp', 'Glucose': 'glucose mmol/l'}, inplace=True)
        df_cgm.drop_duplicates(subset=['RecID','id'], inplace=True)

        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%m/%d/%Y %I:%M:%S %p", errors ='coerce')
        df_cgm.dropna(subset=['timestamp'], inplace=True)
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds())
        
        df_cgm = df_cgm[['id', 'timestamp', 'timestamp_seconds', 'glucose mmol/l' ]]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #Extra features
        df_age = pd.read_csv(f'{csv_folder}/original/PtRoster.txt', sep="|")
        df_age.rename(columns={'PtID':'id','AgeAtConsent': 'age'}, inplace=True)
        df_age.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_age.sort_values(by=['id'], ascending=True, inplace=True)
        df_age.dropna(subset=['age'], inplace=True)
        df_randdt = df_age[['id', 'RandDt']]
        df_randdt.loc[:, 'RandDt'] = pd.to_datetime(df_age['RandDt'], format="%m/%d/%Y %I:%M:%S %p", errors ='coerce')
        df_age = df_age[['id', 'age']]
        df_age.to_csv(f'{csv_folder}/df_age.csv', index=False)

        df_hba1c_local = pd.read_csv(f'{csv_folder}/original/CloudDiabLocalHbA1c.txt', sep="|")
        df_hba1c_local.rename(columns={'PtID':'id', 'Visit':'visit', 'HbA1cTestRes': 'hb_a1c_local'}, inplace=True)
        df_hba1c_local['HbA1cTestDt'] = pd.to_datetime(df_hba1c_local['HbA1cTestDt'])
        df_hba1c_local.sort_values(by=['id','HbA1cTestDt'], ascending=True, inplace=True)
        df_hba1c_local.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_hba1c_local.dropna(subset=['hb_a1c_local'], inplace=True)
        df_hba1c_local = df_hba1c_local[['id','visit','HbA1cTestDt', 'hb_a1c_local']]
        df_hba1c_local.to_csv(f'{csv_folder}/df_hba1c_local.csv',index=False)

        df_hba1c_cap = pd.read_csv(f'{csv_folder}/original/CloudLabDataHbA1cCap.txt', sep="|")
        df_hba1c_cap.rename(columns={'PtID':'id', 'Visit':'visit'}, inplace=True)
        df_hba1c_cap = df_hba1c_cap[~df_hba1c_cap['VisitCollected'].isin(['Visit15', 'Visit17'])]
        df_hba1c_cap = pd.merge(df_hba1c_cap, df_randdt, on=['id'])
        df_hba1c_cap['MonthsOffset'] = (df_hba1c_cap['VisitCollected'].str.extract(r'(\d+)').fillna(0).astype(int))
        df_hba1c_cap['VisitDate'] = df_hba1c_cap.apply(lambda row: row['RandDt'] + pd.DateOffset(months=row['MonthsOffset']), axis=1)
        df_hba1c_cap['VisitDate'] = pd.to_datetime(df_hba1c_cap['VisitDate'])
        df_hba1c_cap['hb_a1c_cap'] = (df_hba1c_cap['HbA1cMMol'] + 2.152)/ 10.929
        df_hba1c_cap.sort_values(by=['id','VisitDate'], ascending=True, inplace=True)
        df_hba1c_cap = df_hba1c_cap[['id', 'VisitCollected', 'VisitDate', 'hb_a1c_cap']]
        df_hba1c_cap.dropna(subset=['VisitDate'], inplace=True)
        df_hba1c_cap.to_csv(f'{csv_folder}/df_hba1c_cap.csv',index=False)
        cap_ids = df_hba1c_cap['id'].unique()

        df_hba1c_ven = pd.read_csv(f'{csv_folder}/original/CloudLabDataHbA1cVen.txt', sep="|")
        df_hba1c_ven.rename(columns={'PtID':'id', 'Visit':'visit'}, inplace=True)
        df_hba1c_ven = df_hba1c_ven[~df_hba1c_ven['VisitCollected'].isin(['Visit15', 'Visit17'])]
        df_hba1c_ven = pd.merge(df_hba1c_ven, df_randdt, on=['id'])
        df_hba1c_ven['MonthsOffset'] = (df_hba1c_ven['VisitCollected'].str.extract(r'(\d+)').fillna(0).astype(int))
        df_hba1c_ven['VisitDate'] = df_hba1c_ven.apply(lambda row: row['RandDt'] + pd.DateOffset(months=row['MonthsOffset']), axis=1)
        df_hba1c_ven['VisitDate'] = pd.to_datetime(df_hba1c_ven['VisitDate'])
        df_hba1c_ven['hb_a1c_ven'] = (df_hba1c_ven['HbA1cMMol'] + 2.152)/ 10.929
        df_hba1c_ven.sort_values(by=['id','VisitDate'], ascending=True, inplace=True)
        df_hba1c_ven = df_hba1c_ven[['id', 'VisitCollected','VisitDate', 'hb_a1c_ven']]
        df_hba1c_ven.dropna(subset=['VisitDate'], inplace=True)
        df_hba1c_ven.to_csv(f'{csv_folder}/df_hba1c_ven.csv',index=False)
        
        df_cpep = pd.read_csv(f'{csv_folder}/original/CloudLabCPeptide.txt', sep="|")
        df_cpep.rename(columns={'PtID':'id', 'Height':'height', 'Weight':'weight', 'CPeptide0Min': 'cpep_0_min',
                                'CPeptide10Min': 'cpep_10_min', 'CPeptide15Min': 'cpep_15_min', 'CPeptide30Min': 'cpep_30_min',
                                'CPeptide60Min': 'cpep_60_min', 'CPeptide90Min': 'cpep_90_min', 'CPeptide120Min': 'cpep_120_min'}, inplace=True)
        df_cpep.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_cpep.dropna(subset='cpep_10_min', inplace=True)
        df_cpep.loc[df_cpep['VisitCollected'] == 'Baseline', 'VisitCollected'] = '0 Months'
        df_cpep['VisitMonth'] = df_cpep['VisitCollected'].str.extract(r'(\d+)').astype(float)
        df_cpep.sort_values(by=['id','VisitMonth'], ascending=True, inplace=True)
        df_cpep = df_cpep[['id', 'VisitCollected', 'VisitMonth', 'cpep_0_min', 'cpep_10_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min']]
        df_cpep.to_csv(f'{csv_folder}/df_cpep.csv', index=False)

        df_screening = pd.read_csv(f'{csv_folder}/original/CloudMMTTProced.txt', sep="|")
        df_screening.rename(columns={'PtID':'id', 'Height':'height', 'Weight':'weight', 'Visit': 'visit'}, inplace=True)
        df_screening.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_screening.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_screening.dropna(subset=['weight'], inplace=True)
        df_screening.dropna(subset=['height'], inplace=True)
        df_screening = df_screening[['id','visit','height','weight']]
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)
        df_screening = df_screening.groupby("id")[["height", "weight"]].mean().reset_index()

        df_demographic = pd.read_csv(f'{csv_folder}/original/CloudRecruitment.txt', sep="|")
        df_demographic.rename(columns={'PtID':'id', 'Sex':'sex', 'Ethnicity':'ethnicity', 'Race':'race'}, inplace=True)
        df_demographic.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_demographic.sort_values(by=['id'], ascending=True, inplace=True)
        df_demographic = df_demographic[['id', 'sex', 'ethnicity', 'race']]
        df_demographic['sex'] = df_demographic['sex'].map({'M': 0, 'F': 1})
        df_demographic.to_csv(f'{csv_folder}/df_demographic.csv', index=False)

        df_final = df_cgm.copy()
        df_final = self.create_3_month_time_bins(df_final)
        df_final = df_final[['id','timestamp', 'time_bins', 'timestamp_seconds', 'glucose mmol/l']]
        df_final['VisitMonth_final'] = df_final['time_bins'].str.extract(r'(\d+)').astype(float)

        df_final = df_final.merge(df_age, on=['id'], how='inner')
        df_final = df_final.merge(df_screening, on='id', how='inner')
        df_final = df_final.merge(df_demographic, on='id', how='inner')

        common_ids = set(df_hba1c_cap['id']) & set(df_hba1c_ven['id']) & set(df_final['id']) & set(df_hba1c_local['id']) & set(df_cpep['id'])
        df_final = df_final[df_final['id'].isin(common_ids)].sort_values(by=['timestamp']).reset_index(drop=True)
        df_hba1c_cap = df_hba1c_cap[df_hba1c_cap['id'].isin(common_ids)].sort_values(by=['VisitDate']).reset_index(drop=True)
        df_hba1c_ven = df_hba1c_ven[df_hba1c_ven['id'].isin(common_ids)].sort_values(by=['VisitDate']).reset_index(drop=True)
        df_hba1c_local = df_hba1c_local[df_hba1c_local['id'].isin(common_ids)].sort_values(by='HbA1cTestDt').reset_index(drop=True)
        df_cpep = df_cpep[df_cpep['id'].isin(common_ids)].sort_values(by='VisitMonth').reset_index(drop=True)

        df_final = pd.merge_asof(df_final, df_hba1c_cap, by='id', left_on='timestamp', right_on='VisitDate', direction='nearest')
        df_final = pd.merge_asof(df_final, df_hba1c_ven, by='id', left_on='timestamp', right_on='VisitDate', direction='nearest')
        df_final = pd.merge_asof(df_final, df_hba1c_local, by='id', left_on='timestamp', right_on='HbA1cTestDt', direction='nearest')
        df_final = df_final.sort_values(by=['VisitMonth_final']).reset_index(drop=True)
        df_final = pd.merge_asof(df_final, df_cpep, by='id', left_on='VisitMonth_final', right_on='VisitMonth', direction='nearest')
        df_final = df_final.sort_values(by=['id', 'timestamp']).reset_index(drop=True)
        df_final.drop(columns=['VisitCollected_x','VisitDate_x','VisitCollected_y', 'VisitDate_y','HbA1cTestDt', 'VisitMonth_final', 'VisitMonth', 'VisitCollected'], inplace=True)

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
        

        return df_final, pd.DataFrame(df_summary)

    def preprocess_clvr(self, csv_folder):
        """Custom preprocessing for CLVR dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/cgmAnalysis.txt', sep="|")
        df_cgm.rename(columns={'PtID':'id', 'DeviceDtTm':'timestamp', 'glucose': 'glucose mmol/l', 'Visit':'visit'}, inplace=True)

        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%d%b%Y:%H:%M:%S.%f", errors ='coerce')
        df_cgm.dropna(subset=['timestamp'], inplace=True)
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds())
        
        df_cgm.loc[:,'glucose mmol/l'] = df_cgm[['glucose mmol/l']]/18
        
        df_cgm = df_cgm[['id', 'visit', 'timestamp', 'timestamp_seconds', 'glucose mmol/l' ]]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #Extra features
        df_demographic = pd.read_csv(f'{csv_folder}/original/subjectsEnroll.txt', sep="|")
        df_demographic.rename(columns={'PtID':'id', 'Gender':'sex', 'Ethnicity':'ethnicity', 'Race':'race'}, inplace=True)
        df_demographic.drop_duplicates(subset=['id'], inplace=True)
        df_demographic.sort_values(by=['id'], ascending=True, inplace=True)
        df_demographic = df_demographic[['id', 'sex', 'ethnicity', 'race']]
        df_demographic['sex'] = df_demographic['sex'].map({'M': 0, 'F': 1})
        df_demographic.to_csv(f'{csv_folder}/df_demographic.csv', index=False)

        df_screening = pd.read_csv(f'{csv_folder}/original/visits.txt', sep="|")
        df_screening.rename(columns={'PtID':'id', 'heightCm':'height', 'weightKg':'weight', 'Visit': 'visit', 'ageAtVisit': 'age'}, inplace=True)
        df_screening.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_screening.dropna(subset=['weight'], inplace=True)
        df_screening.dropna(subset=['height'], inplace=True)
        df_screening = df_screening[['id','visit','height','weight', 'age']]
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)

        df_mmtt = pd.read_csv(f'{csv_folder}/original/mmttResults.txt', sep="|")
        df_mmtt.rename(columns={'PtID':'id', 'Visit': 'visit'}, inplace=True)
        df_mmtt = df_mmtt.pivot(index=['id', 'visit', 'CollectionDt'], columns='ResultName', values='Value').reset_index()
        df_mmtt.rename(columns={'C-PEP-0': 'cpep_0_min', 'C-PEP-15': 'cpep_15_min', 'C-PEP-30': 'cpep_30_min',
                                'C-PEP-60': 'cpep_60_min', 'C-PEP-90': 'cpep_90_min', 'C-PEP-120': 'cpep_120_min'}, inplace=True)
        df_mmtt['CollectionDt'] = pd.to_datetime(df_mmtt['CollectionDt'], format = "%d%b%Y")
        df_mmtt.drop(columns=['GLU-0','GLU-120','GLU-15','GLU-30','GLU-60','GLU-90'], inplace=True, axis=1)
        df_mmtt.to_csv(f'{csv_folder}/df_mmtt.csv', index=False)

        df_final = df_cgm.copy()
        df_final = self.create_3_month_time_bins(df_final)
        df_final = df_final[['id','timestamp', 'time_bins', 'visit', 'timestamp_seconds', 'glucose mmol/l']]

        df_final = df_final.merge(df_screening, on=['id', 'visit'], how='inner')
        df_final = df_final.merge(df_demographic, on=['id'], how='inner')

        common_ids = set(df_mmtt['id']) & set(df_final['id'])
        df_final = df_final[df_final['id'].isin(common_ids)].sort_values(by=['timestamp']).reset_index(drop=True)
        df_mmtt = df_mmtt[df_final['id'].isin(common_ids)].sort_values(by=['CollectionDt']).reset_index(drop=True)
        df_final = pd.merge_asof(df_final, df_mmtt, by='id', left_on='timestamp', right_on='CollectionDt', direction='nearest')

        df_final['hb_a1c_local'] = 0

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

        return df_final, pd.DataFrame(df_summary)

    def preprocess_defend(self, csv_folder):
        """Custom preprocessing for DEFEND-2 dataset."""
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/defend_2.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        df_cgm = df_cgm[['id','timestamp','timestamp_seconds','glucose mmol/l','time_bins']]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final[['sex', 'age', 'weight', 'height', 'cpep_0_min', 'cpep_10_min', 'cpep_15_min',
                  'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'hb_a1c_local']] = 0
        df_final = self.create_3_month_time_bins(df_final)

        '''Data info'''
        num_people = df_final['id'].nunique()

        num_male = 'N/A'
        num_female = 'N/A'

        min_age = 'N/A'
        max_age = 'N/A'
        num_less_than_18 = 'N/A'
        num_18_or_older = 'N/A'

        c_peptide_interval = 'N/A'
        beta2_interval = {'N/A'}
        cgm_interval = {'N/A'}

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
    
    def preprocess_diagnode(self, csv_folder):
        """Custom preprocessing for DIAGNODE dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/CGM_RANDOMISED_preprocessed.csv')

        df_cgm[['id','visit','device_id']] = df_cgm['ID_VISIT_DEVICEID'].str.split('_',expand=True)
        df_cgm = df_cgm[['id','visit','TIMESTAMP','GLUCOSE']]
        df_cgm.rename(columns={'TIMESTAMP': 'timestamp', 'GLUCOSE': 'glucose mmol/l'}, inplace=True)

        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format = "%Y-%m-%d %H:%M:%S", errors ='coerce')
        df_cgm.dropna(subset=['timestamp'], inplace=True)
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds())

        df_cgm['time_diff'] = df_cgm.groupby('id')['timestamp_seconds'].diff()
        # df_cons_intervals = df_cgm[(df_cgm['time_diff'] < 930) & (df_cgm['time_diff']>870)]
        # const_ratio = df_cons_intervals.shape[0]/df_cgm.shape[0]
        # print(const_ratio)

        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm.to_csv('./data/diagnode/df_cgm.csv', index=False)

        df_cgm.drop(['time_diff'], axis=1, inplace=True)

        #Extra feature
        df_aljc = pd.read_csv(f'{csv_folder}/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')

        df_screening = df_aljc[['id','visit','treatment','sex','weight','height','age']]
        df_screening.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_screening.dropna(subset=['weight'], inplace=True)
        df_screening.dropna(subset=['height'], inplace=True)
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)
        
        df_hba_a1c = df_aljc[['id', 'visit', 'hb_a1c']]
        df_hba_a1c.rename(columns={'hb_a1c' : 'hb_a1c_local'}, inplace=True)
        df_hba_a1c.dropna(subset=['hb_a1c_local'], inplace=True)
        df_hba_a1c.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_hba_a1c.to_csv(f'{csv_folder}/df_hba_a1c.csv', index=False)

        df_cpep = df_aljc[['id', 'visit', 'cpep0', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
        df_cpep.rename(columns={'cpep0': 'cpep_0_min', 'cpep30': 'cpep_30_min',
                                'cpep60': 'cpep_60_min', 'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min'}, inplace=True)
        df_cpep.dropna(subset=['cpep_0_min'], inplace=True)
        df_cpep.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_cpep.to_csv(f'{csv_folder}/df_cpep.csv', index=False)

        #Merged dataframe
        df_final = df_cgm.copy()
        df_final = self.create_3_month_time_bins(df_final)
        df_final = df_final.merge(df_screening, on=['id','visit'], how='inner')
        df_final = df_final.merge(df_hba_a1c, on=['id','visit'], how='inner')
        df_final = df_final.merge(df_cpep, on=['id','visit'], how='inner')

        df_final = df_final[['id','timestamp','timestamp_seconds', 'time_bins', 'glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'cpep_0_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'hb_a1c_local']]
        df_final['sex'] = df_final['sex'].map({'Male': 0, 'Female': 1})

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
        
        return df_final, pd.DataFrame(df_summary)
    
    def preprocess_gskalb(self, csv_folder):
        """Custom preprocessing for GSKALB dataset."""
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/gskalb.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        df_cgm = df_cgm[['id','timestamp','timestamp_seconds','glucose mmol/l','time_bins']]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final[['sex', 'age', 'weight', 'height', 'cpep_0_min', 'cpep_10_min', 'cpep_15_min',
                  'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'hb_a1c_local']] = 0
        df_final = self.create_3_month_time_bins(df_final)

        '''Data info'''
        num_people = df_final['id'].nunique()

        num_male = 'N/A'
        num_female = 'N/A'

        min_age = 'N/A'
        max_age = 'N/A'
        num_less_than_18 = 'N/A'
        num_18_or_older = 'N/A'

        c_peptide_interval = 'N/A'
        beta2_interval = {'N/A'}
        cgm_interval = {'N/A'}

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

    def preprocess_itx(self, csv_folder):
        """Custom preprocessing for Islet Transplant dataset"""
        df_cgm = itx_dataloader.get_cgm_data()
        df_final = itx_dataloader.get_final_data()
        df_final = self.create_3_month_time_bins(df_final)
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

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

    def preprocess_jaeb_healthy(self, csv_folder):
        """Custom preprocessing for JAEB healthy dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/NonDiabDeviceCGM.csv')
        df_cgm.rename(columns={'PtID': 'id'}, inplace=True)
        df_cgm = df_cgm[df_cgm['RecordType'] == 'CGM']

        df_cgm['glucose mmol/l'] = df_cgm['Value'] / 18 #Converting mg/dl to mmol/l
        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm.drop(columns=('Value'), axis=1)

        #Cleaning timestamp
        df_cgm.loc[:, 'timestamp_seconds'] = pd.to_timedelta(df_cgm['DeviceTm']).dt.total_seconds() + df_cgm['DeviceDtDaysFromEnroll'] * 86400
        df_cgm['time_diff'] = df_cgm.groupby('id')['timestamp_seconds'].diff()

        initial_date = pd.to_datetime("2024-01-01")
        df_cgm['timestamp'] = initial_date + pd.to_timedelta(df_cgm['DeviceDtDaysFromEnroll'], unit='D')
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'].astype(str) + ' ' + df_cgm['DeviceTm'])

        # df_cons_intervals = df_cgm[(df_cgm['time_diff'] < 930) & (df_cgm['time_diff']>870)]
        # const_ratio = df_cons_intervals.shape[0]/df_cgm.shape[0]
        # print(const_ratio)

        df_cgm.drop(['RecordType', 'time_diff', 'DeviceTm', 'DeviceDtDaysFromEnroll', 'Value'], axis=1, inplace=True)
        df_cgm.to_csv('./data/jaeb_healthy/df_cgm.csv', index=False)

        #Adding Age
        df_age = pd.read_csv(f'{csv_folder}/original/NonDiabPtRoster.csv')
        df_age.rename(columns={'PtID': 'id'}, inplace=True)        
        df_age.drop_duplicates(['RecID', 'id'], inplace=True)
        df_age = df_age[['id','AgeAsOfEnrollDt']]
        df_age.to_csv('./data/jaeb_healthy/df_age.csv', index=False)

        #Extra features
        df_screening = pd.read_csv(f'{csv_folder}/original/NonDiabScreening.csv')
        df_screening.rename(columns={'PtID': 'id'}, inplace=True)
        df_screening.drop_duplicates(['RecID', 'id'], inplace=True)

        df_screening = df_screening[['id','Weight', 'Height', 'HbA1c', 'Gender', 'Race']]
        df_screening['Gender'] = df_screening['Gender'].map({'M' : 0, 'F': 1})
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)

        df_final = df_cgm.copy()
        df_final = self.create_3_month_time_bins(df_final)
        df_final = df_final.merge(df_age, on='id', how='inner')
        df_final = df_final.merge(df_screening, on='id', how='inner')
        df_final.rename(columns={'DeviceTm':'timestamp', 'Value':'glucose mmol/l', 'AgeAsOfEnrollDt':'age',
                                'Weight':'weight', 'Height':'height', 'HbA1c':'hb_a1c_local', 'Gender': 'sex'}, inplace=True)
        df_final = df_final[['id','timestamp','timestamp_seconds', 'time_bins', 'glucose mmol/l', 'scaled_glucose',
                             'age', 'weight', 'height', 'sex', 'hb_a1c_local']]
        df_final[['cpep_0_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']] = 0

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

    def preprocess_jaeb_t1d(self, csv_folder):
        '''Custom preprocessing for JABE Type 1 Diabetes Dataset'''
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/2025-03-13_T1D_JAEB_C-path_CGM_clean.csv')
        df_cgm.rename(columns={'PtID':'id','DeviceTm': 'timestamp', 'Value' : 'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final[['sex', 'age', 'weight', 'height', 'cpep_0_min', 'cpep_10_min', 'cpep_15_min',
                  'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'hb_a1c_local']] = 0
        df_final = self.create_3_month_time_bins(df_final)

        '''Data info'''
        num_people = df_final['id'].nunique()

        num_male = 'N/A'
        num_female = 'N/A'

        min_age = 'N/A'
        max_age = 'N/A'
        num_less_than_18 = 'N/A'
        num_18_or_older = 'N/A'

        c_peptide_interval = 'N/A'
        beta2_interval = {'N/A'}
        cgm_interval = {'N/A'}

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

    def create_3_month_time_bins(self, df):
        df_time_bin = df.copy()
        df_grouped_time_bin = df_time_bin.groupby('id')

        all_patients = []

        for pt_id, pt_df in df_grouped_time_bin:
            pt_df['time_bins'] = 'N/A'
            pt_df['timestamp'] = pd.to_datetime(pt_df['timestamp'])
            start = pt_df['timestamp'].min()
            max_time = pt_df['timestamp'].max()
            months = 0

            while start < max_time:
                if months == 0:
                    end = start + pd.DateOffset(months=2, weeks=3)
                else:
                    end = start + pd.DateOffset(months=3)

                mask = (pt_df['timestamp'] >= start) & (pt_df['timestamp'] < end)
                pt_df.loc[mask, 'time_bins'] = f'Months {months}'
                start = end
                months+=3
        
            all_patients.append(pt_df)

        return pd.concat(all_patients, ignore_index=True)
    
    def hypoglycemia_rates(self, df):
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
    
    def hypoglycemia_rates_std_cv(self, df_analysis):
        '''Returns standard deviation and coefficient variation of hypoglycemia rates'''
        hypo_level_1_mean = df_analysis['hypoglycemia_level_1'].mean()
        hypo_level_1_std = df_analysis['hypoglycemia_level_1'].std()
        hypo_level_1_cv = hypo_level_1_std/hypo_level_1_mean * 100

        hypo_level_2_mean = df_analysis['hypoglycemia_level_2'].mean()
        hypo_level_2_std = df_analysis['hypoglycemia_level_2'].std()
        hypo_level_2_cv = hypo_level_2_std/hypo_level_2_mean * 100

        cl_sig_hypo_mean = df_analysis['cl_sig_hypo'].mean()
        cl_sig_hypo_std = df_analysis['cl_sig_hypo'].std()
        cl_sig_hypo_cv = cl_sig_hypo_std/cl_sig_hypo_mean * 100

        return {
            'hypo_level_1_mean' : hypo_level_1_mean,
            'hypo_level_1_std' : hypo_level_1_std,
            'hypo_level_1_cv' : hypo_level_1_cv,
            'hypo_level_2_mean' : hypo_level_2_mean,
            'hypo_level_2_std' : hypo_level_2_std,
            'hypo_level_2_cv' : hypo_level_2_cv,
            'cl_sg_hypo_mean' : cl_sig_hypo_mean,
            'cl_sig_hypo_std' : cl_sig_hypo_std,
            'cl_sig_hypo_cv' : cl_sig_hypo_cv
        }
    
    def get_dataset(self, name):
        '''Retrieves a dataset by name.'''
        return self.datasets.get(name, None)
    
    def get_dataset_summary(self, name):
        return self.datasets_summary.get(name, None)
    
    def get_cgm_core_endpoints_per_group(self, df_group, **kwargs):
        '''
            Returns core CGM endpoints such as:
            Time in range
            Time below range <3.9 mmol/l, including readings of <3.0 mmol/l
            Time below range <3.0 mmol/l
            Time above range >10.0 mmol/l, including readings of >13.9 mmol/l
            Time above range >13.9 mmol/l
            Coefficient of variation
            Standard deviation of mean glucose
            Mean sensor glucose
        '''
        df_group = df_group.copy()
        df_group['timestamp'] = pd.to_datetime(df_group['timestamp'])

        df_group['time_diff'] = df_group['timestamp'].diff().dt.total_seconds() / 3600
        median_interval = df_group['time_diff'].median()
        df_group['time_diff'] = df_group['time_diff'].fillna(median_interval)

        total_time = df_group['time_diff'].sum()
        mean_glucose = df_group['glucose mmol/l'].mean()
        std_glucose = df_group['glucose mmol/l'].std()

        if total_time == 0 or mean_glucose == 0:
            return {}

        cv = std_glucose / mean_glucose * 100

        in_range = (df_group['glucose mmol/l'] >= 3.9) & (df_group['glucose mmol/l'] <= 10.0)
        below_3_9 = df_group['glucose mmol/l'] < 3.9
        below_3_0 = df_group['glucose mmol/l'] < 3.0
        above_10 = df_group['glucose mmol/l'] > 10.0
        above_13_9 = df_group['glucose mmol/l'] > 13.9

        tir = df_group.loc[in_range, 'time_diff'].sum()
        tbr_1 = df_group.loc[below_3_9, 'time_diff'].sum()
        tbr_2 = df_group.loc[below_3_0, 'time_diff'].sum()
        tar_1 = df_group.loc[above_10, 'time_diff'].sum()
        tar_2 = df_group.loc[above_13_9, 'time_diff'].sum()

        max_wear = (df_group['timestamp'].iloc[-1] - df_group['timestamp'].iloc[0]).total_seconds() / 3600
        wear_pct = (total_time / max_wear) * 100 if max_wear > 0 else 0

        return {
            'percent_wear_time': wear_pct,
            'percent_in_range': (tir / total_time) * 100,
            'percent_tbr_level_1': (tbr_1 / total_time) * 100,
            'percent_tbr_level_2': (tbr_2 / total_time) * 100,
            'percent_tar_level_1': (tar_1 / total_time) * 100,
            'percent_tar_level_2': (tar_2 / total_time) * 100,
            'mean_glucose': mean_glucose,
            'std_glucose': std_glucose,
            'cv_percent': cv
        }
    
    def get_cgm_core_endpoints_general(self, df: pd.DataFrame, **kwargs):
        df_cgm = df.copy()
        df_cgm = df_cgm[['id', 'timestamp', 'glucose mmol/l']]
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])

        core_endpoints = []
        for pt_id, pt_df in df_cgm.groupby(['id']):
            core_endpoint = self.get_cgm_core_endpoints_per_group(pt_df)
            if core_endpoint:  # ensure it's not empty
                core_endpoint['id'] = pt_id[0]
                core_endpoints.append(core_endpoint)

        return pd.DataFrame(core_endpoints)

    def get_cgm_avg_core_endpoints(self, df_results):

        # #TODO: fix for DataFrame
        # '''
        # Returns average core endpoints of the dataframe.
        # '''

        # avg_results = {
        #         'avg_percent_wear_time' : 0,
        #         'avg_time_in_range_hours' : 0,
        #         'avg_perctent_in_range' : 0,
        #         'avg_time_below_range_level_1' : 0,
        #         'avg_time_below_range_level_2' : 0,
        #         'avg_time_above_range_level_1' : 0,
        #         'avg_time_above_range_level_2' : 0,
        #         'avg_mean_glucose' : 0,
        #         'avg_std_glucose' : 0,
        #         'avg_cv_percent' : 0
        #     }

        # for ptid, result in results.items():
        #     for core_endpoint_name, core_endpoint_value in result.items():
        #         avg_core_endpoint_name = 'avg_' + core_endpoint_name
        #         avg_results[avg_core_endpoint_name] += core_endpoint_value

        # for avg_core_endpoint_name in avg_results.keys():
        #     num_patients = len(results.keys())
        #     avg_results[avg_core_endpoint_name] = avg_results[avg_core_endpoint_name]/num_patients

        # return avg_results
        pass

    def list_datasets(self):
        """Lists all loaded datasets."""
        return list(self.datasets.keys())
    
    def length_data(self, df):

        df = df.copy()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        median_time_interval = (df['timestamp'].diff().median()).seconds / 60
        
        if median_time_interval < 6:
            df = df.iloc[::3].reset_index(drop=True)

        total_distance_i = 0
        total_distance_t = 0
        total_time_i = 0
        total_time_t = 0
        valid_diffs = 0

        for i in range(1,df.shape[0]):
            current_time_interval = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 60

            if current_time_interval > 16:
                total_distance_t += total_distance_i
                total_time_t += total_time_i

                total_distance_i = 0
                total_time_i = 0

            else:  
                current_glucose_diff = abs(df.iloc[i]['glucose mmol/l'] - df.iloc[i-1]['glucose mmol/l'])
                current_glucose_diff *= 18 #Converting mmol/l to mg/dl
                current_time_diff = current_time_interval/60 #Converting minutes to hours

                total_distance_i += np.sqrt(current_time_diff**2 + current_glucose_diff**2)
                total_time_i += current_time_diff 
                valid_diffs += 1

        total_distance_t += total_distance_i
        total_time_t += total_time_i

        return total_time_t, total_distance_t, total_distance_t/max(total_time_t,1)

    def plot_hypothesis_test_general(self, df_healthy, df_t1d, feature_name, test_results, path):
        '''Plots distributions of the given feature for healthy vs a T1D group'''
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

    def plot_hypothesis_test_time_bins(self, df, feature, p_kruskal, rho_spearman, p_spearman, path):
        full_time_bin_order = [
            'Months 0', 'Months 3', 'Months 6', 'Months 9',
            'Months 12', 'Months 15', 'Months 18', 'Months 21', 'Healthy'
        ]
        df = df[df['time_bins'] != 'N/A']
        time_bin_order = [b for b in full_time_bin_order if b in df['time_bins'].unique()]
        missing_bins = [b for b in full_time_bin_order if b not in df['time_bins'].unique()]
        # if missing_bins:
        #     print(f"⚠️ Skipping bins not present in data: {missing_bins}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='time_bins', y=feature, data=df, order=time_bin_order, showfliers=True)
        sns.stripplot(x='time_bins', y=feature, data=df, order=time_bin_order, color='black', alpha=0.4, jitter=True)
        plt.title(f"{feature}\nKruskal p={p_kruskal:.4f}, Spearman ρ={rho_spearman:.2f} (p={p_spearman:.4f})", fontsize=13)
        plt.ylabel(feature)
        plt.xlabel('Time Bin')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
    def plot_analyzed_features_general(self, feature_name):
        '''Plots distribution of given feature for all groups together'''
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
        plt.savefig(f'./data/feature_analysis/general/{feature_name}_analysis.png')
        plt.close()
    
    """Feature analysis methods"""

    def print_summaries(self):
        '''Prints out summaries of available datasets'''
        for df_name, df_summary in self.datasets_summary.items():
            print(f'\n{df_name} Summary:')
            for summary_feature in df_summary.columns:
                print(f'\t{summary_feature}: {df_summary[summary_feature].values[0]}')

    def statistical_analysis(self, csv_folder, df):
        df = df.copy()
        grouped_patients = df.groupby('id')
    
        patients_analysis = {}
        patients_analysis['id'] = []
        patients_analysis['glucose_avg'] = []
        patients_analysis['total_length'] = []
        patients_analysis['normalized_length'] = []
        patients_analysis['hypoglycemia_level_1'] = []
        patients_analysis['hypoglycemia_level_2'] = []
        patients_analysis['cl_sig_hypo'] = []
        patients_analysis['prc_hypoglycemia_level_1'] = []
        patients_analysis['prc_hypoglycemia_level_2'] = []
        patients_analysis['prc_cl_sig_hypo'] = []

        for patient_id, patient_df in grouped_patients:
            num_cgm_data = patient_df.shape[0]
            total_time, length, norm_length = self.length_data(patient_df)
            level_1, level_2, cl_sig_events = self.hypoglycemia_rates(patient_df)
            glucose_avg = patient_df['glucose mmol/l'].mean()

            patients_analysis['id'].append(patient_id)
            patients_analysis['glucose_avg'].append(glucose_avg)
            patients_analysis['total_length'].append(length)
            patients_analysis['normalized_length'].append(norm_length)
            patients_analysis['hypoglycemia_level_1'].append(level_1)
            patients_analysis['hypoglycemia_level_2'].append(level_2)
            patients_analysis['cl_sig_hypo'].append(cl_sig_events)
            patients_analysis['prc_hypoglycemia_level_1'].append(level_1/num_cgm_data * 100) 
            patients_analysis['prc_hypoglycemia_level_2'].append(level_2/num_cgm_data * 100)
            patients_analysis['prc_cl_sig_hypo'].append(cl_sig_events/num_cgm_data * 100)

        df_patients_analysis = pd.DataFrame(patients_analysis)
        df_patients_analysis = df_patients_analysis.loc[:, ~df_patients_analysis.columns.duplicated()]
        df_patients_analysis.to_csv(f'{csv_folder}/df_analysis.csv', index=False)

        return df_patients_analysis
    
    def kruskal_across_time_bins(self, df, feature):
        groups = [group[feature].dropna().values for name, group in df.groupby('time_bins')]
        stat, p_value = kruskal(*groups)
        return stat, p_value
    
    def spearman_trend(self, df, feature, bin_order):
        df = df.copy()
        df['time_index'] = df['time_bins'].map(bin_order)
        df = df[['time_index', feature]].dropna()
        rho, p_value = spearmanr(df['time_index'], df[feature])
        return rho, p_value
    
    def pairwise_bin_tests(self, df, feature, alpha=0.05):
        results = []
        bins = df['time_bins'].dropna().unique()
        
        for b1, b2 in combinations(bins, 2):
            v1 = pd.to_numeric(df[df['time_bins'] == b1][feature], errors='coerce').dropna()
            v2 = pd.to_numeric(df[df['time_bins'] == b2][feature], errors='coerce').dropna()

            if len(v1) < 2 or len(v2) < 2:
                continue  # Skip if not enough data to test

            stat, p = mannwhitneyu(v1, v2, alternative='two-sided')
            results.append({
                'bin_1': b1,
                'bin_2': b2,
                'p_value': p,
                'significant': p < alpha
            })
        return pd.DataFrame(results)
    
    def hypothesis_test(self, df_healthy, df_t1d, feature_name, alpha = 0.05):
        '''This is used to find if there is significant differnece between healthy and t1d features'''

        values_healthy = df_healthy[feature_name].dropna()
        values_t1d = df_t1d[feature_name].dropna()

        #Checking for normality of the data
        stat_healthy, p_healthy = shapiro(values_healthy)
        stat_t1d, p_t1d = shapiro(values_t1d)

        normal_healthy = p_healthy > alpha
        normal_t1d = p_t1d > alpha

        if normal_healthy and normal_t1d:
            # Using independent t-test if data isnormally distributed
            stat, p_value = ttest_ind(values_healthy, values_t1d, equal_var=False)
            test_used = "T-test (Welch's correction)"
        
        else:
            # Using Mann-Whitney U test if data is not normally distributed
            stat, p_value = mannwhitneyu(values_healthy, values_t1d, alternative='two-sided')
            test_used = "Mann-Whitney U test"

        is_significant = p_value < alpha

        return {
            "feature": feature_name,
            "test_used": test_used,
            "statistic": stat,
            "p_value": p_value,
            "significant": is_significant,
            "normality_healthy": normal_healthy,
            "normality_t1d": normal_t1d
        }
    
    def hypothesis_test_LoL(self):
        '''Hypothesis test of Length of Line'''
        analyzed_datasets_t1d_dict = self.analyzed_datasets.copy()
        analyzed_datasets_t1d_dict.pop('jaeb_healthy')
        df_analyzed_healthy = self.analyzed_datasets['jaeb_healthy']

        for df_name, df_analyzed_t1d in analyzed_datasets_t1d_dict.items():
            hypo_test_results = self.hypothesis_test(df_analyzed_healthy, df_analyzed_t1d, feature_name='normalized_length')
            self.plot_hypothesis_test_general(df_analyzed_healthy, df_analyzed_t1d, 'normalized_length', hypo_test_results, f'./data/feature_analysis/general/LoL/jh_vs_{df_name}.png')   

    def hypothesis_test_CGM_core_endpoints_general(self):
        '''Hypothesis test of CGM Core Endpoints in total time'''
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy']
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)

        for df_name, df_t1d in dict_df_t1d.items():
            t1d_core_endpoints = self.get_cgm_core_endpoints_general(df_t1d)
            for core_endpoints_feature in jaeb_healthy_core_endpoints.columns:
                hypo_test_results = self.hypothesis_test(jaeb_healthy_core_endpoints, t1d_core_endpoints, core_endpoints_feature)
                self.plot_hypothesis_test_general(jaeb_healthy_core_endpoints, t1d_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/general/{core_endpoints_feature}/jh_vs_{df_name}.png')

    def hypothesis_test_CGM_core_endpoints_time_bins(self):
        dict_df_t1d = self.datasets.copy()
        dict_df_t1d.pop('jaeb_healthy')
        df_jaeb_healthy = self.datasets['jaeb_healthy'].copy()
        jaeb_healthy_core_endpoints = self.get_cgm_core_endpoints_general(df_jaeb_healthy)
        jaeb_healthy_core_endpoints['time_bins'] = 'Healthy'
        jaeb_healthy_core_endpoints['id'] = 'Healthy_' + jaeb_healthy_core_endpoints['id'].astype('str')
        jaeb_healthy_core_endpoints.to_csv('./healthy.csv', index=False)

        bin_order = {
            'Months 0': 0,
            'Months 3': 1,
            'Months 6': 2,
            'Months 9': 3,
            'Months 12': 4,
            'Months 15': 5,
            'Months 18': 6,
            'Months 21': 7,
}


        for df_name, df_t1d in dict_df_t1d.items():
            df_t1d = df_t1d[~(df_t1d['time_bins'] == 'N/A')]
            df_time_bins = df_t1d.groupby(['time_bins']).apply(self.get_cgm_core_endpoints_general, include_groups = True).reset_index()
            df_time_bins['months'] = df_time_bins['time_bins'].str.extract(r'(\d+)').fillna(0).astype(int)
            df_time_bins.sort_values(by=['months'], inplace=True)
            df_time_bins.to_csv(f'./data/core_endpoints/{df_name}.csv', index=False)
            print(f'\tResults of {df_name}:')
            for core_endpoints_feature in jaeb_healthy_core_endpoints.columns.drop(['id','time_bins']):
                df_time_bins = df_time_bins.groupby('time_bins').filter(lambda g: len(g) >= 3)
                stat, p_kruskal = self.kruskal_across_time_bins(df_time_bins, core_endpoints_feature)
                rho, p_spearman = self.spearman_trend(df_time_bins, core_endpoints_feature, bin_order)
                pairwise_results = self.pairwise_bin_tests(df_time_bins, core_endpoints_feature)
                df_time_bins = pd.concat([df_time_bins, jaeb_healthy_core_endpoints], ignore_index=True)
                self.plot_hypothesis_test_time_bins(df_time_bins, core_endpoints_feature,p_kruskal,rho, p_spearman, path=f'./data/feature_analysis/time_bins/{core_endpoints_feature}/jh_vs_{df_name}.png')
                print(f'Feature: {core_endpoints_feature}')
                print(f"Kruskal-Wallis p = {p_kruskal:.10f}")
                print(f"Spearman ρ = {rho:.2f}, p = {p_spearman:.10f}\n")
if __name__ == '__main__':
    data_analysis = DataAnalysis()

    df_cloud = data_analysis.get_dataset('cloud')
    df_clvr = data_analysis.get_dataset('clvr')
    df_defend = data_analysis.get_dataset('defend')
    df_diagnode = data_analysis.get_dataset('diagnode')
    df_gskalb = data_analysis.get_dataset('gskalb')
    df_itx = data_analysis.get_dataset('itx')
    df_jaeb_healthy = data_analysis.get_dataset('jaeb_healthy')
    df_jaeb_t1d = data_analysis.get_dataset('jaeb_t1d')

    '''Printing dataset summaries'''
    data_analysis.print_summaries()

    '''Hypoglycemia rates statistics '''
    # # hypo_rate_statistics = data_analysis.hypoglycemia_rates_std_cv(df_diagnode)
    # # print(hypo_rate_statistics)

    '''Hypothesis test: Length of Line'''
    # data_analysis.hypothesis_test_LoL()

    '''Hypothesis test: CGM core endpoints GENERAL'''
    # data_analysis.hypothesis_test_CGM_core_endpoints_general()

    '''Hypothesis test: CGM core endpoints TIME BINS'''
    # data_analysis.hypothesis_test_CGM_core_endpoints_time_bins()

    '''Boxplot hypoglycemia rates'''
    # data_analysis.plot_analyzed_features_general('prc_hypoglycemia_level_1')
    # data_analysis.plot_analyzed_features_general('prc_hypoglycemia_level_2')
    # data_analysis.plot_analyzed_features_general('prc_cl_sig_hypo')

    # data_analysis.get_cgm_core_endpoints_general(df_jaeb_healthy).reset_index().to_csv('./healthy.csv', index=False)
