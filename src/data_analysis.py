import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import seaborn as sns
import random
import warnings
import matplotlib.pyplot as plt
import importlib
import os
from itx_package import ITXData

warnings.simplefilter(action='ignore', category=FutureWarning)
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
        self.load_data()

    def load_data(self): 
        """Loads, cleans, and normalizes all datasets from CSV files."""

        print('\n\tLoading datasets...')
        for dataset_name, config in DATASET_CONFIG.items():
            csv_folder = config["csv_folder"]
            if not os.path.exists(csv_folder):
                print(f"Warning: {csv_folder} not found. Skipping {dataset_name}.")
                continue

            print(f"Loading {dataset_name} from {csv_folder}...")
            df, df_summary = self._preprocess_data(csv_folder, config["preprocess"])
            self.datasets[dataset_name] = df
            self.datasets_summary[dataset_name] = df_summary

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
        df_age = df_age[['id', 'age']]
        df_age.to_csv(f'{csv_folder}/df_age.csv', index=False)

        df_hba1c = pd.read_csv(f'{csv_folder}/original/CloudDiabLocalHbA1c.txt', sep="|")
        df_hba1c.rename(columns={'PtID':'id', 'Visit':'visit'}, inplace=True)
        df_hba1c['HbA1cTestDt'] = pd.to_datetime(df_hba1c['HbA1cTestDt'])
        df_hba1c.sort_values(by=['id','HbA1cTestDt'], ascending=True, inplace=True)
        df_hba1c.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_hba1c.dropna(subset=['HbA1cTestRes'], inplace=True)
        df_hba1c = df_hba1c[['id','visit','HbA1cTestDt', 'HbA1cTestRes']]
        df_hba1c.to_csv(f'{csv_folder}/df_hba1c.csv',index=False)

        df_cpep = pd.read_csv(f'{csv_folder}/original/CloudLabCPeptide.txt', sep="|")
        df_cpep.rename(columns={'PtID':'id', 'Height':'height', 'Weight':'weight'}, inplace=True)
        df_cpep.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_cpep.sort_values(by=['id'], ascending=True, inplace=True)
        df_cpep.to_csv(f'{csv_folder}/df_cpep.csv', index=False)

        df_screening = pd.read_csv(f'{csv_folder}/original/CloudMMTTProced.txt', sep="|")
        df_screening.rename(columns={'PtID':'id', 'Height':'height', 'Weight':'weight', 'Visit': 'visit'}, inplace=True)
        df_screening.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_screening.sort_values(by=['id','visit'], ascending=True, inplace=True)
        df_screening.dropna(subset=['weight'], inplace=True)
        df_screening.dropna(subset=['height'], inplace=True)
        df_screening = df_screening[['id','visit','height','weight']]
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)
        df_avg = df_screening.groupby("id")[["height", "weight"]].mean().reset_index()

        df_demographic = pd.read_csv(f'{csv_folder}/original/CloudRecruitment.txt', sep="|")
        df_demographic.rename(columns={'PtID':'id', 'Sex':'sex', 'Ethnicity':'ethnicity', 'Race':'race'}, inplace=True)
        df_demographic.drop_duplicates(subset=['RecID','id'], inplace=True)
        df_demographic.sort_values(by=['id'], ascending=True, inplace=True)
        df_demographic = df_demographic[['id', 'sex', 'ethnicity', 'race']]
        df_demographic['sex'] = df_demographic['sex'].map({'M': 0, 'F': 1})
        df_demographic.to_csv(f'{csv_folder}/df_demographic.csv', index=False)

        df_final = pd.merge(df_cgm, df_age, on=['id'], how='inner')
        df_final = df_final.merge(df_avg, on='id', how='inner')
        df_final = df_final.merge(df_demographic, on='id', how='inner')

        df_final['hb_a1c'] = 0
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}
        

        return df_final, df_summary

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

        df_final = pd.merge(df_cgm, df_screening, on=['id', 'visit'], how='inner')
        df_final = pd.merge(df_final, df_demographic, on=['id'], how='inner')
        df_final['hb_a1c'] = 0

        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}

        return df_final, df_summary

    def preprocess_defend(self, csv_folder):
        """Custom preprocessing for DEFEND-2 dataset."""
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/defend_2.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        df_cgm = df_cgm[['id','timestamp','timestamp_seconds','glucose mmol/l','time_bins']]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final['sex'] = 0
        df_final['age'] = 0
        df_final['weight'] = 0
        df_final['height'] = 0
        df_final['hb_a1c'] = 0
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}
        
        return df_final, df_summary
    
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
        df_aljc = df_aljc[['id','visit','treatment','sex','weight','height','age','hb_a1c',
                       'c_peptide_fasting','glucose_fasting']]

        #Merged dataframe
        df_final = pd.merge(df_cgm, df_aljc, on=['id','visit'], how='inner')
        df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
        df_final['sex'] = df_final['sex'].map({'Male': 0, 'Female': 1})
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)


        '''Data info:'''
        no_dup_df = df_aljc.drop_duplicates(subset='id')

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}
        
        return df_final, df_summary
    
    def preprocess_gskalb(self, csv_folder):
        """Custom preprocessing for GSKALB dataset."""
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/gskalb.csv')
        df_cgm.rename(columns={'usubjid':'id', 'sensorglucose':'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format="%Y-%m-%dT%H:%M:%SZ")
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        df_cgm = df_cgm[['id','timestamp','timestamp_seconds','glucose mmol/l','time_bins']]
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)

        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final['sex'] = 0
        df_final['age'] = 0
        df_final['weight'] = 0
        df_final['height'] = 0
        df_final['hb_a1c'] = 0
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}
        
        return df_final, df_summary

    def preprocess_itx(self, csv_folder):
        """Custom preprocessing for Islet Transplant dataset"""
        df_cgm = itx_dataloader.get_cgm_data()
        df_final = itx_dataloader.get_final_data()
        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)
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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}

        return df_final, df_summary

    def preprocess_jaeb_healthy(self, csv_folder):
        """Custom preprocessing for JAEB healthy dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/NonDiabDeviceCGM.csv')
        df_cgm = df_cgm[df_cgm['RecordType'] == 'CGM']

        df_cgm['glucose mmol/l'] = df_cgm['Value'] / 18 #Converting mg/dl to mmol/l
        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm.drop(columns=('Value'), axis=1)

        #Cleaning timestamp
        df_cgm.loc[:, 'timestamp_seconds'] = pd.to_timedelta(df_cgm['DeviceTm']).dt.total_seconds() + df_cgm['DeviceDtDaysFromEnroll'] * 86400
        df_cgm['time_diff'] = df_cgm.groupby('PtID')['timestamp_seconds'].diff()

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
        df_age.drop_duplicates(['RecID', 'PtID'], inplace=True)
        df_age = df_age[['PtID','AgeAsOfEnrollDt']]
        df_age.to_csv('./data/jaeb_healthy/df_age.csv', index=False)

        #Extra features
        df_screening = pd.read_csv(f'{csv_folder}/original/NonDiabScreening.csv')
        df_screening.drop_duplicates(['RecID', 'PtID'], inplace=True)

        df_screening = df_screening[['PtID','Weight', 'Height', 'HbA1c', 'Gender', 'Race']]
        df_screening['Gender'] = df_screening['Gender'].map({'M' : 0, 'F': 1})
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)

        df_final = pd.merge(df_cgm, df_age, on='PtID', how='inner')
        df_final = pd.merge(df_final, df_screening, on='PtID', how='inner')

        df_final.rename(columns={'DeviceTm':'timestamp', 'Value':'glucose mmol/l', 'AgeAsOfEnrollDt':'age',
                                'Weight':'weight', 'Height':'height', 'HbA1c':'hb_a1c', 'PtID': 'id', 'Gender': 'sex'}, inplace=True)
        df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
        df_final.to_csv('./data/jaeb_healthy/df_final.csv',index=False)

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
        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}

        return df_final, df_summary

    def preprocess_jaeb_t1d(self, csv_folder):
        '''Custom preprocessing for JABE Type 1 Diabetes Dataset'''
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/2025-03-13_T1D_JAEB_C-path_CGM_clean.csv')
        df_cgm.rename(columns={'PtID':'id','DeviceTm': 'timestamp', 'Value' : 'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        #TODO: Add other features once i get them
        df_final = df_cgm.copy()
        df_final['sex'] = 0
        df_final['age'] = 0
        df_final['weight'] = 0
        df_final['height'] = 0
        df_final['hb_a1c'] = 0
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

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

        df_summary = {'Number of patients':num_people,
                    'Number of males':num_male,
                    'Number of females': num_female,
                    'Minimum age': min_age,
                    'Maximum age': max_age,
                    'Number of people < 18 years': num_less_than_18,
                    'Number of people >= 18 years': num_18_or_older,
                    'C-Peptide interval': c_peptide_interval,
                    'BETA2 Score interval': beta2_interval,
                    'CGM data interval': cgm_interval}
        
        return df_final, df_summary

    def get_dataset(self, name):
        '''Retrieves a dataset by name.'''
        return self.datasets.get(name, None)
    
    def get_dataset_summary(self, name):
        return self.datasets_summary.get(name, None)
    
    def get_cgm_core_endpoints(self, df):
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

        df_cgm = df.copy()
        df_cgm = df_cgm[['id','timestamp','glucose mmol/l']]

        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
        df_grouped_cgm = df_cgm.groupby('id')
        results = {}

        for ptid, df_pt_cgm in df_grouped_cgm:
            df_pt_cgm['time_diff'] = df_pt_cgm['timestamp'].diff().dt.total_seconds()/3600
            median_interval = df_pt_cgm['time_diff'].median()
            df_pt_cgm['time_diff'] = df_pt_cgm['time_diff'].fillna(median_interval)

            total_time = df_pt_cgm['time_diff'].sum()
            
            in_range = (df_pt_cgm['glucose mmol/l'] >= 3.9) & (df_pt_cgm['glucose mmol/l'] <= 10.0)
            in_hypo_level_1_range = (df_pt_cgm['glucose mmol/l'] < 3.9)
            in_hypo_level_2_range = (df_pt_cgm['glucose mmol/l'] < 3.0)
            in_hyper_level_1_range = (df_pt_cgm['glucose mmol/l'] > 10.0)
            in_hyper_level_2_range = (df_pt_cgm['glucose mmol/l'] > 13.9)

            time_in_range = df_pt_cgm.loc[in_range, 'time_diff'].sum()
            time_below_range_level_1 = df_pt_cgm.loc[in_hypo_level_1_range, 'time_diff'].sum()
            time_below_range_level_2 = df_pt_cgm.loc[in_hypo_level_2_range, 'time_diff'].sum()
            time_above_range_level_1 = df_pt_cgm.loc[in_hyper_level_1_range, 'time_diff'].sum()
            time_above_range_level_2 = df_pt_cgm.loc[in_hyper_level_2_range, 'time_diff'].sum()

            mean_glucose = df_pt_cgm['glucose mmol/l'].mean()
            std_glucose = df_pt_cgm['glucose mmol/l'].std()
            cv_glucose = std_glucose / mean_glucose * 100

            max_wear_time = (df_pt_cgm['timestamp'].iloc[-1] - df_pt_cgm['timestamp'].iloc[0]).total_seconds() / 3600
            percent_wear_time = (total_time/max_wear_time) * 100

            results_pt = {
                'percent_wear_time' : percent_wear_time,
                'time_in_range_hours' : time_in_range,
                'percent_in_range' : time_in_range/total_time * 100,
                'time_below_range_level_1' : time_below_range_level_1,
                'percent_tbr_level_1' : time_below_range_level_1/total_time * 100,
                'time_below_range_level_2' : time_below_range_level_2,
                'percent_tbr_level_2' : time_below_range_level_2/total_time * 100,
                'time_above_range_level_1' : time_above_range_level_1,
                'percent_tar_level_1' : time_above_range_level_1/total_time * 100,
                'time_above_range_level_2' : time_above_range_level_2,
                'percent_tar_level_2' : time_above_range_level_2/total_time * 100,
                'mean_glucose' : mean_glucose,
                'std_glucose' : std_glucose,
                'cv_percent' : cv_glucose
            }

            results[ptid] = results_pt

        df_results = pd.DataFrame.from_dict(results, orient='index').reset_index(names='id')
        df_results = df_results.sort_values(by='id')

        return df_results
    
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

        return total_distance_t, total_distance_t/max(total_time_t,1)
    
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
    
    def statystical_analysis(self, csv_folder, df):
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

        for patient_id, patient_db in grouped_patients:
            length, norm_length = self.length_data(patient_db)
            level_1, level_2, cl_sig_events = self.hypoglycemia_rates(patient_db)
            glucose_avg = patient_db['glucose mmol/l'].mean()

            patients_analysis['id'].append(patient_id)
            patients_analysis['glucose_avg'].append(glucose_avg)
            patients_analysis['total_length'].append(length)
            patients_analysis['normalized_length'].append(norm_length)
            patients_analysis['hypoglycemia_level_1'].append(level_1)
            patients_analysis['hypoglycemia_level_2'].append(level_2)
            patients_analysis['cl_sig_hypo'].append(cl_sig_events)

        df_patients_analysis = pd.DataFrame(patients_analysis)
        df_patients_analysis = df_patients_analysis.loc[:, ~df_patients_analysis.columns.duplicated()]
        df_patients_analysis.to_csv(f'{csv_folder}/df_analysis.csv', index=False)

        return df_patients_analysis

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
    
    def plot_hypothesis_test(self, df_healthy, df_t1d, feature_name, test_results, path):
        '''Plots distributions of the given feature for healthy vs T1D groups'''
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
    print('CLOUD dataset summary:')
    print(data_analysis.get_dataset_summary('cloud'))
    print()

    print('CLVR dataset summary:')
    print(data_analysis.get_dataset_summary('clvr'))
    print()

    print('DEFEND-2 dataset summary:')
    print(data_analysis.get_dataset_summary('defend'))
    print()

    print('Diagnode dataset summary:')
    print(data_analysis.get_dataset_summary('diagnode'))
    print()

    print('GSKALB dataset summary:')
    print(data_analysis.get_dataset_summary('gskalb'))
    print()

    print('Islet transplant dataset summary:')
    print(data_analysis.get_dataset_summary('itx'))
    print()

    print('JAEB Healthy dataset summary:')
    print(data_analysis.get_dataset_summary('jaeb_healthy'))
    print()

    print('JAEB T1D dataset summary:')
    print(data_analysis.get_dataset_summary('jaeb_t1d'))
    print()

    '''Analyze datasets'''
    # If not analyzed before
    # analyzed_cloud = data_analysis.statystical_analysis('./data/cloud', df_cloud)
    # analyzed_clvr = data_analysis.statystical_analysis('./data/clvr', df_clvr)
    # analyzed_defend = data_analysis.statystical_analysis('./data/defend', df_defend)
    # analyzed_diagnode = data_analysis.statystical_analysis('./data/diagnode', df_diagnode)
    # analyzed_gskalb = data_analysis.statystical_analysis('./data/gskalb', df_gskalb)
    # analyzed_itx = data_analysis.statystical_analysis('./data/itx', df_itx)
    # analyzed_jaeb_healthy = data_analysis.statystical_analysis('./data/jaeb_healthy', df_clvr)
    # analyzed_jaeb_t1d = data_analysis.statystical_analysis('./data/jaeb_t1d', df_jaeb_t1d)

    #If analyzed before
    analyzed_cloud = pd.read_csv('./data/cloud/df_analysis.csv')
    analyzed_clvr = pd.read_csv('./data/clvr/df_analysis.csv')
    analyzed_defend = pd.read_csv('./data/defend/df_analysis.csv')
    analyzed_diagnode = pd.read_csv('./data/diagnode/df_analysis.csv')
    analyzed_gskalb = pd.read_csv('./data/gskalb/df_analysis.csv')
    analyzed_itx = pd.read_csv('./data/itx/df_analysis.csv')
    analyzed_jaeb_healthy = pd.read_csv('./data/jaeb_healthy/df_analysis.csv')
    analyzed_jaeb_t1d = pd.read_csv('./data/jaeb_t1d/df_analysis.csv')

    '''Hypoglycemia rates statistics '''
    # hypo_rate_statistics = data_analysis.hypoglycemia_rates_std_cv(df_diagnode)
    # print(hypo_rate_statistics)

    '''Get core endpoints '''
    cloud_core_endpoints = data_analysis.get_cgm_core_endpoints(df_cloud)
    clvr_core_endpoints = data_analysis.get_cgm_core_endpoints(df_clvr)
    defend_core_endpoints = data_analysis.get_cgm_core_endpoints(df_defend)
    diagnode_core_endpoints = data_analysis.get_cgm_core_endpoints(df_diagnode)
    gskalb_core_endpoints = data_analysis.get_cgm_core_endpoints(df_gskalb)
    itx_core_endpoints = data_analysis.get_cgm_core_endpoints(df_itx)
    jaeb_healthy_core_endpoints = data_analysis.get_cgm_core_endpoints(df_jaeb_healthy)
    jaeb_t1d_core_endpoints = data_analysis.get_cgm_core_endpoints(df_jaeb_t1d)

    '''Get Avg Core Endpoints '''
    #TODO: Fix for DataFrames
    # cloud_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(cloud_core_endpoints)
    # clvr_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(clvr_core_endpoints)
    # defend_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(defend_core_endpoints)
    # diagnode_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(diagnode_core_endpoints)
    # gskalb_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(gskalb_core_endpoints)
    # itx_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(itx_core_endpoints)
    # jaeb_healthy_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(jaeb_healthy_core_endpoints)
    # jaeb_t1d_avg_core_endpoints = data_analysis.get_cgm_avg_core_endpoints(jaeb_t1d_core_endpoints)

    # print(itx_avg_core_endpoints)

    '''Hypothesis test: Length of Line'''
    #Hypothesis test CLOUD vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_cloud, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_cloud, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_cloud.png')
    print('\nCLOUD vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test CLVR vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_clvr, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_clvr, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_clvr.png')
    print('\nCLVR vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test DEFEND-2 vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_defend, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_defend, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_defend.png')
    print('\nDEFEND-2 vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test JAEB T1D vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_jaeb_t1d, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_jaeb_t1d, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_jt1d.png')
    print('\nJAEB T1D vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test Diagnode vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_diagnode, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_diagnode, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_diagnode.png')
    print('\nDiagnode vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test GSKALB vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_gskalb, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_gskalb, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_gskalb.png')
    print('\nGSKALB vs JAEB Healthy LoL:')
    print(hypo_test_results)

    #Hypothesis test iTX vs JAEB Healthy LoL
    hypo_test_results = data_analysis.hypothesis_test(analyzed_jaeb_healthy, analyzed_itx, 'normalized_length')
    data_analysis.plot_hypothesis_test(analyzed_jaeb_healthy, analyzed_itx, 'normalized_length', hypo_test_results, './data/feature_analysis/LoL/jh_vs_itx.png')
    print('\niTX vs JAEB Healthy LoL:')
    print(hypo_test_results)

    '''Hypothesis test: CGM core endpoints'''
    for core_endpoints_feature in jaeb_healthy_core_endpoints.columns.drop('id'):
        #Hypothesis test CLOUD vs JAEB Healthy Average CGM core endpoints
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, cloud_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, cloud_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_cloud.png')
        
        #Hypothesis test CLVR vs JAEB Healthy Average CGM core endpoints 
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, clvr_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, clvr_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_clvr.png')

        #Hypothesis test DEFEND-2 vs JAEB Healthy Average CGM core endpoints 
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, defend_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, defend_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_defend.png')
        
        #Hypothesis test JAEB T1D vs JAEB Healthy Average CGM core endpoints
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, jaeb_t1d_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, jaeb_t1d_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_jt1d.png')
    
        #Hypothesis test Diagnode vs JAEB Healthy Average CGM core endpoints
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, diagnode_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, diagnode_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_diagnode.png')

        #Hypothesis test GSKALB vs JAEB Healthy Average CGM core endpoints 
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, gskalb_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, gskalb_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_gskalb.png')
        
        #Hypothesis test iTX vs JAEB Healthy Average CGM core endpoints
        hypo_test_results = data_analysis.hypothesis_test(jaeb_healthy_core_endpoints, itx_core_endpoints, core_endpoints_feature)
        data_analysis.plot_hypothesis_test(jaeb_healthy_core_endpoints, itx_core_endpoints, core_endpoints_feature, hypo_test_results, f'./data/feature_analysis/{core_endpoints_feature}/jh_vs_itx.png')    #If analyzed before