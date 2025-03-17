import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    "diagnode": {"csv_folder": "./data/diagnode", "preprocess": "preprocess_diagnode"},
    "itx": {"csv_folder": "./data/itx", "preprocess": "preprocess_itx"},
    "jaeb_healthy": {"csv_folder": "./data/jaeb_healthy", "preprocess": "preprocess_jaeb_healthy"},
    "jaeb_t1d": {"csv_folder": "./data/jaeb_t1d", "preprocess": "preprocess_jaeb_t1d"},
}

class DataAnalysis:
    def __init__(self):
        """Initializes and loads all datasets dynamically."""
        self.datasets = {}
        self.load_data()

    def load_data(self): 
        """Loads, cleans, and normalizes all datasets from CSV files."""
        for dataset_name, config in DATASET_CONFIG.items():
            csv_folder = config["csv_folder"]
            if not os.path.exists(csv_folder):
                print(f"Warning: {csv_folder} not found. Skipping {dataset_name}.")
                continue

            print(f"Loading {dataset_name} from {csv_folder}...")
            df = self._preprocess_data(csv_folder, config["preprocess"])
            self.datasets[dataset_name] = df

    def _preprocess_data(self, csv_folder, preprocess_func_name):
        """
        Dynamically loads the preprocessing function from either the current class
        or the ITX package.
        """
        preprocess_func = getattr(self, preprocess_func_name, None)

        if preprocess_func:
            return preprocess_func(csv_folder)

    def preprocess_cloud(self, csv_folder):
        """Custom preprocessing for CLOUD dataset."""
        #TODO
        
        return None

    def preprocess_clvr(self, csv_folder):
        """Custom preprocessing for CLVR dataset."""
        #TODO
        
        return None

    def preprocess_diagnode(self, csv_folder):
        """Custom preprocessing for DIAGNODE dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/CGM_RANDOMISED_preprocessed.csv')

        df_cgm[['id','visit','device_id']] = df_cgm['ID_VISIT_DEVICEID'].str.split('_',expand=True)
        df_cgm = df_cgm[['id','visit','TIMESTAMP','GLUCOSE']]
        df_cgm.rename(columns={'TIMESTAMP': 'timestamp', 'GLUCOSE': 'glucose mmol/l'}, inplace=True)

        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format = "%Y-%m-%d %H:%M:%S", errors ='coerce')
        df_cgm.sort_values(by=['id','timestamp'], ascending=True, inplace=True)
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x-x.min()).dt.total_seconds())

        df_cgm['time_diff'] = df_cgm.groupby('id')['timestamp_seconds'].diff()
        # df_cons_intervals = df_cgm[(df_cgm['time_diff'] < 930) & (df_cgm['time_diff']>870)]
        # const_ratio = df_cons_intervals.shape[0]/df_cgm.shape[0]
        # print(const_ratio)

        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm['scaled_glucose'] = scaler.fit_transform(df_cgm[['scaled_glucose']])
        df_cgm.to_csv('./data/diagnode/df_cgm.csv', index=False)

        df_cgm.drop(['time_diff'], axis=1, inplace=True)

        #Extra feature
        df_aljc = pd.read_csv(f'{csv_folder}/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')
        df_aljc = df_aljc[['id','visit','treatment','sex','weight','height','age','hb_a1c',
                       'c_peptide_fasting','glucose_fasting']]

        #Merged dataframe
        df_final = pd.merge(df_cgm, df_aljc, on=['id','visit'], how='inner')
        df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        #TODO: data_info

        return df_final
    
    def preprocess_itx(self, csv_folder):
        """Custom preprocessing for Islet Transplant dataset"""
        df_cgm = itx_dataloader.get_cgm_data()
        df_final = itx_dataloader.get_final_data()


        df_cgm.to_csv(f'{csv_folder}/df_cgm.csv', index=False)
        df_final.to_csv(f'{csv_folder}/df_final.csv', index=False)

        return df_final    

    def preprocess_jaeb_healthy(self, csv_folder):
        """Custom preprocessing for JAEB healthy dataset."""

        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/NonDiabDeviceCGM.csv')
        df_cgm = df_cgm[df_cgm['RecordType'] == 'CGM']

        df_cgm['glucose mmol/l'] = df_cgm['Value'] / 18 #Converting mg/dl to mmol/l
        df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
        df_cgm.loc[:, 'scaled_glucose'] = scaler.fit_transform(df_cgm[['scaled_glucose']])
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

        #Adding additional features
        df_screening = pd.read_csv(f'{csv_folder}/original/NonDiabScreening.csv')
        df_screening.drop_duplicates(['RecID', 'PtID'], inplace=True)

        df_screening = df_screening[['PtID','Weight', 'Height', 'HbA1c', 'Gender', 'Race']]
        df_screening['Gender'] = df_screening['Gender'].map({'M' : 0, 'F': 1})
        df_screening.to_csv(f'{csv_folder}/df_screening.csv', index=False)

        df_final = pd.merge(df_cgm, df_age, on='PtID', how='inner')
        df_final = pd.merge(df_final, df_screening, on='PtID', how='inner')
        

        columns_to_scale = ['timestamp_seconds','AgeAsOfEnrollDt','Weight','Height','HbA1c']
        df_final.loc[:, columns_to_scale] = scaler.fit_transform(df_final[columns_to_scale])

        df_final.rename(columns={'DeviceTm':'timestamp', 'Value':'glucose mmol/l', 'AgeAsOfEnrollDt':'age',
                                'Weight':'weight', 'Height':'height', 'HbA1c':'hb_a1c', 'PtID': 'id', 'Gender': 'sex'}, inplace=True)
        df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
        df_final.to_csv('./data/jaeb_healthy/df_final.csv',index=False)

        #TODO data_info

        return df_final

    def preprocess_jaeb_t1d(self, csv_folder):
        '''Custom preprocessing for JABE Type 1 Diabetes Dataset'''
        #CGM data
        df_cgm = pd.read_csv(f'{csv_folder}/original/2025-03-13_T1D_JAEB_C-path_CGM_clean.csv')
        df_cgm.rename(columns={'PtID':'id','DeviceTm': 'timestamp', 'Value' : 'glucose mmol/l'}, inplace=True)
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
        df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds() if not x.isnull().all() else np.nan)
        
        #TODO: Add other features once i get them
        df_final = df_cgm
        df_final.to_csv(f'{csv_folder}/df_final.csv')
        return df_final

    def get_dataset(self, name):
        """Retrieves a dataset by name."""
        return self.datasets.get(name, None)

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

        return total_distance_t, total_distance_t/max(valid_diffs,1)
    
    def hypoglycemia_rates(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_hypo_level_1 = df[(df['glucose mmol/l'] < 3.9) & (df['glucose mmol/l'] >= 3.0)]
        df_hypo_level_2 = df[df['glucose mmol/l'] < 3.0]

        level_1 = df_hypo_level_1.shape[0]
        level_2 = df_hypo_level_2.shape[0]
        
        #TODO: Add clinically significant hypos as feature.
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
    
    def statystical_analysis(self, csv_folder, df):
        df = df.copy()
        grouped_patients = df.groupby('id')
    
        patients_analysis = {}
        patients_analysis['id'] = []
        patients_analysis['total_length'] = []
        patients_analysis['normalized_length'] = []
        patients_analysis['hypoglycemia_level_1'] = []
        patients_analysis['hypoglycemia_level_2'] = []
        patients_analysis['clinically_significant_hypoglycemia'] = []

        for patient_id, patient_db in grouped_patients:
            length, norm_length = self.length_data(patient_db)
            level_1, level_2, cl_sig_events = self.hypoglycemia_rates(patient_db)

            patients_analysis['id'].append(patient_id)
            patients_analysis['total_length'].append(length)
            patients_analysis['normalized_length'].append(norm_length)
            patients_analysis['hypoglycemia_level_1'].append(level_1)
            patients_analysis['hypoglycemia_level_2'].append(level_2)
            patients_analysis['clinically_significant_hypoglycemia'].append(cl_sig_events)

        df_patients_analysis = pd.DataFrame(patients_analysis)
        df_patients_analysis.to_csv(f'{csv_folder}/df_analysis.csv', index=False)


if __name__ == '__main__':
    data_analysis = DataAnalysis()

    #Analyze Diagnode dataset
    df_diagnode = data_analysis.get_dataset('diagnode')
    data_analysis.statystical_analysis('./data/diagnode', df_diagnode)

    #Analyze Islet Transplant dataset
    df_itx = data_analysis.get_dataset('itx')
    data_analysis.statystical_analysis('./data/itx', df_itx)

    #Analyze JAEB Healthy dataset
    df_jaeb_healthy = data_analysis.get_dataset('jaeb_healthy')
    data_analysis.statystical_analysis('./data/jaeb_healthy', df_jaeb_healthy)

    #Analyze JAEB Type 1 Diabetes dataset
    df_jaeb_t1d = data_analysis.get_dataset('jaeb_t1d')
    data_analysis.statystical_analysis('./data/jaeb_t1d', df_jaeb_t1d)

    pass