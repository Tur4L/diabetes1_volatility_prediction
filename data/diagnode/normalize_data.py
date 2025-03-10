import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def clean_aljc(df_aljc: pd.DataFrame) -> pd.DataFrame:
    df_aljc = df_aljc[['id','visit','treatment','sex','weight','height','age','hb_a1c',
                       'c_peptide_fasting','glucose_fasting']]
    return df_aljc

def clean_cgm(df_cgm: pd.DataFrame) -> pd.DataFrame:
    df_cgm[['id','visit','device_id']] = df_cgm['ID_VISIT_DEVICEID'].str.split('_',expand=True)
    df_cgm = df_cgm[['id','visit','TIMESTAMP','GLUCOSE']]
    df_cgm.rename(columns={'TIMESTAMP': 'timestamp', 'GLUCOSE': 'glucose mmol/l'}, inplace=True)
    df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'], format = "%Y-%m-%d %H:%M:%S", errors ='coerce')
    df_cgm['timestamp_seconds'] = df_cgm.groupby('id')['timestamp'].transform(
        lambda x: (x-x.min()).dt.total_seconds())

    df_cgm.to_csv('./data/diagnode/df_cgm.csv')

    return df_cgm

def data_info(df_aljc, df_cgm):

    no_dup_df = df_aljc.drop_duplicates(subset='id')

    num_people = no_dup_df.shape[0]
    num_total_visits = df_aljc.shape[0]

    num_male = no_dup_df[no_dup_df['sex'] == 'Male'].shape[0]
    num_female = no_dup_df[no_dup_df['sex'] == 'Female'].shape[0]
    # num_races = db_screening['Race'].value_counts()

    min_age = no_dup_df['age'].min()
    max_age = no_dup_df['age'].max()
    num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
    num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

    print('Diagnode data info:\n')
    print(f'Number of patients: {num_people}')
    print(f'Number of total visits: {num_total_visits}')
    print()
    print(f'Number of male: {num_male}')
    print(f'Number of female: {num_female}')
    print()
    print(f'Minimum age: {min_age}')
    print(f'Maximum age age: {max_age}')
    print(f'Number of less than 18 years olds: {num_less_than_18}')
    print(f'Number of 18 or older years olds: {num_18_or_older}')
    print()
    print('C-Peptide availability: Yes\nC-Peptide interval: 30 minutes')
    print('Beta2 score availability: Yes\nBeta2 score interval:0-6-15-24 months')
    print('CGM availability: Yes\nCGM device: Maybe FreeStyle Libre\nCGM interval: 15 minutes')
    # print('\n\tRace division:')
    # print(num_races)

def match_cgm_aljc(df_cgm: pd.DataFrame, df_aljc: pd.DataFrame) -> pd.DataFrame:
    df_cgm = clean_cgm(df_cgm)
    df_aljc = clean_aljc(df_aljc)

    df_final = pd.merge(df_cgm, df_aljc, on=['id','visit'], how='inner')
    df_final.to_csv('./data/diagnode/df_final.csv', index=False)

    return df_final
 
def main():
    df_cgm = pd.read_csv('./data/diagnode/original/CGM_RANDOMISED_preprocessed.csv')
    df_aljc = pd.read_csv('./data/diagnode/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')

    df_final = match_cgm_aljc(df_cgm, df_aljc)
    data_info(df_aljc, df_cgm)

    return df_final

if __name__ == '__main__':
    df_final = main()