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

    return df_cgm

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

    return df_final

if __name__ == '__main__':
    df_final = main()