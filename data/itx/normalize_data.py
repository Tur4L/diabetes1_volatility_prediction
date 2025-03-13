import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from create_data import ITXData

scaler = MinMaxScaler()
itx_data_loader = ITXData()

def check_time_gaps(group):
    print(f"{group['window_id'].iloc[0]}: {(group['timestamp'].max() - group['timestamp'].min()).days + 1}")
    group = group.sort_values(by='timestamp')
    group['time_diff'] = group['timestamp'].diff().dt.days
    gaps = group[group['time_diff'] > 1]

    if not gaps.empty:
        print(f"\nGaps greater than 1 have been found: {group['window_id'].iloc[0]}")
        print(gaps[['timestamp','time_diff']])

    return gaps
    
def match_cgm_beta2(df_full, db_beta, db_transplant):
    '''This function is for implementing -/+ 30 days window from the beta2 score for the CGM data'''

    #Note: Currently we do not have access to the JL beta2 scores and dates, therefore, we will skip that data.
    db_match_full = df_full[df_full['patient_id'] != 1]
    db_match_beta = db_beta
    db_match_transplant = db_transplant[db_transplant['patient_id'] != 1]

    all_cgm_windowed = []
    grouped_match_full = db_match_full.groupby('patient_id')

    for _ , patient_beta2 in db_match_beta.iterrows():
        patient_id = patient_beta2['patient_id']
        patient_beta2_timestamp = patient_beta2['timestamp']
        patient_beta2_score = patient_beta2['BETA2_score']
        patient_cgm_data = grouped_match_full.get_group(patient_id)

        start_date = patient_beta2_timestamp - pd.Timedelta(days=30)
        end_date = patient_beta2_timestamp + pd.Timedelta(days=30)
        patient_cgm_windowed = patient_cgm_data[(patient_cgm_data['timestamp'] >= start_date) & (patient_cgm_data['timestamp'] <= end_date)]
        patient_cgm_windowed['BETA2_score'] = patient_beta2_score
        patient_cgm_windowed['BETA2_timestamp'] = patient_beta2_timestamp
        patient_cgm_windowed['window_id'] = patient_cgm_windowed['patient_id'].astype(str) + '_' + patient_cgm_windowed['BETA2_timestamp'].astype(str)
        all_cgm_windowed.append(patient_cgm_windowed)
       
    db_windowed = pd.concat(all_cgm_windowed, ignore_index = True)

    '''Dropping inconsistent and less-than-10-days windows'''
    # window_id: 5_2022-07-15, Reason: 5 days covered
    # window_id: 5_2023-01-12, Reason: 1 day covered
    # window_id: 5_2024-07-17, Reason: inconsistent

    db_windowed = db_windowed[~db_windowed['window_id'].isin(['5_2022-07-15','5_2023-01-12','5_2024-07-17'])]
    db_windowed.to_csv('./data/itx/db_windowed_all.csv', index=False)

    '''Choosing random window_id'''
    grouped_windows = db_windowed.groupby('window_id', group_keys=True)
    print('\nNumber of consecutive days')
    grouped_windows.apply(check_time_gaps)
    group_keys = list(grouped_windows.groups.keys())
    random_window = grouped_windows.get_group(np.random.choice(group_keys))

    '''Choosing random 10days window from random window_id'''
    db_10days_windows = window_10days(random_window)
    random_10days_window, end_date = random.choice(db_10days_windows)
    random_3days_window = window_3days(random_10days_window, end_date)

    random_10days_window.to_csv('./data/itx/db_window_random_10days.csv',index=False)
    random_3days_window.to_csv('./data/itx/db_window_random_3days.csv',index=False)
    
    return db_windowed


def interpolate_data(db_patient, db_beta):
    db_patient['date_only'] = db_patient['timestamp'].dt.date
    full_date_range = pd.date_range(start=db_beta['timestamp'].min(), end=db_patient['timestamp'].max(), freq='D')

    db_beta = db_beta.set_index('timestamp').reindex(full_date_range)
    db_beta['BETA2_score'] = db_beta['BETA2_score'].interpolate(method='linear')    
    db_patient['beta2_score'] = db_patient['date_only'].apply(lambda x: db_beta.loc[pd.to_datetime(x, format="%Y-%m-%d"), 'BETA2_score'])

    return db_patient

def patient_beta2_windows(db, patient_id):
    '''This function creates a new data base that consists of different beta2 windows of the same patient '''
    grouped_db = db.groupby('patient_id')
    patient_windows = grouped_db.get_group(patient_id)
    patient_windows_grouped = patient_windows.groupby('window_id')
    patient_all_windows = []

    for window_id, window in patient_windows_grouped:
        patient_10days_window = window_10days(window)
        patient_random_10days_window = random.choice(patient_10days_window)
        patient_all_windows.append(patient_random_10days_window[0])

    patient_windowed_final = pd.concat(patient_all_windows, ignore_index=True)
    patient_windowed_final.to_csv('./data/itx/db_windowed_patient.csv',index=False)

    return patient_windowed_final

def window_10days(selected_window):
    sub_windows = []

    start_date = selected_window['timestamp'].min()
    end_date = selected_window['timestamp'].max()

    current_start = start_date
    current_end = start_date
    while current_end <= end_date:
        current_end = current_start + pd.Timedelta(days=10)
        sub_window = selected_window[(selected_window['timestamp'] >= current_start) &
                                     (selected_window['timestamp'] < current_end)]
        
        if not sub_window.empty:
            sub_windows.append([sub_window,current_end])

        current_start += pd.Timedelta(days=1)

    return sub_windows

def window_3days(selected_window, end_date):
    
    start_date = end_date - pd.Timedelta(days=3)
    sub_window = selected_window[(selected_window['timestamp'] >= start_date) &
                                 (selected_window['timestamp'] < end_date)]

    return sub_window

def data_info(df_final):
    df_final.rename(columns={'AgeAsOfEnrollDt':'age', 'Weight':'weight', 'Height':'height',
                             'HbA1c':'hb_a1c', 'Gender': 'sex'}, inplace=True)

    no_dup_df = df_final.drop_duplicates(subset='id')

    num_people = no_dup_df['id'].size

    num_male = no_dup_df[no_dup_df['sex'] == 0].shape[0]
    num_female = no_dup_df[no_dup_df['sex'] == 1].shape[0]

    min_age = no_dup_df['age'].min()
    max_age = no_dup_df['age'].max()
    num_less_than_18 = no_dup_df[no_dup_df['age'] < 18].shape[0]
    num_18_or_older = no_dup_df[no_dup_df['age'] >= 18].shape[0]

    print('Islet transplant data info:\n')
    print(f'Number of patients: {num_people}')
    print()
    print(f'Number of male: {num_male}')
    print(f'Number of female: {num_female}')
    print()
    print(f'Minimum age: {min_age}')
    print(f'Maximum age age: {max_age}')
    print(f'Number of less than 18 years olds: {num_less_than_18}')
    print(f'Number of 18 or older years olds: {num_18_or_older}')
    print()
    print('C-Peptide availability: No\nC-Peptide interval: N/A')
    print('Beta function availability: Yes\nBeta function interval: Inconsistent ')
    print('CGM availability: Yes\nCGM device: Dexcom and FreeStyle Libre\nCGM interval: 5 and 15 minutes respectively')

def main():
    df_final = itx_data_loader.get_final_data()
    df_cgm = itx_data_loader.get_cgm_data()
    data_info(df_final)

    columns_to_scale = ['timestamp_seconds','scaled_glucose','age','weight','height','hb_a1c']
    df_final[columns_to_scale] = scaler.fit_transform(df_final[columns_to_scale])
    df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
    df_cgm.to_csv('./data/itx/df_cgm.csv', index=False)
    df_final.to_csv('./data/itx/df_final.csv', index=False)

    # windowed_db = match_cgm_beta2(full_df, full_df_beta, full_df_transplant)
    # patient_windowed_db = patient_beta2_windows(windowed_db,4)
 

if __name__ == "__main__":
    main()
