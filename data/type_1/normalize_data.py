import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
def check_time_gaps(group):
    print(f"{group['window_id'].iloc[0]}: {(group['timestamp'].max() - group['timestamp'].min()).days + 1}")
    group = group.sort_values(by='timestamp')
    group['time_diff'] = group['timestamp'].diff().dt.days
    gaps = group[group['time_diff'] > 1]

    if not gaps.empty:
        print(f"\nGaps greater than 1 have been found: {group['window_id'].iloc[0]}")
        print(gaps[['timestamp','time_diff']])

    return gaps

def create_dbs():
    '''This function creates necessary databases to get final interpolated database'''
    ''' Creating the CGM database'''

    jl_db = normalize_jl(pd.read_csv("./data/type_1/original/original_JL.csv"))
    js_db = normalize_js(pd.read_csv("./data/type_1/original/original_JS.csv"))
    jt_db = normalize_jt(pd.read_csv("./data/type_1/original/original_JT.csv"))
    kb_db = normalize_js(pd.read_csv("./data/type_1/original/original_KB.csv"))
    sn_db = normalize_js(pd.read_csv("./data/type_1/original/original_SN.csv"))
    cd_db = normalize_cd(pd.read_csv("./data/type_1/original/Clarity_Export_CD_2024-08-30_154812 copy.csv"))

    jl_db['patient_id'] = 1
    js_db['patient_id'] = 2
    jt_db['patient_id'] = 3
    kb_db['patient_id'] = 4
    sn_db['patient_id'] = 5
    cd_db['patient_id'] = 6

    full_db = pd.concat([jl_db, js_db, jt_db, kb_db, sn_db, cd_db])
    full_db.dropna(axis=0, subset=['timestamp'],inplace=True)

    ''' Creating beta2 database '''
    
    js_beta2 = {
    'timestamp': ['08/10/2020', '22/02/2021', '18/08/2021', '16/02/2022', '20/09/2022'],
    'BETA2_score': [10.05, 9.88, 10.11, 7.64, 8.57],
    'is_relevant': ['NO', 'NO', 'NO', 'YES', 'YES']
}
    jt_beta2 = {
    'timestamp': ['13/07/2020', '21/09/2020', '21/12/2020', '29/06/2021', '04/02/2022', '25/04/2023', '30/05/2024'],
    'BETA2_score': [4.65, 6.34, 5.91, 7.53, 2.03, 7.31, 5.31],
    'is_relevant': ['NO', 'NO', 'NO', 'NO', 'NO', 'YES', 'YES']
}
    kb_beta2 = {
    'timestamp': ['14/05/2020', '12/09/2020', '31/05/2021', '06/12/2021', '05/12/2022', '29/11/2023'],
    'BETA2_score': [9.97, 8.76, 7.29, 5.55, 4.54, 4.8],
    'is_relevant': ['NO', 'NO', 'NO', 'NO', 'YES', 'YES']
}
    sn_beta2 = {
    'timestamp': ['25/05/2020', '02/10/2020', '08/12/2021', '15/07/2022', '12/01/2023', '18/01/2024', '17/07/2024'],
    'BETA2_score': [3.84, 5.41, 5.1, 4.13, 5.83, 5.19, 4.35],
    'is_relevant': ['NO', 'NO', 'NO', 'YES', 'YES', 'YES', 'YES']
}
    cd_beta2 = {
    'timestamp': ['28/02/2024', '27/08/2024'],
    'BETA2_score': [12.98, 17.2],
    'is_relevant': ['YES', 'YES']
}  

    js_beta2_db = pd.DataFrame(js_beta2)
    jt_beta2_db = pd.DataFrame(jt_beta2)
    kb_beta2_db = pd.DataFrame(kb_beta2)
    sn_beta2_db = pd.DataFrame(sn_beta2)
    cd_beta2_db = pd.DataFrame(cd_beta2)

    js_beta2_db['patient_id'] = 2
    jt_beta2_db['patient_id'] = 3
    kb_beta2_db['patient_id'] = 4
    sn_beta2_db['patient_id'] = 5
    cd_beta2_db['patient_id'] = 6

    js_beta2_db['timestamp'] = pd.to_datetime(js_beta2_db['timestamp'], format='%d/%m/%Y')
    jt_beta2_db['timestamp'] = pd.to_datetime(jt_beta2_db['timestamp'], format='%d/%m/%Y')
    kb_beta2_db['timestamp'] = pd.to_datetime(kb_beta2_db['timestamp'], format='%d/%m/%Y')
    sn_beta2_db['timestamp'] = pd.to_datetime(sn_beta2_db['timestamp'], format='%d/%m/%Y')
    cd_beta2_db['timestamp'] = pd.to_datetime(cd_beta2_db['timestamp'], format='%d/%m/%Y')

    full_db_beta = pd.concat([js_beta2_db, jt_beta2_db, kb_beta2_db, sn_beta2_db, cd_beta2_db])

    ''' Creating the transplants database '''

    jl_transplant = {
    'timestamp': ['18/08/2022', '08/03/2023']
}
    js_transplant = {
    'timestamp': ['06/02/2015', '30/04/2015', '15/06/2015', '23/07/2015', '24/04/2016']
}
    jt_transplant = {
    'timestamp': ['12/07/2005', '14/10/2007', '24/04/2017']
}
    kb_transplant = {
    'timestamp': ['30/06/2016', '01/08/2016', '04/10/2018']
}
    sn_transplant = {
    'timestamp': ['19/03/2013', '23/10/2018', '05/08/2024']
}
    cd_transplant = {
    'timestamp': ['15/11/2023', '17/08/2024']
}
    
    jl_transplant_db = pd.DataFrame(jl_transplant)
    js_transplant_db = pd.DataFrame(js_transplant)
    jt_transplant_db = pd.DataFrame(jt_transplant)
    kb_transplant_db = pd.DataFrame(kb_transplant)
    sn_transplant_db = pd.DataFrame(sn_transplant)
    cd_transplant_db = pd.DataFrame(cd_transplant)

    jl_transplant_db['patient_id'] = 1
    js_transplant_db['patient_id'] = 2
    jt_transplant_db['patient_id'] = 3
    kb_transplant_db['patient_id'] = 4
    sn_transplant_db['patient_id'] = 5
    cd_transplant_db['patient_id'] = 6
    
    jl_transplant_db['timestamp'] = pd.to_datetime(jl_transplant_db['timestamp'], format='%d/%m/%Y')
    js_transplant_db['timestamp'] = pd.to_datetime(js_transplant_db['timestamp'], format='%d/%m/%Y')
    jt_transplant_db['timestamp'] = pd.to_datetime(jt_transplant_db['timestamp'], format='%d/%m/%Y')
    kb_transplant_db['timestamp'] = pd.to_datetime(kb_transplant_db['timestamp'], format='%d/%m/%Y')
    sn_transplant_db['timestamp'] = pd.to_datetime(sn_transplant_db['timestamp'], format='%d/%m/%Y')
    cd_transplant_db['timestamp'] = pd.to_datetime(cd_transplant_db['timestamp'], format='%d/%m/%Y')

    full_db_transplant = pd.concat([jl_transplant_db, js_transplant_db, jt_transplant_db, kb_transplant_db, sn_transplant_db, cd_transplant_db])

    return full_db, full_db_beta, full_db_transplant

def interpolate_data(db_patient, db_beta):
    db_patient['date_only'] = db_patient['timestamp'].dt.date
    full_date_range = pd.date_range(start=db_beta['timestamp'].min(), end=db_patient['timestamp'].max(), freq='D')

    db_beta = db_beta.set_index('timestamp').reindex(full_date_range)
    db_beta['BETA2_score'] = db_beta['BETA2_score'].interpolate(method='linear')    
    db_patient['beta2_score'] = db_patient['date_only'].apply(lambda x: db_beta.loc[pd.to_datetime(x, format="%Y-%m-%d"), 'BETA2_score'])

    return db_patient

def lstm_data(db_cgm, db_beta, db_transplants):
    db_final = db_cgm.copy()
    db_final = db_final[~(db_final['patient_id'] == 1)]

    #TODO Add 'Scaled_Value','AgeAsOfEnrollDt','Weight','Height','HbA1c','Gender'.
    db_final.rename(columns={'glucose mmol/l':'Value','timestamp':'DeviceTm', 'patient_id':'PtID'}, inplace=True)
    db_final['Scaled_Value'] = db_final[['Value']]
    db_final['AgeAsOfEnrollDt'] = 0
    db_final['Weight'] = 0.0
    db_final['Height'] = 0.0
    db_final['HbA1c'] = 0.0
    db_final['Gender'] = 0

    #Age added
    db_final.loc[db_final['PtID'] == 2, 'AgeAsOfEnrollDt'] = 47
    db_final.loc[db_final['PtID'] == 3, 'AgeAsOfEnrollDt'] = 34
    db_final.loc[db_final['PtID'] == 4, 'AgeAsOfEnrollDt'] = 42
    db_final.loc[db_final['PtID'] == 5, 'AgeAsOfEnrollDt'] = 45
    db_final.loc[db_final['PtID'] == 6, 'AgeAsOfEnrollDt'] = 60

    #Weight added
    #TODO

    #Height added
    #TODO

    #HbA1c added
    db_final.loc[db_final['PtID'] == 2, 'HbA1c'] = 8.5
    db_final.loc[db_final['PtID'] == 3, 'HbA1c'] = 8.0
    db_final.loc[db_final['PtID'] == 4, 'HbA1c'] = 7.5
    db_final.loc[db_final['PtID'] == 5, 'HbA1c'] = 7.6
    db_final.loc[db_final['PtID'] == 6, 'HbA1c'] = 0.0

    #Gender added
    db_final.loc[db_final['PtID'].isin([4,5]), 'Gender'] = 1

    #Fix DeviceTM to seconds. Start date being the first date entry
    #TODO
    db_final['DeviceTm'] = pd.to_datetime(db_final['DeviceTm'])
    db_final['DeviceTm'] = db_final.groupby('PtID')['DeviceTm'].transform(
        lambda x: (x-x.min()).dt.total_seconds())

    return db_final
    
def match_cgm_beta2(db_full, db_beta, db_transplant):
    '''This function is for implementing -/+ 30 days window from the beta2 score for the CGM data'''

    #Note: Currently we do not have access to the JL beta2 scores and dates, therefore, we will skip that data.
    db_match_full = db_full[db_full['patient_id'] != 1]
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
    db_windowed.to_csv('./data/type_1/db_windowed_all.csv', index=False)

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

    random_10days_window.to_csv('./data/type_1/db_window_random_10days.csv',index=False)
    random_3days_window.to_csv('./data/type_1/db_window_random_3days.csv',index=False)
    
    return db_windowed

def normalize_jl(db):
    ''' Cleans csv file of the patient JL'''

    #creating initial db
    normalized_db = db[["timestamp","sensorglucose"]]
    normalized_db.columns = ["timestamp","glucose mmol/l"]
    normalized_db = normalized_db.dropna(subset=['glucose mmol/l'])

    #converting timestamps
    normalized_db["timestamp"] = pd.to_datetime(normalized_db["timestamp"], format='%Y-%m-%dT%H:%M:%SZ')
    normalized_db = normalized_db.sort_values(by="timestamp")

    #normalizing glucose levels
    normalized_db["glucose mmol/l"] = normalized_db["glucose mmol/l"].astype(float)
    return normalized_db

def normalize_js(db):
    ''' Cleans csv file of the patient JS, KB, and SN'''

    #creating initial db
    normalized_db = db.iloc[2:,[2,4]]
    normalized_db.columns = ["timestamp","glucose mmol/l"]
    normalized_db = normalized_db.dropna(subset=['glucose mmol/l'])

    #converting timestamps 
    normalized_db["timestamp"] = pd.to_datetime(normalized_db["timestamp"], format='%d-%m-%Y %H:%M')
    normalized_db = normalized_db.sort_values(by="timestamp")

    #normalizing glucose levels
    normalized_db["glucose mmol/l"] = normalized_db["glucose mmol/l"].astype(float)
    return normalized_db

def normalize_jt(db):
    ''' Cleans csv file of the patient JT'''

    #creating initial db
    normalized_db = db.iloc[2:,[2,4]]
    normalized_db.columns = ["timestamp","glucose mmol/l"]
    normalized_db = normalized_db.dropna(subset=['glucose mmol/l'])

    #converting timestamps 
    normalized_db["timestamp"] = pd.to_datetime(normalized_db["timestamp"], format='%d-%m-%Y %H:%M')
    normalized_db = normalized_db.sort_values(by="timestamp")
    # normalized_db = normalized_db[(normalized_db["timestamp"]>="2024-01-04") & (normalized_db['timestamp']<= "2024-08-31")]

    #normalizing glucose levels
    normalized_db["glucose mmol/l"] = normalized_db["glucose mmol/l"].astype(float)
    return normalized_db

def normalize_cd(db):
    ''' Cleans csv file of the patient CD'''

    #creating initial db
    normalized_db = db.iloc[:,[1,7]]
    normalized_db.columns = ["timestamp", "glucose mmol/l"]
    normalized_db = normalized_db.dropna(subset=['glucose mmol/l'])

    #converting timestamps
    normalized_db["timestamp"] = pd.to_datetime(normalized_db["timestamp"], format='%Y-%m-%dT%H:%M:%S')
    normalized_db = normalized_db.sort_values(by="timestamp")

    #normalizing glucose levels
    normalized_db["glucose mmol/l"] = normalized_db["glucose mmol/l"].astype(float)
    return normalized_db

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
    patient_windowed_final.to_csv('./data/type_1/db_windowed_patient.csv',index=False)

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

def data_info(db_final):

    no_dup_df = db_final.drop_duplicates(subset='PtID')

    num_people = no_dup_df['PtID'].size

    num_male = no_dup_df[no_dup_df['Gender'] == 0].shape[0]
    num_female = no_dup_df[no_dup_df['Gender'] == 1].shape[0]

    min_age = no_dup_df['AgeAsOfEnrollDt'].min()
    max_age = no_dup_df['AgeAsOfEnrollDt'].max()
    num_less_than_18 = no_dup_df[no_dup_df['AgeAsOfEnrollDt'] < 18].shape[0]
    num_18_or_older = no_dup_df[no_dup_df['AgeAsOfEnrollDt'] >= 18].shape[0]

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
    full_db, full_db_beta, full_db_transplant = create_dbs()
    df_final = lstm_data(full_db, full_db_beta.drop(['timestamp','is_relevant'], axis=1), full_db_transplant)
    data_info(df_final)

    columns_to_scale = ['DeviceTm','Scaled_Value','AgeAsOfEnrollDt','Weight','Height','HbA1c']
    df_final[columns_to_scale] = scaler.fit_transform(db_final[columns_to_scale])
    db_final.to_csv('./data/type_1/df_final.csv', index=False)


    # windowed_db = match_cgm_beta2(full_db, full_db_beta, full_db_transplant)
    # patient_windowed_db = patient_beta2_windows(windowed_db,4)
 

if __name__ == "__main__":
    main()