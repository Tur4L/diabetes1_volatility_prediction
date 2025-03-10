import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


scaler = MinMaxScaler()

def create_age_data():
    df_path = './data/jaeb_healthy/original/NonDiabPtRoster.csv'

    df_age = pd.read_csv(df_path)
    df_age.drop_duplicates(['RecID', 'PtID'], inplace=True)
    df_age.drop(['RecID','SiteID'], axis=1, inplace=True)

    # df_age.to_csv('./data/jaeb_healthy/df_age.csv')
    return df_age

def create_cgm_data():
    '''
    Creates CGM database
    '''

    df_path = './data/jaeb_healthy/original/NonDiabDeviceCGM.csv'
    df_cgm = pd.read_csv(df_path)
    df_cgm = df_cgm[df_cgm['RecordType'] == 'CGM']

    df_cgm['glucose mmol/l'] = df_cgm['Value'] / 18 #Converting mg/dl to mmol/l
    df_cgm['scaled_glucose'] = df_cgm['glucose mmol/l']
    df_cgm.loc[:, 'scaled_glucose'] = scaler.fit_transform(df_cgm[['scaled_glucose']])
    df_cgm.drop(columns=('Value'), axis=1)

    #Cleaning timestamp
    df_cgm.loc[:, 'timestamp_seconds'] = pd.to_timedelta(df_cgm['DeviceTm']).dt.total_seconds() + df_cgm['DeviceDtDaysFromEnroll'] * 86400
    df_cgm['time_diff'] = df_cgm.groupby('PtID')['timestamp_seconds'].diff()
    df_cons_intervals = df_cgm[(df_cgm['time_diff'] < 930) & (df_cgm['time_diff']>870)]

    initial_date = pd.to_datetime("2024-01-01")
    df_cgm['timestamp'] = initial_date + pd.to_timedelta(df_cgm['DeviceDtDaysFromEnroll'], unit='D')
    df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'].astype(str) + ' ' + df_cgm['DeviceTm'])

    const_ratio = df_cons_intervals.shape[0]/df_cgm.shape[0]
    print(const_ratio)

    df_cgm.drop(['RecordType', 'time_diff', 'DeviceTm', 'DeviceDtDaysFromEnroll', 'Value'], axis=1, inplace=True)

    df_cgm.to_csv('./data/jaeb_healthy/df_cgm.csv', index=False)
    return df_cgm

def create_logs_data():
    '''
    Creates participant logs database
    '''

    df_path = './data/jaeb_healthy/original/NonDiabParticipantLogs.csv'
    df_logs = pd.read_csv(df_path)
    df_logs.drop_duplicates(['RecID', 'PtID'], inplace=True)
    df_logs.drop(['RecID'], axis=1, inplace=True)

    # df_logs.to_csv('./data/jaeb_healthy/db_logs.csv', index=False)
    return df_logs

def create_medication_data():
    '''
    Creates medication database
    '''

    df_path = './data/jaeb_healthy/original/NonDiabMedication.csv'
    df_medication = pd.read_csv(df_path)
    df_medication.drop_duplicates(['RecID', 'PtID'], inplace=True)

    #No unknow dose; Only oral route; No eye or ear treatments; No medical conditions; No adverse events; No treatment started after enrollment;
    df_medication.drop(['MedDoseUnk', 'MedRoute', 'MedLocSide', 'MedicalCondition1', 'AdverseEvent1',
                        'MedStartDaysFromEnroll', 'MedStartDtApprox', 'MedStopDaysFromEnroll', 'MedStopDtApprox',
                        'MedStartMonthsAfterEnroll', 'MedStartYearsAfterEnroll', 'MedStopMonthsAfterEnroll',
                        'MedStopYearsAfterEnroll', 'MedStartDtUnk', 'MedCondNotReqd', 'MedCondNotReqd',
                        'AdvEventNotReqd', 'MedicalCondition2', 'PreExistingCondition2', 'AdverseEvent2', 'RecID',
                        'MedStartTrtCat', 'MedFreqUnk', 'PreExistCondNotReqd', 'MedStopDtUnk'], axis=1, inplace=True) 

    #PtID 7 takes medication only for prevention
    df_medication.loc[df_medication['PtID'] == 7, 'MedFreqNum'] = 0
    df_medication.loc[df_medication['PtID'] == 7, 'MedFreqPer'] = 'N/A'

    #PtID 38 has unknown end date of medication
    df_medication.loc[df_medication['PtID'] == 38, 'MedOngoing'] = 0


    ''' Medication encoding'''
    # df_medication['Medication_Num'] = df_medication.groupby('PtID').cumcount() + 1
    # df_medication = df_medication.pivot(index='PtID',columns='Medication_Num', values=['MedDose','MedUnit','MedFreqType','MedFreqNum','MedFreqPer','MedInd','MedOngoing','PreExistingCondition1','DrugName'])
    # df_medication.columns = [f'{col[0]}_{col[1]}' for col in df_medication.columns]
    # df_medication.reset_index(inplace=True)

    # medications_combined = df_medication[['DrugName_1', 'DrugName_2', 'DrugName_3', 'DrugName_4']].apply(lambda x: ','.join(x.dropna()), axis=1)
    # df_medications = pd.DataFrame(medications_combined.str.get_dummies(sep=','))
    # df_medication = pd.concat([df_medication[['PtID']], df_medications], axis=1)

    # df_medication = pd.get_dummies(df_medication, columns=['MedUnit','MedFreqType','MedFreqPer','MedInd','PreExistingCondition1','DrugName'])

    # df_medication.to_csv('./data/jaeb_healthy/df_medication.csv', index=False)
    return df_medication

def create_pre_conditions_data():
    '''
    Creates pre-existing conditions database
    '''

    df_path = './data/jaeb_healthy/original/NonDiabPreExistingCondition.csv'
    df_preconditions = pd.read_csv(df_path)
    df_preconditions.drop_duplicates(['RecID', 'PtID'])

    #No Medical Condition Status; No Recovery date; No approximation
    df_preconditions.drop(['MedCondStatus', 'MedCondResDtDaysFromEnroll', 'MedCondResDtApprox', 'RecID'], axis=1, inplace=True)
    df_preconditions = df_preconditions[~df_preconditions['PtID'].isin([14, 164])] #no CGM data available for ptid 14, 164.
    
    # df_preconditions = pd.get_dummies(df_preconditions, columns=['MedicalCondition','PreExistDuration'])
    # df_preconditions['PreExistMedCurr'] = df_preconditions['PreExistMedCurr'].map({'Yes' : 1, 'No': 0})

    # df_preconditions.to_csv('./data/jaeb_healthy/df_precondition.csv', index= False)
    return df_preconditions

def create_screening_data():
    '''
    Creates screening database
    '''

    df_path = './data/jaeb_healthy/original/NonDiabScreening.csv'
    df_screening = pd.read_csv(df_path)
    df_screening.drop_duplicates(['RecID', 'PtID'], inplace=True)

    #No need for ParentLoginVisitID; Visit is Baseline; InclCritMet and ExclCritAbsent is checked for all. Missing worktype for almost half of the patients.
    df_screening.drop(['ParentLoginVisitID', 'Visit', 'InclCritMet', 'ExclCritAbsent', 'RecID',
                       'HbA1cTestDtDaysFromEnroll', 'WorkType','Ethnicity',
                       'LastMenstCycStartDtDaysFromEnroll','LastMenstCycStartDtUnk',
                       'LastMenstCycStartDtNA','LastMenstCycEndDtDaysFromEnroll','LastMenstCycEndDtUnk',
                       'LastMenstCycEndDtNA','T1DBioFamParent','T1DBioFamSibling','T1DBioFamChild','T1DBioFamUnk',
                       'T1DBioFamily'], axis=1, inplace=True)
    
    df_screening['Gender'] = df_screening['Gender'].map({'M' : 0, 'F': 1})
    # df_screening.to_csv('./data/jaeb_healthy/df_screening.csv', index=False)
    return df_screening

def data_info(df_age, df_screening, df_medication, df_precondition):

    num_people = df_age['PtID'].size
    num_dropped = df_age[df_age['PtStatus'] == 'Dropped'].shape[0]
    num_completed = num_people - num_dropped

    num_male = df_screening[df_screening['Gender'] == 0].shape[0]
    num_female = df_screening[df_screening['Gender'] == 1].shape[0]
    num_races = df_screening['Race'].value_counts()

    min_age = df_age['AgeAsOfEnrollDt'].min()
    max_age = df_age['AgeAsOfEnrollDt'].max()
    num_less_than_18 = df_age[df_age['AgeAsOfEnrollDt'] < 18].shape[0]
    num_18_or_older = df_age[df_age['AgeAsOfEnrollDt'] >= 18].shape[0]

    print('JAEB data info:\n')
    print(f'Number of patients: {num_people}')
    print(f'Number of completed patients : {num_completed}')
    print()
    print(f'Number of male: {num_male}')
    print(f'Number of female: {num_female}')
    print()
    print(f'Minimum age: {min_age}')
    print(f'Maximum age age: {max_age}')
    print(f'Number of less than 18 years olds: {num_less_than_18}')
    print(f'Number of 18 or older years olds: {num_18_or_older}')
    print()
    print('C-Peptide availability: No\nC-Peptide nterval: N/A')
    print('Beta function availability: No\nBeta function interval: N/A ')
    print('CGM availability: Yes\nCGM device: Dexcom\nCGM interval: 5 minutes')
    print('\n\tRace division:')
    print(num_races)

    df_screening.drop(['Race'], axis=1, inplace=True)
    df_age = df_age[df_age['PtStatus'] == 'Completed']
    df_age.drop(['PtStatus'], axis=1, inplace=True)

def seperate_for_ages(df):
    less_than_18 = df[df['AgeAsOfEnrollDt'] < 18]
    greater_than_18 = df[df['AgeAsOfEnrollDt'] > 18]
    between_12_and_18 = df[(df['AgeAsOfEnrollDt'] < 18) & (df['AgeAsOfEnrollDt'] >= 12)]

    return less_than_18, greater_than_18, between_12_and_18

def main():
    df_cgm = create_cgm_data()
    df_medication = create_medication_data()
    df_logs = create_logs_data()
    df_precondition = create_pre_conditions_data()
    df_screening = create_screening_data()
    df_age = create_age_data()

    data_info(df_age, df_screening, df_medication, df_precondition) #prints info about the data

    df_final = pd.merge(df_cgm, df_age, on='PtID', how='inner')
    df_final = pd.merge(df_final, df_screening, on='PtID', how='inner')
    
    less_than_18, greater_than_18, between_12_and_18 = seperate_for_ages(df_final)

    columns_to_scale = ['timestamp_seconds','AgeAsOfEnrollDt','Weight','Height','HbA1c']
    df_final.loc[:, columns_to_scale] = scaler.fit_transform(df_final[columns_to_scale])
    less_than_18.loc[:, columns_to_scale] = scaler.fit_transform(less_than_18[columns_to_scale])
    greater_than_18.loc[:, columns_to_scale] = scaler.fit_transform(greater_than_18[columns_to_scale])
    between_12_and_18.loc[:, columns_to_scale] = scaler.fit_transform(between_12_and_18[columns_to_scale])

    df_final.rename(columns={'DeviceTm':'timestamp', 'Value':'glucose mmol/l', 'AgeAsOfEnrollDt':'age',
                            'Weight':'weight', 'Height':'height', 'HbA1c':'hb_a1c', 'PtID': 'id', 'Gender': 'sex'}, inplace=True)
    df_final = df_final[['id','timestamp','timestamp_seconds','glucose mmol/l', 'scaled_glucose', 'age', 'weight', 'height', 'sex', 'hb_a1c']]
    df_final.to_csv('./data/jaeb_healthy/df_final.csv',index=False)
    
    # less_than_18.to_csv('./data/jaeb_healthy/df_age_less_than_18.csv', index=False)
    # greater_than_18.to_csv('./data/jaeb_healthy/df_age_greater_than_18.csv',index=False)
    # between_12_and_18.to_csv('./data/jaeb_healthy/df_age_between_12_and_18.csv',index=False)

    return df_final

if __name__ == '__main__':
    main()