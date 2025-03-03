import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


scaler = MinMaxScaler()
def create_cgm_data():
    '''
    Creates CGM database
    '''

    db_path = './data/normal/original/NonDiabDeviceCGM.csv'
    db_cgm = pd.read_csv(db_path)
    db_cgm = db_cgm[db_cgm['RecordType'] == 'CGM']
    db_cgm['Value'] = db_cgm['Value'] / 18 #Converting mg/dl to mmol/l
    db_cgm['DeviceTm'] = pd.to_timedelta(db_cgm['DeviceTm']).dt.total_seconds() + db_cgm['DeviceDtDaysFromEnroll'] * 86400
    db_cgm['Scaled_Value'] = db_cgm['Value']

    db_cgm.drop(['RecordType','DeviceDtDaysFromEnroll'], axis=1, inplace=True)

    db_cgm.to_csv('./data/normal/db_cgm.csv', index=False)
    return db_cgm

def create_medication_data():
    '''
    Creates medication database
    '''

    db_path = './data/normal/original/NonDiabMedication.csv'
    db_medication = pd.read_csv(db_path)
    db_medication.drop_duplicates(['RecID', 'PtID'], inplace=True)

    #No unknow dose; Only oral route; No eye or ear treatments; No medical conditions; No adverse events; No treatment started after enrollment;
    db_medication.drop(['MedDoseUnk', 'MedRoute', 'MedLocSide', 'MedicalCondition1', 'AdverseEvent1',
                        'MedStartDaysFromEnroll', 'MedStartDtApprox', 'MedStopDaysFromEnroll', 'MedStopDtApprox',
                        'MedStartMonthsAfterEnroll', 'MedStartYearsAfterEnroll', 'MedStopMonthsAfterEnroll',
                        'MedStopYearsAfterEnroll', 'MedStartDtUnk', 'MedCondNotReqd', 'MedCondNotReqd',
                        'AdvEventNotReqd', 'MedicalCondition2', 'PreExistingCondition2', 'AdverseEvent2', 'RecID',
                        'MedStartTrtCat', 'MedFreqUnk', 'PreExistCondNotReqd', 'MedStopDtUnk'], axis=1, inplace=True) 

    #PtID 7 takes medication only for prevention
    db_medication.loc[db_medication['PtID'] == 7, 'MedFreqNum'] = 0
    db_medication.loc[db_medication['PtID'] == 7, 'MedFreqPer'] = 'N/A'

    #PtID 38 has unknown end date of medication
    db_medication.loc[db_medication['PtID'] == 38, 'MedOngoing'] = 0


    ''' Medication encoding'''
    # db_medication['Medication_Num'] = db_medication.groupby('PtID').cumcount() + 1
    # db_medication = db_medication.pivot(index='PtID',columns='Medication_Num', values=['MedDose','MedUnit','MedFreqType','MedFreqNum','MedFreqPer','MedInd','MedOngoing','PreExistingCondition1','DrugName'])
    # db_medication.columns = [f'{col[0]}_{col[1]}' for col in db_medication.columns]
    # db_medication.reset_index(inplace=True)

    # medications_combined = db_medication[['DrugName_1', 'DrugName_2', 'DrugName_3', 'DrugName_4']].apply(lambda x: ','.join(x.dropna()), axis=1)
    # df_medications = pd.DataFrame(medications_combined.str.get_dummies(sep=','))
    # db_medication = pd.concat([db_medication[['PtID']], df_medications], axis=1)

    # db_medication = pd.get_dummies(db_medication, columns=['MedUnit','MedFreqType','MedFreqPer','MedInd','PreExistingCondition1','DrugName'])

    db_medication.to_csv('./data/normal/db_medication.csv', index=False)
    return db_medication

def create_logs_data():
    '''
    Creates participant logs database
    '''

    db_path = './data/normal/original/NonDiabParticipantLogs.csv'
    db_logs = pd.read_csv(db_path)
    db_logs.drop_duplicates(['RecID', 'PtID'], inplace=True)
    db_logs.drop(['RecID'], axis=1, inplace=True)

    db_logs.to_csv('./data/normal/db_logs.csv', index=False)
    return db_logs

def create_pre_conditions_data():
    '''
    Creates pre-existing conditions database
    '''

    db_path = './data/normal/original/NonDiabPreExistingCondition.csv'
    db_preconditions = pd.read_csv(db_path)
    db_preconditions.drop_duplicates(['RecID', 'PtID'])

    #No Medical Condition Status; No Recovery date; No approximation
    db_preconditions.drop(['MedCondStatus', 'MedCondResDtDaysFromEnroll', 'MedCondResDtApprox', 'RecID'], axis=1, inplace=True)
    db_preconditions = db_preconditions[~db_preconditions['PtID'].isin([14, 164])] #no CGM data available for ptid 14, 164.
    
    # db_preconditions = pd.get_dummies(db_preconditions, columns=['MedicalCondition','PreExistDuration'])
    # db_preconditions['PreExistMedCurr'] = db_preconditions['PreExistMedCurr'].map({'Yes' : 1, 'No': 0})

    db_preconditions.to_csv('./data/normal/db_precondition.csv', index= False)
    return db_preconditions

def create_screening_data():
    '''
    Creates screening database
    '''

    db_path = './data/normal/original/NonDiabScreening.csv'
    db_screening = pd.read_csv(db_path)
    db_screening.drop_duplicates(['RecID', 'PtID'], inplace=True)

    #No need for ParentLoginVisitID; Visit is Baseline; InclCritMet and ExclCritAbsent is checked for all. Missing worktype for almost half of the patients.
    db_screening.drop(['ParentLoginVisitID', 'Visit', 'InclCritMet', 'ExclCritAbsent', 'RecID',
                       'HbA1cTestDtDaysFromEnroll', 'WorkType','Ethnicity',
                       'LastMenstCycStartDtDaysFromEnroll','LastMenstCycStartDtUnk',
                       'LastMenstCycStartDtNA','LastMenstCycEndDtDaysFromEnroll','LastMenstCycEndDtUnk',
                       'LastMenstCycEndDtNA'], axis=1, inplace=True)
    
    db_screening['Gender'] = db_screening['Gender'].map({'M' : 0, 'F': 1})
    db_screening['T1DBioFamily'] = db_screening['T1DBioFamily'].map({'No' : 0, 'Yes': 1})
    db_screening[['T1DBioFamParent','T1DBioFamSibling','T1DBioFamChild','T1DBioFamUnk']] = db_screening[['T1DBioFamParent','T1DBioFamSibling','T1DBioFamChild','T1DBioFamUnk']].fillna(0)

    db_screening.to_csv('./data/normal/db_screening.csv', index=False)
    return db_screening

def create_age_data():
    db_path = './data/normal/original/NonDiabPtRoster.csv'

    db_age = pd.read_csv(db_path)
    db_age.drop_duplicates(['RecID', 'PtID'], inplace=True)
    db_age.drop(['RecID','SiteID'], axis=1, inplace=True)

    db_age.to_csv('./data/normal/db_age.csv')
    return db_age

def seperate_for_ages(db):
    less_than_18 = db[db['AgeAsOfEnrollDt'] < 18]
    greater_than_18 = db[db['AgeAsOfEnrollDt'] > 18]
    between_12_and_18 = db[(db['AgeAsOfEnrollDt'] < 18) & (db['AgeAsOfEnrollDt'] >= 12)]

    return less_than_18, greater_than_18, between_12_and_18

def data_info(db_age, db_screening, db_medication, db_precondition):

    num_people = db_age['PtID'].size
    num_dropped = db_age[db_age['PtStatus'] == 'Dropped'].shape[0]
    num_completed = num_people - num_dropped

    num_male = db_screening[db_screening['Gender'] == 0].shape[0]
    num_female = db_screening[db_screening['Gender'] == 1].shape[0]
    num_races = db_screening['Race'].value_counts()

    min_age = db_age['AgeAsOfEnrollDt'].min()
    max_age = db_age['AgeAsOfEnrollDt'].max()
    num_less_than_18 = db_age[db_age['AgeAsOfEnrollDt'] < 18].shape[0]
    num_18_or_older = db_age[db_age['AgeAsOfEnrollDt'] >= 18].shape[0]

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

def main():
    db_cgm = create_cgm_data()
    db_medication = create_medication_data()
    db_logs = create_logs_data()
    db_precondition = create_pre_conditions_data()
    db_screening = create_screening_data()
    db_age = create_age_data()

    data_info(db_age, db_screening, db_medication, db_precondition)

    db_screening.drop(['Race'], axis=1, inplace=True)
    db_age = db_age[db_age['PtStatus'] == 'Completed']
    db_age.drop(['PtStatus'], axis=1, inplace=True)

    db_final = pd.merge(db_cgm, db_age, on='PtID', how='inner')
    db_final = pd.merge(db_final, db_screening, on='PtID', how='inner')
    less_than_18, greater_than_18, between_12_and_18 = seperate_for_ages(db_final)

    columns_to_scale = ['DeviceTm','Scaled_Value','AgeAsOfEnrollDt','Weight','Height','HbA1c']
    db_final.loc[:, columns_to_scale] = scaler.fit_transform(db_final[columns_to_scale])
    less_than_18.loc[:, columns_to_scale] = scaler.fit_transform(less_than_18[columns_to_scale])
    greater_than_18.loc[:, columns_to_scale] = scaler.fit_transform(greater_than_18[columns_to_scale])
    between_12_and_18.loc[:, columns_to_scale] = scaler.fit_transform(between_12_and_18[columns_to_scale])

    db_final.to_csv('./data/normal/db_final.csv',index=False)
    less_than_18.to_csv('./data/normal/db_age_less_than_18.csv', index=False)
    greater_than_18.to_csv('./data/normal/db_age_greater_than_18.csv',index=False)
    between_12_and_18.to_csv('./data/normal/db_age_between_12_and_18.csv',index=False)

    return db_final

if __name__ == '__main__':
    main()