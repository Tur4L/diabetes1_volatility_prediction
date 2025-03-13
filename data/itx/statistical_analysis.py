import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_data():
    df = pd.read_csv("./data/normalized/normalized_JL.csv")

    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'],df['glucose mmol/l'])
    plt.title('Glucose level over time')
    plt.xlabel('Timestamps')
    plt.ylabel("Glucose Level mmol/l")
    plt.legend()
    plt.savefig('./data/normalized/jl_graphed.png')


def length_data(df):
    df = df.copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_interval = (df['timestamp'].diff().median()).seconds / 60
    
    if time_interval < 6:
        df = df.iloc[::3].reset_index(drop=True)

    total_diff_i = 0
    total_diff_t = 0
    valid_diffs = 0

    for i in range(1,df.shape[0]):
        time_interval = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 60

        if time_interval > 16:
            total_diff_t += total_diff_i
            total_diff_i = 0

        else:  
            glucose_diff = abs(df.iloc[i]['glucose mmol/l'] - df.iloc[i-1]['glucose mmol/l'])
            glucose_diff *= 18
            total_diff_i += glucose_diff
            valid_diffs += 1

    total_diff_t += total_diff_i

    return total_diff_t, total_diff_t/max(valid_diffs,1)
               
def hypoglycemia_rates(df):
    df = df.copy()
    df_hypo_level_1 = df[(df['glucose mmol/l'] < 3.9) & (df['glucose mmol/l'] >= 3.0)]
    df_hypo_level_2 = df[df['glucose mmol/l'] < 3.0]

    level_1 = df_hypo_level_1.shape[0]
    level_2 = df_hypo_level_2.shape[0]

    return level_1, level_2

if __name__ == "__main__":
    df = pd.read_csv('./data/itx/df_final.csv')
    grouped_patients = df.groupby('id')
    
    patients_analysis = {}
    patients_analysis['id'] = []
    patients_analysis['total_length'] = []
    patients_analysis['normalized_length'] = []
    patients_analysis['hypoglycemia_level_1'] = []
    patients_analysis['hypoglycemia_level_2'] = []

    for patient_id, patient_db in grouped_patients:
        length, norm_length = length_data(patient_db)
        level_1, level_2 = hypoglycemia_rates(patient_db)

        patients_analysis['id'].append(patient_id)
        patients_analysis['total_length'].append(length)
        patients_analysis['normalized_length'].append(norm_length)
        patients_analysis['hypoglycemia_level_1'].append(level_1)
        patients_analysis['hypoglycemia_level_2'].append(level_2)

    df_patients_analysis = pd.DataFrame(patients_analysis)
    df_patients_analysis.to_csv('./data/itx/df_analysis.csv', index=False)
