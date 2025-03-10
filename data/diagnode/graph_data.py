import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_data():
    db = pd.read_csv("./data/diagnode/df_final.csv")

    plt.figure(figsize=(10,5))
    plt.plot(db['timestamp'],db['glucose mmol/l'])
    plt.title('Glucose level over time')
    plt.xlabel('Timestamps')
    plt.ylabel("Glucose Level mmol/l")
    plt.legend()
    plt.savefig('./data/normalized/jl_graphed.png')

def length_data(db):
    db['timestamp'] = pd.to_datetime(db['timestamp'])
    db['time_minutes'] = (db['timestamp'] - db['timestamp'].min()).dt.total_seconds()/60
    distance_final = 0
    index = int(db.index[0])

    for i in range(index+1, index+len(db)):
        delta_time = db.loc[i, 'time_minutes'] - db.loc[i - 1, 'time_minutes']
        if delta_time > 16:
            return distance_final, distance_final/(i-index+1)

        delta_glucose = db.loc[i, 'glucose mmol/l'] - db.loc[i - 1, 'glucose mmol/l']
        distance = np.sqrt(delta_time**2 + delta_glucose**2)
        distance_final += distance

    return distance_final, distance_final/len(db)

if __name__ == "__main__":
    db = pd.read_csv('./data/diagnode/df_final.csv')
    grouped_patients = db.groupby('id')
    
    patients_analysis = {}
    patients_analysis['id'] = []
    patients_analysis['total_length'] = []
    patients_analysis['normalized_length'] = []
    for patient_id, patient_db in grouped_patients:
        length, norm_length = length_data(patient_db)

        patients_analysis['id'].append(patient_id)
        patients_analysis['total_length'].append(length)
        patients_analysis['normalized_length'].append(norm_length)

    df_patients_analysis = pd.DataFrame(patients_analysis)
    df_patients_analysis.to_csv('./data/diagnode/df_analysis.csv', index=False)