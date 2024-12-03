import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_data():
    db = pd.read_csv("./data/normalized/normalized_JL.csv")

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
        delta_glucose = db.loc[i, 'glucose mmol/l'] - db.loc[i - 1, 'glucose mmol/l']
        distance = np.sqrt(delta_time**2 + delta_glucose**2)
        distance_final += distance

    return distance_final, distance_final/len(db)

if __name__ == "__main__":
    db = pd.read_csv('./data/type_1/db_windowed_all.csv')
    grouped_patients = db.groupby('window_id')
    
    for window_id, patient_db in grouped_patients:
        patient_id = patient_db.iloc[0,2]
        beta2_score = patient_db.iloc[0,3]
        print(f'Analysis for patient {patient_id}, BETA2 Score: {beta2_score}:')

        length, norm_length = length_data(patient_db)
        print(f'\tTotal length is: {length}')
        print(f'\tNormalized length is: {norm_length}\n')