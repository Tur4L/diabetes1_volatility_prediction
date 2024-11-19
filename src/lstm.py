import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import *


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

WINDOW_SIZE = 40
def seq_data(df, window_size):
    """
    Create sequential numpy data from df for lstm to process
    Parameters:
    - df: dataframe
        df of patient data
    - window_size: int
        how far back to look

    Returns:
    - X: 
        Numpy of X input
    - y:
        Numpy of y input

    """

    counts = df.groupby(['PtID']).size()
    label_data = df['Value'].to_numpy()
    input_data = df.to_numpy()
    frequencies = counts.to_numpy()
    X = {}
    y = {}
    current_index = 0

    for freq in frequencies:
        if freq > window_size+5:
            for i in range(freq-window_size-5):
                pt_id = input_data[current_index][0]
                label = label_data[i+window_size+current_index+5]
                row = [a for a in input_data[i + current_index: i + current_index + window_size]]

                if pt_id in list(X.keys()):
                    X[pt_id].append(row)
                    y[pt_id].append(label)

                else:
                    X[pt_id] = [row]
                    y[pt_id] = [label]
        current_index += freq

    # with open('./data/normal/lstm.pkl','wb') as f:
    #     pickle.dump({'X': X, 'y': y}, f)
        
    # print('Dictionary saved to lstm.pkl')
    return X, y

if __name__ == '__main__':
    
    '''Data preprocessing step '''
    df = pd.read_csv('./data/normal/db_final.csv')
    df = df[['PtID','DeviceTm','Value','Scaled_Value','AgeAsOfEnrollDt','Weight','Height','HbA1c','Gender']]
    X,y = seq_data(df, 8)

    # with open('./data/normal/lstm.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # X = data['X']
    # y = data['y']

    pt_id = list(X.keys())
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(pt_id)):
        train_ids = [pt_id[i] for i in train_idx]
        
        val_size = int(len(train_ids) * 0.2)
        val_ids = train_ids[:val_size]
        train_ids = train_ids[val_size:]
        test_ids = [pt_id[i] for i in test_idx]

        X_train = [x for pid in train_ids for x in X[pid]]
        X_train = np.array(X_train)
        y_train = [label for pid in train_ids for label in y[pid]]
        y_train = np.array(y_train)

        X_val = [x for pid in val_ids for x in X[pid]]
        X_val = np.array(X_val)
        y_val = [label for pid in val_ids for label in y[pid]]
        y_val = np.array(y_val)

        '''Creating LSTM model '''
        lstm_model = Sequential()
        lstm_model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
        lstm_model.add(LSTM(64))
        lstm_model.add(Dense(8, activation="relu"))
        lstm_model.add(Dense(1, activation="linear"))


        cp = ModelCheckpoint(filepath='./data/normal/predictions/lstm/model', save_best_only=True)
        lstm_model.compile(loss='mae', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])
        history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10, callbacks=[cp])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/30min/loss.png')

        '''Performance on training data: '''
        train_predictions = lstm_model.predict(X_train).flatten()
        train_results = pd.DataFrame(
            data={"Train Predictions": train_predictions, 'Actuals': y_train})

        plt.figure(figsize=(14, 8))
        plt.plot(train_results['Train Predictions'][:100],
                label="Train Predictions", color="red")
        plt.plot(train_results['Actuals'][:100], label="Actuals", color="blue")
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/30min/train.png')

        '''Performance on validation data: '''
        val_predictions = lstm_model.predict(X_val).flatten()
        val_results = pd.DataFrame(
            data={"Train Predictions": val_predictions, 'Actuals': y_val})

        plt.figure(figsize=(14, 8))
        plt.plot(val_results['Train Predictions'][:100],
                label="Validation Predictions", color="red")
        plt.plot(val_results['Actuals'][:100], label="Actuals", color="blue")
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/30min/val.png')

        '''Performance on test data: '''

        for test_id in test_ids:
            X_test = np.array(X[test_id])
            y_test = np.array(y[test_id])
            test_predictions = lstm_model.predict(X_test).flatten()
            test_results = pd.DataFrame(
                data={"Test Predictions": test_predictions, 'Actuals': y_test})

            plt.figure(figsize=(14, 8))
            plt.plot(test_results['Test Predictions'][:100],
                    label="Test Predictions", color="red")
            plt.plot(test_results['Actuals'][:100], label="Actuals", color="blue")
            plt.legend()
            plt.savefig(f'./data/normal/predictions/lstm/30 min/test/test_ptid{int(test_id)}.png')
