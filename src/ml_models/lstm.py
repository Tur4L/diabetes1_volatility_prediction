import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import *


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

WINDOW_SIZE = 8
N_SPLIT = 5


if __name__ == '__main__':
    
    '''Data preprocessing step '''
    df = pd.read_csv('./data/normal/db_final.csv')
    df = df[['PtID','DeviceTm','Value','Scaled_Value','AgeAsOfEnrollDt','Weight','Height','HbA1c','Gender']]
    X,y = lstm_data(df, WINDOW_SIZE)

    # with open('./data/normal/lstm.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # X = data['X']
    # y = data['y']

    pt_id = list(X.keys())
    kf = KFold(n_splits=N_SPLIT, shuffle=True)

    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae_score = 0
    mape_score = 0
    rmse_score = 0
    grmse_score = 0
    mard_score= 0
    gmard_score = 0

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


        cp = ModelCheckpoint(filepath='./data/normal/predictions/lstm/model.keras', save_best_only=True)
        lstm_model.compile(loss='mae', optimizer=Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.RootMeanSquaredError(), gRMSE, tf.keras.metrics.MeanAbsolutePercentageError(), MARD, gMARD])
        history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10, callbacks=[cp])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/age_all/30min/loss.png')

        '''Performance on training data: '''
        lstm_model = load_model("./data/normal/predictions/lstm/model/",
                        custom_objects={'gRMSE': gRMSE, 'MARD': MARD, 'gMARD': gMARD, 'gMAE': gMAE})
        train_predictions = lstm_model.predict(X_train).flatten()
        train_results = pd.DataFrame(
            data={"Train Predictions": train_predictions, 'Actuals': y_train})

        plt.figure(figsize=(14, 8))
        plt.plot(train_results['Train Predictions'][:100],
                label="Train Predictions", color="red")
        plt.plot(train_results['Actuals'][:100], label="Actuals", color="blue")
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/age_all/30min/train.png')

        '''Performance on validation data: '''
        val_predictions = lstm_model.predict(X_val).flatten()
        val_results = pd.DataFrame(
            data={"Train Predictions": val_predictions, 'Actuals': y_val})

        plt.figure(figsize=(14, 8))
        plt.plot(val_results['Train Predictions'][:100],
                label="Validation Predictions", color="red")
        plt.plot(val_results['Actuals'][:100], label="Actuals", color="blue")
        plt.legend()
        plt.savefig('./data/normal/predictions/lstm/age_all/30min/val.png')

        '''Performance on test data: '''

        for test_id in test_ids:
            X_test = np.array(X[test_id])
            y_test = np.array(y[test_id])
            test_predictions = lstm_model.predict(X_test).flatten()

            mae_score += tf.keras.metrics.MAE(y_test, test_predictions)
            mape_score += tf.keras.metrics.MAPE(y_test, test_predictions)
            rmse.update_state(y_true=y_test, y_pred=test_predictions)
            rmse_score += rmse.result().numpy()


            test_results = pd.DataFrame(
                data={"Test Predictions": test_predictions, 'Actuals': y_test})

            plt.figure(figsize=(14, 8))
            plt.plot(test_results['Test Predictions'][:100],
                    label="Test Predictions", color="red")
            plt.plot(test_results['Actuals'][:100], label="Actuals", color="blue")
            plt.legend()
            plt.savefig(f'./data/normal/predictions/lstm/age_all/30min/test/test_ptid{int(test_id)}.png')

    print('\nLSTM model statistics:')
    print(f'\tMAE score: {mae_score/(len(test_ids) * N_SPLIT)}')
    print(f'\tMAPE score: {mape_score/(len(test_ids) * N_SPLIT)}')
    print(f'\tRMSE score: {rmse_score/(len(test_ids) * N_SPLIT)}')

