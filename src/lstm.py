import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

WINDOW_SIZE = 8

def seq_data(df, window_size):
    """
    Create sequential numpy data from df for lstm to process
    Parameters:
    - df: dataframe
        df of windowed patient data
    - window_size: int
        how far back to look

    Returns:
    - X: 
        Numpy of X input
    - y:
        Numpy of y input

    """

    counts = df.groupby(['window_id']).size()
    lstm_data = df.to_numpy()
    frequencies = counts.to_numpy()
    X = []
    y = []
    group_ids = []
    current_index = 0

    for freq in frequencies:
        if freq > window_size:
            for i in range(freq-window_size):
                window_id = lstm_data[current_index][-1]
                group_ids.append(window_id)
                label = lstm_data[i+window_size+current_index][1]
                row = [a for a in lstm_data[i + current_index: i + current_index + window_size]]
                X.append(row)
                y.append(label)
        current_index += freq
'''Data preprocessing step '''
db = pd.read_csv('./data/db_windowed_all.csv')

'''Creating LSTM model '''
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
'''Results: '''

'''Plotting: '''