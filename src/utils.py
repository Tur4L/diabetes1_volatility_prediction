import numpy as np
import tensorflow as tf
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

GLUCOSE_COL = 3
np.set_printoptions(threshold=np.inf)


def test_shape(np_matrix_1, np_matrix_2=[]):
    print()
    print("-----------------------------")
    print("Checking shapes of data")
    print(f"X: {np_matrix_1.shape}")
    if np_matrix_2 != []:
        print(f"y: {np_matrix_2.shape}")
    print("-----------------------------")
    print()


def clarke_error_zone_detailed(y_true, y_pred):
    """
    Calculate Clarke Error Grid zones for y_predictions.
    Zones range from 0 (Zone A) to 8 (Zone E - upper left and right lower),
    with higher numbers indicating worse accuracy according to the Clarke Error Grid.

    Based on paper: https://care.diabetesjournals.org/content/10/5/622

    Values in zones A and B are clinically acceptable, whereas values in zones C, D, and E are potentially dangerous and therefore are clinically significant errors

    Parameters:
    - y_true: tensor of floats 
         of all test targets.
    - y_pred: tensor of floats 
         of all test predictions
    return:
    - int
        of penalty/zone
    """
    y_true = 18 * y_true    # convert to mg/dl
    y_pred = 18 * y_pred    # convert to mg/dl

    # Initialize zones tensor, defaulting everything to Zone B - lower as a starting point
    zones = tf.ones_like(y_true, dtype=tf.float32)

    # Zone A
    zone_a = tf.logical_or(
        tf.logical_and(y_true < 70, y_pred < 70),
        tf.abs(y_true - y_pred) < 0.2 * y_true
    )
    zones = tf.where(zone_a, tf.constant(0.0, dtype=tf.float32), zones)

    # Zone E - left upper
    zone_e_left_upper = tf.logical_and(y_true <= 70, y_pred >= 180)
    zones = tf.where(zone_e_left_upper, tf.constant(
        8.0, dtype=tf.float32), zones)

    # Zone E - right lower
    zone_e_right_lower = tf.logical_and(y_true >= 180, y_pred <= 70)
    zones = tf.where(zone_e_right_lower, tf.constant(
        7.0, dtype=tf.float32), zones)

    # Zone D - right
    zone_d_right = tf.logical_and(tf.logical_and(
        y_true >= 240, y_pred >= 70), y_pred <= 180)
    zones = tf.where(zone_d_right, tf.constant(6.0, dtype=tf.float32), zones)

    # Zone D - left
    zone_d_left = tf.logical_and(tf.logical_and(
        y_true <= 70, y_pred >= 70), y_pred <= 180)
    zones = tf.where(zone_d_left, tf.constant(5.0, dtype=tf.float32), zones)

    # Zone C - upper
    zone_c_upper = tf.logical_and(tf.logical_and(
        y_true >= 70, y_true <= 290), y_pred >= y_true + 110)
    zones = tf.where(zone_c_upper, tf.constant(4.0, dtype=tf.float32), zones)

    # Zone C - lower
    zone_c_lower = tf.logical_and(tf.logical_and(
        y_true >= 130, y_true <= 180), y_pred <= (7/5) * y_true - 182)
    zones = tf.where(zone_c_lower, tf.constant(3.0, dtype=tf.float32), zones)

    # Zone B - upper
    zone_b_upper = tf.less(y_true, y_pred)
    zones = tf.where(zone_b_upper, tf.constant(2.0, dtype=tf.float32), zones)

    return zones


def clark_error_perc(t_test_true, y_test_pred):
    """
    Return clarke error percentage

    Parameters:
    - t_test_true: numpy array 
         of all test targets.
    - t_test_pred: numpy array
         of all test predictions
    Returns: float
        percentage between 0 and 1

    """
    clinic_sig_sum = 0
    t_test_true = tf.convert_to_tensor(t_test_true, dtype=tf.float32)
    y_test_pred = tf.convert_to_tensor(y_test_pred, dtype=tf.float32)
    zones = clarke_error_zone_detailed(t_test_true, y_test_pred)
    zone_bool = tf.logical_or(tf.logical_or(
        tf.equal(zones, 0.0), tf.equal(zones, 1.0)), tf.equal(zones, 2.0))

    # Convert boolean mask to float (1.0 for True, 0.0 for False)
    zone_float = tf.cast(zone_bool, tf.float32)

    # Sum up all 1's (True values) to get the count of clinically significant predictions
    clinic_sig_sum = tf.reduce_sum(zone_float)

    # Divide by the total number of predictions to get the fraction
    clinic_sig_fraction = clinic_sig_sum / tf.cast(tf.size(zones), tf.float32)

    return clinic_sig_fraction


def gMAE(y_true, y_pred, axis=None):
    """
    Return glucose specific mean absolute error
    formula: mean(penalty(true,pred) * |true-pred|)

    Parameters:
    - y_true: tensor of floats 
         of all true BG.
    - t_test_pred: tensor 
         of BG predictions
    Returns: float
        glucose-mae

    """
    penalty = clarke_error_zone_detailed(
        y_true, y_pred)
    # e = pred−truth
    e = y_pred - y_true
    product = tf.multiply(penalty, tf.abs(e))
    gmae = tf.reduce_mean(product, axis=axis)
    return (gmae)


def gRMSE(y_true, y_pred, axis=None):
    """
    Return glucose specific root mean square error
    formula: sqrt(mean(penalty(true,pred) * (true-pred)^2))

    Parameters:
    - y_true: tensor of floats 
         of all true BG.
    - t_test_pred: tensor 
         of BG predictions
    Returns: float
        glucose-gRMSE

    """
    penalty = clarke_error_zone_detailed(
        y_true, y_pred)
    square = tf.square(y_true - y_pred)
    product = tf.multiply(penalty, square)
    mean = tf.reduce_mean(product, axis=axis)
    gRMSE = tf.sqrt(mean)
    return gRMSE


def MARD(y_true, y_pred, axis=None):
    """
    Return mean absolute relative difference
    formula: mean(|true-pred|/true)

    Parameters:
    - y_true: tensor of floats 
         of all true BG.
    - t_test_pred: tensor 
         of BG predictions
    Returns: float
        MARD

    """
    # e = pred−truth
    e = y_pred - y_true
    div = tf.divide(e, y_true)
    mard = tf.reduce_mean(div, axis=axis)
    return mard


def gMARD(y_true, y_pred, axis=None):
    """
    Return glucose-specific mean absolute relative difference
    formula: mean(penalty(true,pred) * |true-pred|/true)

    Parameters:
    - y_true: tensor of floats 
         of all true BG.
    - t_test_pred: tensor 
         of BG predictions
    Returns: float
        gMARD

    """
    penalty = clarke_error_zone_detailed(
        y_true, y_pred)
    # e = pred−truth
    e = y_pred - y_true
    div = tf.divide(e, y_true)
    product = tf.multiply(div, penalty)
    gmard = tf.reduce_mean(product, axis=axis)
    return gmard

def lstm_data(df, window_size):
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
        if freq//6 > window_size:
            for i in range(0, freq//6-window_size):
                pt_id = input_data[current_index][0]
                
                ''' 5-minute intervals '''
                # label = label_data[i+window_size+current_index+5]
                # row = [a for a in input_data[i + current_index: i + current_index + window_size]]

                ''' 30-minute intervals '''

                row = [input_data[current_index + i *6 + j * 6] for j in range(window_size)]
                label_index = current_index + (i + window_size) * 6
                label = label_data[label_index]

                ''' 90-minute intervals '''
                # row = [input_data[current_index + i *18 + j * 18] for j in range(window_size)]
                # label_index = current_index + (i + window_size) * 18
                # label = label_data[label_index]


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

def transformer_data(df, encoder_length, prediction_length):
    """
    Filters original dataframe so that no input has less than 
    encoder_length + prediction_length rows
    ----------
    df : Original dataframe
    encoder_length : encoder length of the model
    prediction_length : prediction length of the model

    Returns
    -------
    tft_data : filtered dataframe for TFT model

    """
    tft_ptid_list = []
    grouped_df = df.groupby(['PtID'])

    for ptID, group_data in grouped_df:
        if group_data.size >= encoder_length + prediction_length:
            tft_ptid_list.append(ptID)

    df['time_idx'] = df.groupby('PtID').cumcount()
    tft_df = df[df['PtID'].isin(tft_ptid_list)]
    tft_df = tft_df.drop(['DeviceTm','T1DBioFamily','T1DBioFamParent','T1DBioFamSibling','T1DBioFamChild','T1DBioFamUnk'], axis=1)
    return tft_df
