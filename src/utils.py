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


def preprocess_numerical(X, X_train, np_column_list):
    for column in np_column_list:
        mean = np.mean(X_train[:, :, column])
        std = np.std(X_train[:, :, column])
        X[:, :, column] = (X[:, :, column] - mean) / std
    return X


def remove_id_columns(X, columns_list=[0, 1]):
    """
    Remove study ID and encounter number columns
    Parameters:
    - X: numpy 
         with columns to be removed .
    - columns_list: list of ints 
         indices of cols to be removed
    Returns: 
        X: numpy 

    """
    X = np.delete(X, columns_list, 2)
    return X


def seq_data(df, window_size):
    """
    Create sequential numpy data from df for lstm to process
    Parameters:
    - df: dataframe 
         df of processed dataset .
    - window_size: int
         how far back to look
    Returns: 
        X: numpy
          of X input
        y: numpy
         of y input
        group_ids: list of str
         ids (study_id) of each patient with their corresponding BG in the y array
        average_y: numpy
         of the corresponding patient's average BG for that entire encounter

    """
    counts = df.groupby(['STUDY_ID', 'ENCOUNTER_NUM']).size()
    glucose_df = make_average_prediction(df)
    lstm_data = df.to_numpy()
    frequencies = counts.to_numpy()
    X = []
    y = []
    average_y = []
    group_ids = []
    current_index = 0
    for freq in frequencies:
        if freq > window_size:
            for i in range(freq-window_size):
                study_id, encounter_number = int(
                    lstm_data[current_index][0]), int(lstm_data[current_index][1])
                group_ids.append(str(
                    study_id))
                # glucose level in the next meal
                label = lstm_data[i+window_size+current_index][GLUCOSE_COL]
                # all data from window_size number of previous meals
                row = [a for a in lstm_data[i +
                                            current_index:i+window_size+current_index]]
                X.append(row)
                y.append(label)
                average_y.append(
                    glucose_df.loc[study_id, encounter_number])
        current_index += freq

    return np.array(X).astype(float), np.array(y).astype(float), group_ids, np.array(average_y)


def select_data(data_type, ids, group_ids, X, y):
    """
    Given a list of ids, return the corresponding entries in X and y
    Parameters:
    - data_type: str
         "test" for test data, "average" for average BG, "fold" for fold train and val, else for normal selection .
    - ids: list of str
         list of ids we want to select
    - group_ids: list of str
         full list of ids for X and y
    - X: numpy
         numpy of input X
    - y: numpy
         numpy of output y
    Returns: 
        X: numpy
          of X input after selection
        y: numpy
         of y input after selection
    """
    if data_type == "test":
        mask = np.isin(group_ids, ids)
        X_test, y_test = X[mask], y[mask]

        patients = {}
        patient_input = []
        patient_actual = []
        num_patient = 0
        prev_stud_id = ""
        prev_enc_id = ""

        for list_index, data_list in enumerate(X_test):
            stud_id = data_list[0][0]
            enc_id = data_list[0][1]

            if stud_id == prev_stud_id and enc_id == prev_enc_id:
                patient_input.append(data_list)
                patient_actual.append(y_test[list_index])
            else:
                if prev_stud_id != "" and prev_enc_id != "":
                    patients[num_patient] = (
                        remove_id_columns(patient_input), patient_actual)
                    num_patient += 1

                patient_input = [data_list]
                patient_actual = [y_test[list_index]]

            prev_stud_id = stud_id
            prev_enc_id = enc_id

        return patients, remove_id_columns(X_test), y_test

    elif data_type == 'average':
        mask = np.isin(group_ids, ids)
        return y[mask]
    elif data_type == 'fold':
        mask = np.isin(group_ids, ids)
        return X[mask], y[mask]
    else:
        mask = np.isin(group_ids, ids)
        return remove_id_columns(X[mask]), y[mask]


def KFold(k, train_ids):
    """
    Return k number of splits with k number of folds in each, in the form of ordered group ids
    Parameters:
    - k: int
        number of folds .
    - train_ids: list of ints
        IDs we are using to sort X and y train .
    Returns: 
    - train_test_ids: list of lists of ints
        splits with folds
    """
    split_ids = np.array_split(
        train_ids, k)

    train_test_ids = []

    for i in range(k):
        # Slicing the list
        # Includes lists from the start up to and including index i
        slice_up_to_i = split_ids[:i+1]
        # Includes lists from just after index i to the end
        slice_after_i = split_ids[i+1:]
        train_list = [item for sublist in slice_up_to_i +
                      slice_after_i for item in sublist]
        val_list = split_ids[i]
        train_test_ids.append([train_list, val_list])
    return train_test_ids


def plot(predictions, actuals, i, type, fig_name=None):
    """
    Plot predictions vs actuals
    Parameters:
    - predictions: numpy
        of predictions .
    - actuals: numpy
         of actuals
    - i: int
        fold #
    - type: str
         type of prediction
    - fig_name (opt): str
        name of figure
    Returns: 
        None
    """
    results = pd.DataFrame(
        data={type+"Predictions": predictions, 'Actuals': actuals})

    plt.figure(figsize=(14, 8))
    plt.plot(results[type+'Predictions'][:100],
             label=type+" Predictions", color="red")
    plt.plot(results['Actuals'][:100], label="Actuals", color="blue")
    plt.ylabel('Patient Meal BG (mmol/L)')
    plt.xlabel('Meals')
    plt.legend()
    if fig_name:
        plt.savefig(fig_name)
    else:
        plt.savefig(f'./rnn/models/lstm_model/{i}_{type}.png')
    plt.close()


def make_average_prediction(df):
    """
    create a dataframe of average predictions for patients over their entire encounter
    Parameters:
    - df: dataframe
         of input
    Returns: 
        None
    """
    mean_df = df.groupby(['STUDY_ID', 'ENCOUNTER_NUM'])[
        "GLUCOSE (mmol/L)"].mean().rename('Mean Glucose')
    return mean_df


def test_errors(preds, actuals, axis=None):
    """
    Find errors of predictions vs actuals.
    If provided array is 1D, it finds the errors of the test set and outputs a single number. 
    If provided array is 2D, it finds the errors of the tests set row-wise and outputs lists of errors. 
    Parameters
    ----------
    preds : 1D or 2D numpy array
        Example: [14,10,12] or [[14],[10],[12]]
        Numpy array of glucose predictions.
    actuals: 1D or 2D array
       Numpy array of glucose predictions.

    Returns
    -------
    if array is 1D, return numbers (float) for each error.
    if array is 2D, return list of floats for each error.

    """

    preds = tf.convert_to_tensor(preds, dtype=tf.float32)
    actuals = tf.convert_to_tensor(
        actuals, dtype=tf.float32)

    t_MAE = tf.keras.losses.MAE(preds, actuals).numpy()
    t_gMAE = gMAE(preds, actuals, axis=axis).numpy()
    t_RMSE = tf.sqrt(tf.reduce_mean(
        tf.square(tf.subtract(preds, actuals)), axis=axis)).numpy()
    t_MAPE = tf.keras.losses.mean_absolute_percentage_error(
        preds, actuals).numpy()
    t_gRMSE = gRMSE(preds, actuals, axis=axis).numpy()
    t_MARD = MARD(preds, actuals, axis=axis).numpy()
    t_gMARD = gMARD(preds, actuals, axis=axis).numpy()

    return t_MAE, t_gMAE, t_RMSE, t_gRMSE, t_MAPE, t_MARD, t_gMARD


def find_confidence_errors_w_intervals(preds, actuals):
    """
    Find errors and confidence intervals of predictions vs actuals.
    Parameters
    ----------
    preds : 1D or 2D numpy array
        Example: [14,10,12] or [[14],[10],[12]]
        Numpy array of glucose predictions.
    actuals: 1D or 2D array
       Numpy array of glucose predictions.

    Returns
    -------
    errors: list of lists
        return a list of lists of errors and intervals
        ex: {[1.4,0.04]..}

    """
    interval = 0.90

    # if 1D, convert to 2D
    if len(preds.shape) == 1:
        preds = preds.reshape(-1, 1)
    if len(actuals.shape) == 1:
        actuals = actuals.reshape(-1, 1)

    # provide 1D, return error numbers
    t_MAE, t_gMAE, t_RMSE, t_gRMSE, t_MAPE, t_MARD, t_gMARD = test_errors(
        preds.flatten(), actuals.flatten())

    # provide 2D,, return lists of errors
    i_MAE, i_gMAE, i_RMSE, i_gRMSE, i_MAPE, i_MARD, i_gMARD = test_errors(
        preds, actuals, axis=1)
    errors_list = [i_MAE, i_gMAE, i_RMSE, i_gRMSE,
                   i_MAPE, i_MARD, i_gMARD]
    errors = [[t_MAE], [t_gMAE], [t_RMSE], [t_gRMSE], [
        t_MAPE], [t_MARD], [t_gMARD]]

    i = 0
    # create 90% confidence interval
    for i, _ in enumerate(errors):
        i_error = errors_list[i]
        interval_list = st.norm.interval(alpha=interval,
                                         loc=np.mean(i_error),
                                         scale=st.sem(i_error))
        confidence = interval_list[1] - interval_list[0]
        errors[i].append(confidence)
    return errors

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
    df_filtered : filtered dataframe for TFT model

    """

    counts = df.groupby(['STUDY_ID', 'ENCOUNTER_NUM']).size()
    min_data_points = encoder_length + prediction_length
    valid = counts[counts >= min_data_points].index
    df_filtered = df.set_index(['STUDY_ID', 'ENCOUNTER_NUM']).loc[valid].reset_index()
    return df_filtered
