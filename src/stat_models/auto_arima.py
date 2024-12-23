import pandas as pd
import numpy as np
import pmdarima as pm
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#-----------------------------------------------------------------------------------------------------
""" For single window """
# db = pd.read_csv('./data/db_window_random_10days.csv')
# db['timestamp'] = pd.to_datetime(db['timestamp'])
# db.set_index('timestamp', inplace=True)
# patient_id = db['patient_id'][0]
# window_id = db['window_id'][0]
# beta2_score = db['BETA2_score'][0]
# train_90mins = db.iloc[:-6]
# test_90mins = db.iloc[-6:]
# train_30mins = db.iloc[:-2]
# test_30mins = db.iloc[-2:]

# ''' Cheking stationarity '''
# adf_test = adfuller(db['glucose mmol/l'])
# print(f"\nPatient {patient_id}: ")
# print(f'ADF-Statistics: {adf_test[0]}')
# print(f'p-value: {adf_test[1]}')
# print("Critical values: ")
# for key, value in adf_test[4].items():
#     print(f'   {key}: {value}')

# '''90 minute model'''
# model_90mins = pm.auto_arima(train_90mins['glucose mmol/l'], 
#                       start_p=1, start_q=1,      # initial guesses for p and q
#                       max_p=10, max_q=10,          # maximum p and q values to try
#                       seasonal=True,            # no seasonality for CGM data
#                       stepwise=True,             # uses a stepwise approach to find the best model
#                       suppress_warnings=True,    # suppress warnings
#                       trace=False)                # output the results of the search process

# actuals_90mins = test_90mins['glucose mmol/l']
# n_periods = len(actuals_90mins)  # Number of periods to forecast
# forecast_90mins, conf_int_90mins = model_90mins.predict(n_periods=n_periods, return_conf_int=True)
# # forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)

# '''30 minute model '''
# model_30mins = pm.auto_arima(train_30mins['glucose mmol/l'], 
#                       start_p=1, start_q=1,      # initial guesses for p and q
#                       max_p=10, max_q=10,          # maximum p and q values to try
#                       seasonal=True,            # no seasonality for CGM data
#                       stepwise=True,             # uses a stepwise approach to find the best model
#                       suppress_warnings=True,    # suppress warnings
#                       trace=False)                # output the results of the search process

# actuals_30mins = test_30mins['glucose mmol/l']
# n_periods = len(actuals_30mins)  # Number of periods to forecast
# forecast_30mins, conf_int_30mins = model_30mins.predict(n_periods=n_periods, return_conf_int=True)
# # forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)


# ''' Metrics: '''
# print(f"\nForecasts for patient {patient_id} \n")
# print(f'AIC: {model_90mins.aic()}')
# print(f'Sigma^2: {np.var(model_90mins.resid())}')
# print(f'Log likelihood: {model_90mins.arima_res_.llf}')
# print(f"Patient {patient_id} MAE: {mean_absolute_error(actuals_90mins,forecast_90mins)}")
# print(f"Patient {patient_id} MAPE: {mean_absolute_percentage_error(actuals_90mins,forecast_90mins)}")
# mape_30mins = float(mean_absolute_percentage_error(actuals_30mins,forecast_30mins))
# mape_90mins = float(mean_absolute_percentage_error(actuals_90mins,forecast_90mins))
# print(f"Patinet {patient_id} MAPE 30mins/90mins: {mape_30mins/mape_90mins}")
# print(f"Patient {patient_id} MSE: {mean_squared_error(actuals_90mins,forecast_90mins)}")

# ''' Plotting: '''
# # plt.plot(train['glucose mmol/l'], label='Observed')0
# plt.plot(['00:15','00:30','00:45','01:00','01:15','01:30'], forecast_90mins, label='Forecast', color='red')
# plt.plot(['00:15','00:30','00:45','01:00','01:15','01:30'], actuals_90mins, label='Actual', color = 'blue')
# plt.fill_between(['00:15','00:30','00:45','01:00','01:15','01:30'], conf_int_90mins[:, 0], conf_int_90mins[:, 1], color='pink', alpha=0.3)
# plt.ylim(int(actuals_90mins.min() - 1), int(actuals_90mins.max() + 1))
# plt.title(f'Beta2 score: {beta2_score}')
# plt.legend()
# plt.savefig(f'./data/predictions/auto_arima/p{patient_id}_b2{window_id}.png')


# ''' Table of predictions/actuals '''
# results ={'Actuals' : actuals_90mins.reset_index(drop=True), 'Predictions' : forecast_90mins.reset_index(drop=True),
#           'Residuals' : actuals_90mins.reset_index(drop=True) - forecast_90mins.reset_index(drop=True) }

# db_results = pd.DataFrame(results)
# window_id = db['window_id'][0]
# db_results.to_csv(f'./data/predictions/auto_arima/p{patient_id}_b2{window_id}.csv')


#-----------------------------------------------------------------------------------------------------
# ''' For multiple windows '''
df = pd.read_csv('./data/normal/db_final.csv')
beta2_windows = df.groupby('PtID')

''' Looping over beta2 scores '''
for patient_id, window in beta2_windows:
    '''Creating the data'''
    # window['timestamp'] = pd.to_datetime(window['timestamp'])
    window.set_index('DeviceTm', inplace=True)
    train_90mins = window.iloc[:-6]
    test_90mins = window.iloc[-6:]
    train_30mins = window.iloc[:-2]
    test_30mins = window.iloc[-2:]

    '''90 minute model: '''

    model_90mins = pm.auto_arima(train_90mins['Value'], 
                      start_p=1, start_q=1,      # initial guesses for p and q
                      max_p=10, max_q=10,          # maximum p and q values to try
                      seasonal=True,            # no seasonality for CGM data
                      stepwise=True,             # uses a stepwise approach to find the best model
                      suppress_warnings=True,    # suppress warnings
                      trace=False)                # output the results of the search process

    actuals_90mins = test_90mins['Value']
    n_periods = len(actuals_90mins)  # Number of periods to forecast
    forecast_90mins, conf_int_90mins = model_90mins.predict(n_periods=n_periods, return_conf_int=True)
    # forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)

    '''30 minute model'''
    model_30mins = pm.auto_arima(train_30mins['Value'], 
                      start_p=1, start_q=1,      # initial guesses for p and q
                      max_p=10, max_q=10,          # maximum p and q values to try
                      seasonal=True,            # no seasonality for CGM data
                      stepwise=True,             # uses a stepwise approach to find the best model
                      suppress_warnings=True,    # suppress warnings
                      trace=False)                # output the results of the search process

    actuals_30mins = test_30mins['Value']
    n_periods = len(actuals_30mins)  # Number of periods to forecast
    forecast_30mins, conf_int_30mins = model_30mins.predict(n_periods=n_periods, return_conf_int=True)
    # forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)

    ''' Metrics: '''
    print(f"\nForecasts for patient {patient_id} \n")
    print(f'AIC: {model_90mins.aic()}')
    print(f'Sigma^2: {np.var(model_90mins.resid())}')
    print(f'Log likelihood: {model_90mins.arima_res_.llf}')
    print(f"Patient {patient_id} MAE: {mean_absolute_error(actuals_90mins,forecast_90mins)}")
    print(f"Patient {patient_id} MAPE: {mean_absolute_percentage_error(actuals_90mins,forecast_90mins)}")
    mape_30mins = float(mean_absolute_percentage_error(actuals_30mins,forecast_30mins))
    mape_90mins = float(mean_absolute_percentage_error(actuals_90mins,forecast_90mins))
    print(f"Patinet {patient_id} MAPE 30mins/90mins: {mape_30mins/mape_90mins}")
    print(f"Patient {patient_id} MSE: {mean_squared_error(actuals_90mins,forecast_90mins)}")

    ''' Plotting: '''
    plt.figure(figsize=(10,6))
    plt.plot(test_90mins.index, forecast_90mins, label='Forecast', color='red')
    plt.plot(test_90mins.index, actuals_90mins, label='Actual', color = 'blue')
    plt.fill_between(test_90mins.index, conf_int_90mins[:, 0], conf_int_90mins[:, 1], color='pink', alpha=0.3)
    plt.ylim(int(actuals_90mins.min() - 1), int(actuals_90mins.max() + 1))
    plt.legend()
    plt.savefig(f'./data/normal/predictions/auto_arima/p{patient_id}.png')

    ''' Saving Results: '''
    results ={'Actuals' : actuals_90mins.reset_index(drop=True), 'Predictions' : forecast_90mins.reset_index(drop=True),
            'Residuals' : actuals_90mins.reset_index(drop=True) - forecast_90mins.reset_index(drop=True) }
    db_results = pd.DataFrame(results)
    db_results.to_csv(f'./data/normal/predictions/auto_arima/p{patient_id}.csv')


"""Day/Night analysis """
# db = pd.read_csv('./data/db_window_random_10days.csv')
# patient_id = db['patient_id'][0]
# window_id = db['window_id'][0]
# beta2_score = db['BETA2_score'][0]

# db['timestamp'] = pd.to_datetime(db['timestamp'])
# cutoff_period = db['timestamp'].max() - pd.Timedelta(days=1)
# test_data = db[db['timestamp'] >= cutoff_period]
# train_data = db[db['timestamp'] < cutoff_period]
# test_data.set_index('timestamp', inplace=True)
# train_data.set_index('timestamp', inplace=True)

# len_testdata = len(test_data['glucose mmol/l'])
# index = 0
# forecast_horizon = 6
# while index + forecast_horizon <= len_testdata:
#     model = pm.auto_arima(train_data['glucose mmol/l'], 
#                         start_p=1, start_q=1,
#                         max_p=10, max_q=10,
#                         seasonal=True,
#                         stepwise=True,
#                         suppress_warnings=True,
#                         trace=False)

    
#     forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
#     actuals = test_data['glucose mmol/l'][index:index + forecast_horizon]
#     actuals_timestamps = test_data.index[index:index + forecast_horizon]
#     first_timestamp = actuals_timestamps[0]
#     time = first_timestamp.strftime('%H:%M')
#     period = 'Daytime' if 8 <= first_timestamp.hour < 20 else 'Nighttime'

#     # plt.plot(train['glucose mmol/l'], label='Observed')
#     plt.figure(figsize=(10,6))
#     plt.plot(actuals_timestamps.strftime('%H:%M'), forecast, label='Forecast', color='red')
#     plt.plot(actuals_timestamps.strftime('%H:%M'), actuals, label='Actual', color = 'blue')
#     plt.fill_between(actuals_timestamps.strftime('%H:%M'), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
#     plt.ylim(0, 20)
#     plt.title(f'Beta2 score: {beta2_score}')
#     plt.legend()
#     plt.savefig(f'./data/predictions/auto_arima/{period}/p{patient_id}_b2{window_id}_{time}.png')

#     train_data = pd.concat([train_data,test_data.iloc[index:index+forecast_horizon]])
#     index += 6