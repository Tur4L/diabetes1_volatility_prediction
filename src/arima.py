import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

'''For db with group of patients'''
# db = pd.read_csv('./data/normalized_JS_JT_KB.csv')
# patients_group = db.groupby('patient_id')
# arima_order = [(26, 0 , 7), (15, 0 , 4), (20, 0, 5), (11, 0, 5)]
# for patient_id, patient_data in patients_group:
#     patient_data.set_index('timestamp', inplace=True)
#     ''' Performing ADF test to check stationarity. Current p-values: [2.0475783368680417e-28, 0 , 0, 0] '''
#     # adf_test = adfuller(patient_data['glucose mmol/l'])
#     # print(f"\nPatient {patient_id}: ")
#     # print(f'ADF-Statistics: {adf_test[0]}')
#     # print(f'p-value: {adf_test[1]}')
#     # print("Critical values: ")
#     # for key, value in adf_test[4].items():
#     #     print(f'   {key}: {value}')

#     ''' Plotting ACF and PACF for the ARIMA order'''
#     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100), dpi=100)
#     # plot_acf(patient_data['glucose mmol/l'])
#     # plot_pacf(patient_data['glucose mmol/l'], method='ywm')
#     # ax1.tick_params(axis='both', labelsize=12)
#     # ax2.tick_params(axis='both', labelsize=12)
#     # plt.show()

#     ''' Orders determined by ACF and PACF: '''
#     # Patient_id = 1: q = 26, p = 7
#     # Patient_id = 2: q = 15, p = 4
#     # Patient_id = 3: q = 26, p = 5
#     # Patient_id = 4: q = 12, p = 6

#     ''' Creating the ARIMA model for each patient: '''
#     train = patient_data.iloc[:int(0.80 * len(patient_data))]
#     test = patient_data.iloc[int(0.80 * len(patient_data)):]
#     q, d, p = arima_order[patient_id-1]
#     model = ARIMA(train['glucose mmol/l'], order=(p, d, q))
#     model_fit = model.fit()

#     forecast = model_fit.forecast(len(test))
#     forecast.index = test.index
#     forecast_db = pd.DataFrame()
#     forecast_db['actuals'] = test['glucose mmol/l']
#     forecast_db['predictions'] = forecast
#     forecast_db['residuals'] = forecast_db['actuals'] - forecast_db['predictions']
#     forecast_db.to_csv(f'./data/predictions/forecast_p{patient_id}.csv', index=False)

#     print(f"\nForecasts for patient {patient_id} \n")
#     print(f"Patient {patient_id} MAE: {mean_absolute_error(forecast_db['actuals'],forecast_db['predictions'])}")
#     print(f"Patient {patient_id} MAPE: {mean_absolute_percentage_error(forecast_db['actuals'],forecast_db['predictions'])}")
#     print(f"Patient {patient_id} MSE: {mean_squared_error(forecast_db['actuals'],forecast_db['predictions'])}")

''' For db with single window '''
db = pd.read_csv('./data/db_window_random.csv')
db['timestamp'] = pd.to_datetime(db['timestamp'])
db.set_index('timestamp', inplace=True)
patient_id = db['patient_id'][0]


''' Performing ADF test to check stationarity. Current p-values: 0 '''
# adf_test = adfuller(db['glucose mmol/l'])
# print(f"\nPatient {patient_id}: ")
# print(f'ADF-Statistics: {adf_test[0]}')
# print(f'p-value: {adf_test[1]}')
# print("Critical values: ")
# for key, value in adf_test[4].items():
#     print(f'   {key}: {value}')

''' Plotting ACF and PACF for the ARIMA order'''
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100), dpi=100)
# plot_acf(db['glucose mmol/l'])
# plot_pacf(db['glucose mmol/l'], method='ywm')
# ax1.tick_params(axis='both', labelsize=12)
# ax2.tick_params(axis='both', labelsize=12)
# plt.show()

''' Orders determined by ACF and PACF: '''
# Patient_id = 4: q = 6, p = 3

''' Creating ARIMA model'''
train = db.iloc[:int(0.80 * len(db))]
test = db.iloc[int(0.80 * len(db)):]
model = ARIMA(train['glucose mmol/l'], order=(3, 0, 6))
model_fit = model.fit()

forecast = model_fit.forecast(len(test))
forecast_db = pd.DataFrame()

actuals = test['glucose mmol/l']
n_periods = len(actuals) 
forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)


''' Metrics: '''
print(f"\nForecasts for patient {patient_id} \n")
print(f'AIC: {model_fit.aic}')
print(f'Sigma^2: {np.var(model_fit.resid)}')
print(f'Log likelihood: {model_fit.llf}')
print(f"Patient {patient_id} MAE: {mean_absolute_error(actuals,forecast)}")
print(f"Patient {patient_id} MAPE: {mean_absolute_percentage_error(actuals,forecast)}")
print(f"Patient {patient_id} MSE: {mean_squared_error(actuals,forecast)}")

''' Plotting: '''
# plt.plot(train['glucose mmol/l'], label='Observed')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.plot(forecast_index, actuals, label='Actual', color = 'blue')
# plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()