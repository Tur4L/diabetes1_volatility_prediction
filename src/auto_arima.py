import pandas as pd
import numpy as np
import pmdarima as pm
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

''' For single window '''
db = pd.read_csv('./data/db_window_random.csv')
db['timestamp'] = pd.to_datetime(db['timestamp'])
db.set_index('timestamp', inplace=True)
patient_id = db['patient_id'][0]
train = db.iloc[:int(0.8 * len(db))]
test = db.iloc[int(0.8 * len(db)):]

''' Cheking stationarity '''
# adf_test = adfuller(db['glucose mmol/l'])
# print(f"\nPatient {patient_id}: ")
# print(f'ADF-Statistics: {adf_test[0]}')
# print(f'p-value: {adf_test[1]}')
# print("Critical values: ")
# for key, value in adf_test[4].items():
#     print(f'   {key}: {value}')

model = pm.auto_arima(train['glucose mmol/l'], 
                      start_p=1, start_q=1,      # initial guesses for p and q
                      max_p=5, max_q=5,          # maximum p and q values to try
                      seasonal=True,            # no seasonality for CGM data
                      stepwise=True,             # uses a stepwise approach to find the best model
                      suppress_warnings=True,    # suppress warnings
                      trace=False)                # output the results of the search process

actuals = test['glucose mmol/l']
n_periods = len(actuals)  # Number of periods to forecast
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)

''' Metrics: '''
print(f"\nForecasts for patient {patient_id} \n")
print(f'AIC: {model.aic()}')
print(f'Sigma^2: {np.var(model.resid())}')
print(f'Log likelihood: {model.arima_res_.llf}')
print(f"Patient {patient_id} MAE: {mean_absolute_error(actuals,forecast)}")
print(f"Patient {patient_id} MAPE: {mean_absolute_percentage_error(actuals,forecast)}")
print(f"Patient {patient_id} MSE: {mean_squared_error(actuals,forecast)}")

''' Plotting: '''
# plt.plot(train['glucose mmol/l'], label='Observed')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.plot(forecast_index, test['glucose mmol/l'], label='Actual', color = 'blue')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()