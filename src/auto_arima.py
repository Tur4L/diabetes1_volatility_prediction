import pandas as pd
import numpy as np
import pmdarima as pm
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

''' For single window '''
db = pd.read_csv('./data/db_window_random_10days.csv')
db['timestamp'] = pd.to_datetime(db['timestamp'])
db.set_index('timestamp', inplace=True)
patient_id = db['patient_id'][0]
train_90mins = db.iloc[:-6]
test_90mins = db.iloc[-6:]
train_30mins = db.iloc[:-2]
test_30mins = db.iloc[-2:]

''' Cheking stationarity '''
adf_test = adfuller(db['glucose mmol/l'])
print(f"\nPatient {patient_id}: ")
print(f'ADF-Statistics: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
print("Critical values: ")
for key, value in adf_test[4].items():
    print(f'   {key}: {value}')

'''90 minute model'''
model_90mins = pm.auto_arima(train_90mins['glucose mmol/l'], 
                      start_p=1, start_q=1,      # initial guesses for p and q
                      max_p=10, max_q=10,          # maximum p and q values to try
                      seasonal=True,            # no seasonality for CGM data
                      stepwise=True,             # uses a stepwise approach to find the best model
                      suppress_warnings=True,    # suppress warnings
                      trace=False)                # output the results of the search process

actuals_90mins = test_90mins['glucose mmol/l']
n_periods = len(actuals_90mins)  # Number of periods to forecast
forecast_90mins, conf_int_90mins = model_90mins.predict(n_periods=n_periods, return_conf_int=True)
# forecast_index = pd.date_range(start=train['glucose mmol/l'].index[-1] + pd.Timedelta(days=1), periods=n_periods)

'''30 minute model '''
model_30mins = pm.auto_arima(train_30mins['glucose mmol/l'], 
                      start_p=1, start_q=1,      # initial guesses for p and q
                      max_p=10, max_q=10,          # maximum p and q values to try
                      seasonal=True,            # no seasonality for CGM data
                      stepwise=True,             # uses a stepwise approach to find the best model
                      suppress_warnings=True,    # suppress warnings
                      trace=False)                # output the results of the search process

actuals_30mins = test_30mins['glucose mmol/l']
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
# plt.plot(train['glucose mmol/l'], label='Observed')0
plt.plot(test_90mins.index.strftime('%H:%M'), forecast_90mins, label='Forecast', color='red')
plt.plot(test_90mins.index.strftime('%H:%M'), actuals_90mins, label='Actual', color = 'blue')
plt.fill_between(test_90mins.index.strftime('%H:%M'), conf_int_90mins[:, 0], conf_int_90mins[:, 1], color='pink', alpha=0.3)
plt.ylim(int(actuals_90mins.min() - 2), int(actuals_90mins.max()+2))
plt.legend()
plt.show()

''' Table of predictions/actuals '''
results ={'Actuals' : actuals_90mins.reset_index(drop=True), 'Predictions' : forecast_90mins.reset_index(drop=True),
          'Residuals' : actuals_90mins.reset_index(drop=True) - forecast_90mins.reset_index(drop=True) }

db_results = pd.DataFrame(results)
window_id = db['window_id'][0]
# db_results.to_csv(f'./data/predictions/auto_arima/results_p{patient_id}_b2{window_id}.csv')