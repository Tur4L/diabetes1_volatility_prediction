import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np

''' Creating database'''
db = pd.read_csv('./data/db_window_random_10days.csv')
db['timestamp'] = pd.to_datetime(db['timestamp'])
db['glucose mmol/l pct_chng'] = db['glucose mmol/l'].pct_change() * 100
db['glucose mmol/l pct_chng'][0] = 0.0
db.set_index('timestamp', inplace=True)

patient_id = db['patient_id'][0]
window_id = db['window_id'][0]
beta2_score = db['BETA2_score'][0]

# train = db.iloc[:-6]
test = db.iloc[-365:]

''' Finding the order values'''
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100), dpi=100)
# plot_acf(db['glucose mmol/l'])
# plot_pacf(db['glucose mmol/l'], method='ywm')
# ax1.tick_params(axis='both', labelsize=12)
# ax2.tick_params(axis='both', labelsize=12)
# plt.show()

''' Creating GARCH model'''
rolling_predictions = []
test_size = len(test)
for i in range(test_size):
    train = db['glucose mmol/l pct_chng'][:-test_size-i]
    best_aic = np.inf
    best_pq = None
    best_model = None
    max_p = 4
    max_q = 4

    for p in range(1,max_p):
        for q in range(max_q):
            garch_model = arch_model(train, vol='GARCH', p=p, q=q)
            garch_fit = garch_model.fit(disp='off')
            
            if garch_fit.aic < best_aic:
                best_aic = garch_fit.aic
                best_pq = (p, q)
                best_model = garch_fit

    pred = best_model.forecast(horizon=6,reindex=False)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][-1]))

rolling_predictions = pd.Series(rolling_predictions,index=test.index)    

''' Plotting'''
print(f'Patient {patient_id}, Window: {window_id}')
print(f'Best P,Q : {best_pq}')
plt.figure(figsize=(10, 6))
plt.plot(test['glucose mmol/l pct_chng'], label='Glucose_pct_chng', color='blue')
plt.plot(rolling_predictions, label='Predicted Volatility', color='red')
plt.title(f'BETA2 Score: {beta2_score}')
plt.xlabel('Periods')
plt.ylabel('Glucose Percentage Change')
plt.legend()
plt.savefig(f'./data/predictions/garch/GARCH/p{patient_id}_b2{window_id}.png')