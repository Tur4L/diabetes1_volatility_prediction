import pandas as pd
import numpy as np
import GPy

normalized_js_db = pd.read_csv("./data/normalized/normalized_JS.csv")
normalized_jl_db = pd.read_csv("./data/normalized/normalized_JL.csv")

X_train = normalized_js_db["timestamp"].values.reshape(-1,1)
Y_train = normalized_js_db["glucose mmol/l"].values.reshape(-1,1)
kernel = GPy.kern.RBF(input_dim=1)

print("Training model")
gp_model = GPy.models.GPRegression(X_train,Y_train,kernel)

print("Optimizing model")
gp_model.optimize(messages=True)

print("Testing model")
X_pred = normalized_jl_db["timestamp"].values.reshape(-1,1)
Y_pred, Y_pred_var = gp_model.predict(X_pred)
normalized_jl_db['predicted glucose mmol/l'] = Y_pred.flatten()
normalized_jl_db['prediction_std'] = np.sqrt(Y_pred_var).flatten()
normalized_jl_db.to_csv('./data/prediction/gp_prediction_JL.csv')


#32 GB RAM not suitable for data. Will try Sparse GP