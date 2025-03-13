import pandas as pd
import numpy as np


diagnode_data = pd.read_csv('./data/diagnode/df_final.csv')
islet_tp_data = pd.read_csv('./data/type_1/df_final.csv')
jaeb_data_healthy = pd.read_csv('./data/jaeb_healthy/df_final.csv')

'''
Overlapping features:
    id
    timestamp/DeviceTM
    Value
    Age/AgeAsOfEnrollDt
    Weight
    Height
    Sex
    hbA1C

'''

overlapping_features = ['id','timestamp', 'glucose mmol/l', 'sex', 'age', 'weight', 'height','hb_a1c']

#Prepping Diagnode data
agg_diagnode_data = diagnode_data[overlapping_features]
agg_diagnode_data.loc[:, 'id'] = 'diagnode_' + agg_diagnode_data['id']

#Prepping Islet transplant data
agg_islet_tp_data = islet_tp_data[overlapping_features]
agg_islet_tp_data['id'] = 'islet_' + agg_islet_tp_data['id'].astype('str')
 
#Prepping JAEB data
agg_jaeb_data = jaeb_data_healthy[overlapping_features]
agg_jaeb_data['id'] = 'jaeb_' + agg_jaeb_data['id'].astype('str')

print(f'Diagnode data features: {agg_diagnode_data.columns}')
print(f'Islet Transplant data features: {agg_islet_tp_data.columns}')
print(f'JAEB data features: {agg_jaeb_data.columns}')

agg_final = pd.concat([agg_diagnode_data, agg_islet_tp_data, agg_jaeb_data], ignore_index=True)
agg_final.to_csv('./data/aggregated_data.csv', index=False)
