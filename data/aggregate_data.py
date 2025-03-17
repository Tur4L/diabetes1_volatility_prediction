import pandas as pd
import numpy as np


diagnode_data = pd.read_csv('./data/diagnode/df_final.csv')
islet_tp_data = pd.read_csv('./data/itx/df_final.csv')
jaeb_healthy_data_healthy = pd.read_csv('./data/jaeb_healthy/df_final.csv')

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
agg_jaeb_healthy_data = jaeb_healthy_data_healthy[overlapping_features]
agg_jaeb_healthy_data['id'] = 'jaeb_healthy_' + agg_jaeb_healthy_data['id'].astype('str')

agg_final = pd.concat([agg_diagnode_data, agg_islet_tp_data, agg_jaeb_healthy_data], ignore_index=True)
agg_final.to_csv('./data/aggregated_data.csv', index=False)
