import pandas as pd
import numpy as np

cloud_data = pd.read_csv('./data/cloud/df_final.csv')
clvr_data = pd.read_csv('./data/clvr/df_final.csv')
defend_data = pd.read_csv('./data/defend/df_final.csv')
diagnode_data = pd.read_csv('./data/diagnode/df_final.csv')
gskalb_data = pd.read_csv('./data/gskalb/df_final.csv')
islet_tp_data = pd.read_csv('./data/itx/df_final.csv')
jaeb_t1d_data = pd.read_csv('./data/jaeb_t1d/df_final.csv')

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
overlapping_features = ['id','timestamp', 'time_bins', 'glucose mmol/l', 'sex', 'age', 'weight', 'height', 'cpep_0_min',
                        'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'hb_a1c_local']

#Selecting aggregation features
agg_cloud_data = cloud_data[overlapping_features]

agg_clvr_data = clvr_data[overlapping_features]
agg_clvr_data['id'] = 'clvr_' + agg_clvr_data['id'].astype('str')

agg_defend_data = defend_data[overlapping_features]

agg_diagnode_data = diagnode_data[overlapping_features]
agg_diagnode_data.loc[:,'id'] = 'diagnode_' + agg_diagnode_data['id']

agg_gskalb_data = gskalb_data[overlapping_features]

agg_islet_tp_data = islet_tp_data[overlapping_features]
agg_islet_tp_data.loc[:,'id']  = 'islet_' + agg_islet_tp_data['id'].astype('str')
 
agg_jaeb_t1d_data = jaeb_t1d_data[overlapping_features]
agg_jaeb_t1d_data['id']  = 'jaeb_t1d_' + agg_jaeb_t1d_data['id'].astype('str')


agg_final = pd.concat([agg_cloud_data, agg_clvr_data, agg_defend_data, agg_diagnode_data, agg_gskalb_data, agg_islet_tp_data, agg_jaeb_t1d_data], ignore_index=True)
agg_final.to_csv('./data/aggregated_data.csv', index=False)
