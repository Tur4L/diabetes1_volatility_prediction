import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, kruskal, spearmanr, linregress, t, gmean, f_oneway
from itertools import combinations
import seaborn as sns
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import importlib
import math
from pathlib import Path
import os
import re
from itx_package import ITXData
from hupa_ucm_package import HUPA_UCM_Data
from data_analysis import DataAnalysis

data_analysis = DataAnalysis()
def _to_pandas(df_like):
    return df_like.compute() if hasattr(df_like, 'compute') else df_like.copy()

def get_cpep_auc(row):
    times = [0,15,30,60,90,120]
    cpep_values = [row[f'cpep_{t}_min'] for t in times]        
    auc = 0
    for i in range(1, len(times)):
        # width = difference in time
        delta_t = times[i] - times[i-1]
        # height = average of consecutive cpep values
        auc += (cpep_values[i] + cpep_values[i-1]) / 2 * delta_t

    duration = times[-1] - times[0]
    return auc / duration

df_cloud = data_analysis.get_dataset('cloud')
df_clvr = data_analysis.get_dataset('clvr')
df_defend = data_analysis.get_dataset('defend')
df_diagnode = data_analysis.get_dataset('diagnode')
df_gskalb = data_analysis.get_dataset('gskalb')
# df_itx = data_analysis.get_dataset('itx')
df_jaeb_healthy = data_analysis.get_dataset('jaeb_healthy')
df_jaeb_t1d = data_analysis.get_dataset('jaeb_t1d')

def testing_cloud_raw():
    '''AUC for C-Peptide Level'''
    treatment_arm = pd.read_csv('./data/studies/cloud/original/PtRoster.txt', sep='|')
    treatment_arm = treatment_arm[['PtID', 'TrtGroup']]
    treatment_arm.rename(columns={'TrtGroup':'treatment_arm'}, inplace=True)

    df_cpep = pd.read_csv('./data/studies/cloud/original/CloudLabCPeptide.txt', sep='|')
    df_cpep = df_cpep[df_cpep['VisitCollected'].isin(['Baseline', '6 Months', '12 Months', '24 Months'])]
    df_cpep[['CPeptide10Min', 'CPeptide0Min', 'CPeptide15Min', 'CPeptide30Min', 'CPeptide60Min', 'CPeptide90Min', 'CPeptide120Min']] = (
        df_cpep[['CPeptide10Min', 'CPeptide0Min', 'CPeptide15Min', 'CPeptide30Min', 'CPeptide60Min', 'CPeptide90Min', 'CPeptide120Min']]/1000)
    df_cpep.rename(columns={'CPeptide0Min': 'cpep_0_min', 'CPeptide10Min': 'cpep_pre10_min', 'CPeptide15Min': 'cpep_15_min', 'CPeptide30Min': 'cpep_30_min',
                            'CPeptide60Min': 'cpep_60_min', 'CPeptide90Min': 'cpep_90_min', 'CPeptide120Min': 'cpep_120_min'}, inplace=True)
    df_cpep['cpep_auc'] = df_cpep.apply(get_cpep_auc, axis=1)
    df_cpep = df_cpep.merge(treatment_arm, on='PtID')
    df_cpep['VisitNum'] = (
        df_cpep['VisitCollected']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    df_cpep = df_cpep.sort_values('VisitNum').reset_index(drop=True)

    visit_order = (
        df_cpep[['VisitCollected', 'VisitNum']]
        .drop_duplicates()
        .sort_values('VisitNum')['VisitCollected']
        .tolist()
    )

    gmean_auc = (
        df_cpep
        .groupby(['treatment_arm', 'VisitCollected'], sort=False)['cpep_auc']
        .apply(lambda x: gmean(x[x > 0]))  # exclude zeros to avoid math domain error
        .reset_index(name='gmean_cpep_auc')
    )
    iqr_auc = (
        df_cpep[df_cpep['cpep_auc'] > 0]
        .groupby(['treatment_arm', 'VisitCollected'], sort=False)['cpep_auc']
        .quantile([0.25, 0.75])
        .unstack(level=-1)
        .rename(columns={0.25: 'q25', 0.75: 'q75'})
        .reset_index()
    )
    gmean_auc = gmean_auc.merge(iqr_auc, on=['treatment_arm', 'VisitCollected'], how='left')

    treatment_0 = gmean_auc[gmean_auc['treatment_arm'] == 'MDI'].set_index('VisitCollected').reindex(visit_order)
    treatment_1 = gmean_auc[gmean_auc['treatment_arm'] == 'CL'].set_index('VisitCollected').reindex(visit_order)

    base_positions = np.arange(len(visit_order))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['gmean_cpep_auc']
        lower = y - tdf['q25']
        upper = tdf['q75'] - y
        yerr = [lower, upper]
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Control group", -offset)
    add_errorbar(treatment_1, "#f57600", "Closed-loop group", offset)

    plt.title('AUC for C-peptide Level')
    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Geometric Mean (nmol/l)')
    plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=visit_order)
    plt.legend()
    plt.savefig('./data/graphs/testing/cloud/auc_raw.png')
    plt.close()

    '''Glycated Hemoglobin Level (PAPER USES THIS)'''
    treatment_arm = pd.read_csv('./data/studies/cloud/original/PtRoster.txt', sep='|')
    treatment_arm = treatment_arm[['PtID', 'TrtGroup']]
    treatment_arm.rename(columns={'TrtGroup':'treatment_arm'}, inplace=True)

    hb_a1c = pd.read_csv('./data/studies/cloud/original/CloudLabDataHbA1cCap.txt', sep='|')
    hb_a1c = hb_a1c[hb_a1c['VisitCollected'].isin(['Base', '3Mo', '6Mo', '9Mo', '12Mo', '15Mo', '18Mo', '21Mo', '24Mo'])]
    hb_a1c = hb_a1c.merge(treatment_arm, on='PtID')
    hb_a1c['VisitNum'] = (
        hb_a1c['VisitCollected']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    hb_a1c = hb_a1c.sort_values('VisitNum').reset_index(drop=True)
    hb_stats = hb_a1c.groupby(['treatment_arm', 'VisitCollected'], sort=False)['HbA1cMMol'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    visit_order = (
        hb_a1c[['VisitCollected', 'VisitNum']]
        .drop_duplicates()
        .sort_values('VisitNum')['VisitCollected']
        .tolist()
    )

    treatment_0 = hb_stats[hb_stats['treatment_arm'] == 'MDI'].set_index('VisitCollected').reindex(visit_order)
    treatment_1 = hb_stats[hb_stats['treatment_arm'] == 'CL'].set_index('VisitCollected').reindex(visit_order)

    base_positions = np.arange(len(visit_order))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['median']
        lower = y - tdf['q25']
        upper = tdf['q75'] - y
        yerr = [lower, upper]
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Control group", -offset)
    add_errorbar(treatment_1, "#f57600", "Closed-loop group", offset)

    plt.title('Glycated Hemoglobin Level')
    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Median (mmol/mole)')
    plt.xticks(ticks=base_positions, labels=visit_order)
    plt.legend()
    plt.savefig('./data/graphs/testing/cloud/hb_a1c_raw.png')
    plt.close()

def testing_cloud_final():
    ''' AUC for C-Peptide Level'''
    cloud_auc = df_cloud[['id','time_bin', 'treatment_arm', 'cpep_auc']].compute()
    cloud_auc = cloud_auc.drop_duplicates(['id','time_bin']).copy()
    cloud_auc = cloud_auc[cloud_auc['time_bin'].isin(['Baseline', 'Month 6', 'Month 12', 'Month 24'])]

    time_order = ['Baseline', 'Month 6', 'Month 12', 'Month 24']
    gmean_auc = (
        cloud_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .apply(lambda x: gmean(x[x > 0]))  # exclude zeros to avoid math domain error
        .reset_index(name='gmean_cpep_auc')
    )
    iqr_auc = (
        cloud_auc[cloud_auc['cpep_auc'] > 0]
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .quantile([0.25, 0.75])
        .unstack(level=-1)
        .rename(columns={0.25: 'q25', 0.75: 'q75'})
        .reset_index()
    )
    gmean_auc = gmean_auc.merge(iqr_auc, on=['treatment_arm', 'time_bin'], how='left')
    treatment_0 = gmean_auc[gmean_auc['treatment_arm'] == 'MDI'].set_index('time_bin').reindex(time_order)
    treatment_1 = gmean_auc[gmean_auc['treatment_arm'] == 'CL'].set_index('time_bin').reindex(time_order)

    base_positions = np.arange(len(time_order))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['gmean_cpep_auc']
        lower = y - tdf['q25']
        upper = tdf['q75'] - y
        yerr = [lower, upper]
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Control group", -offset)
    add_errorbar(treatment_1, "#f57600", "Closed-loop group", offset)

    plt.title('AUC for C-peptide Level')
    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Geometric Mean (nmol/l)')
    plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=time_order)
    plt.legend()
    plt.savefig('./data/graphs/testing/cloud/auc_final.png')
    plt.close()

    '''Glycated Hemoglobin Level'''
    cloud_hba1c = df_cloud[['id','time_bin', 'treatment_arm', 'hb_a1c']].compute()
    cloud_hba1c = cloud_hba1c.drop_duplicates(['id','time_bin']).copy()
    cloud_hba1c = cloud_hba1c[cloud_hba1c['time_bin'].isin(['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12', 'Month 15', 'Month 18', 'Month 21', 'Month 24'])]
    cloud_hba1c['hb_a1c'] = (cloud_hba1c['hb_a1c'] - 2.15) * 10.929

    hb_stats = cloud_hba1c.groupby(['treatment_arm', 'time_bin'], sort=False)['hb_a1c'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    order_label = ['Base', '3Mo', '6Mo', '9Mo', '12Mo', '15Mo', '18Mo', '21Mo', '24Mo']
    order_bins = ['Baseline', 'Month 3', 'Month 6', 'Month 9', 'Month 12', 'Month 15', 'Month 18', 'Month 21', 'Month 24']
    hb_stats['label'] = hb_stats['time_bin'].map(dict(zip(order_bins, order_label)))
    hb_stats = hb_stats.sort_values(by='label', key=lambda s: [order_label.index(x) for x in s])

    treatment_0 = hb_stats[hb_stats['treatment_arm'] == 'MDI'].set_index('label').reindex(order_label)
    treatment_1 = hb_stats[hb_stats['treatment_arm'] == 'CL'].set_index('label').reindex(order_label)

    base_positions = np.arange(len(order_label))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['median']
        lower = y - tdf['q25']
        upper = tdf['q75'] - y
        yerr = [lower, upper]
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Control group", -offset)
    add_errorbar(treatment_1, "#f57600", "Closed-loop group", offset)

    plt.title('Glycated Hemoglobin Level')
    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Median (mmol/mole)')
    plt.xticks(ticks=base_positions, labels=order_label)
    plt.legend()
    plt.savefig('./data/graphs/testing/cloud/hb_a1c_final.png')
    plt.close()

def testing_clvr_raw():
    visits = ['Randomization', '13 Week', '26 Week', '39 Week', '52 Week']

    '''AUC for C-Peptide Level'''
    treatment_arm = pd.read_csv('./data/studies/clvr/original/subjectsEnroll.txt', sep='|')
    treatment_arm = treatment_arm[['PtID', 'hclGrp']]
    treatment_arm.rename(columns={'hclGrp':'treatment_arm'}, inplace=True)
    treatment_arm['treatment_arm'] = treatment_arm['treatment_arm'].str.replace(r'^\d+\.', '', regex=True).str.strip()

    clvr_auc = pd.read_csv('./data/studies/clvr/original/mmttResults.txt', sep='|')
    clvr_auc.rename(columns={'Visit': 'visit'}, inplace=True)
    clvr_auc['visit'] = clvr_auc['visit'].astype(str).str.strip()
    clvr_auc = clvr_auc[clvr_auc['visit'].isin(visits)]
    clvr_auc['visit'] = pd.Categorical(clvr_auc['visit'], categories=visits, ordered=True)
    clvr_auc = clvr_auc.pivot(index=['PtID', 'visit', 'CollectionDt'], columns='ResultName', values='Value').reset_index()
    clvr_auc = clvr_auc[clvr_auc['visit'].isin(visits)]
    clvr_auc.rename(columns={'C-PEP-0': 'cpep_0_min', 'C-PEP-15': 'cpep_15_min', 'C-PEP-30': 'cpep_30_min',
                            'C-PEP-60': 'cpep_60_min', 'C-PEP-90': 'cpep_90_min', 'C-PEP-120': 'cpep_120_min',
                            'GLU-0': 'glucose_0_min', 'GLU-15': 'glucose_15_min', 'GLU-30': 'glucose_30_min',
                            'GLU-60': 'glucose_60_min', 'GLU-90': 'glucose_90_min', 'GLU-120': 'glucose_120_min'}, inplace=True)
    cpep_cols = [col for col in clvr_auc.columns if col.startswith("cpep_")]
    for col in cpep_cols:
        clvr_auc[col] = clvr_auc[col].apply(
            lambda x: 0.007 if isinstance(x, str) and "<" in x else pd.to_numeric(x, errors='coerce')
        )
    clvr_auc['cpep_auc'] = clvr_auc.apply(get_cpep_auc, axis=1)
    clvr_auc = clvr_auc.merge(treatment_arm, on='PtID')

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_auc,
        x='visit', y='cpep_auc',
        hue='treatment_arm',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    treatments = ['HCL','Non-HCL']

    mean_tbl = (
        clvr_auc.groupby(['visit','treatment_arm'], sort=False)['cpep_auc']
        .mean()
        .unstack('treatment_arm')
        .reindex(index=visits, columns=treatments)
    )
    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = mean_tbl.loc[v, t]
            if pd.isna(y):
                continue
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)
            xs.append(x); ys.append(y)
        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('C-peptide AUC by treatment group')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('C-peptide AUC')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/auc_raw.png')
    plt.close()

    '''Peak C-peptide levels'''
    clvr_auc['peak_cpep'] = clvr_auc[['cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].max(axis=1)
    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_auc,
        x='visit', y='peak_cpep',
        hue='treatment_arm',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    treatments = ['HCL','Non-HCL']

    mean_tbl = (
        clvr_auc.groupby(['visit','treatment_arm'], sort=False)['peak_cpep']
        .mean()
        .unstack('treatment_arm')
        .reindex(index=visits, columns=treatments)
    )

    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = mean_tbl.loc[v, t]
            if pd.isna(y):
                continue
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)
            xs.append(x); ys.append(y)
        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('Peak C-peptide levels')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('Peak C-peptide, nmol/l')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/peak_cpep_raw.png')
    plt.close()

    ''' Time in target glucose range '''
    visits = ['6 Week', '13 Week', '26 Week', '39 Week', '52 Week']
    clvr_cgm = pd.read_csv('./data/studies/clvr/original/cgmAnalysis.txt', sep='|')
    clvr_cgm['DeviceDtTm'] = pd.to_datetime(clvr_cgm['DeviceDtTm'], format="%d%b%Y:%H:%M:%S.%f", errors ='coerce')
    clvr_cgm['glucose'] = clvr_cgm['glucose']/18
    clvr_cgm.rename(columns={'DeviceDtTm' : 'timestamp', 'glucose': 'glucose mmol/l'}, inplace=True)
    clvr_cgm = clvr_cgm.merge(treatment_arm, on='PtID')

    clvr_tir = pd.DataFrame(clvr_cgm.groupby(['PtID', 'treatment_arm', 'Visit'], sort=False).apply(data_analysis.get_cgm_core_endpoints_per_patient)).reset_index()
    metrics_df = pd.json_normalize(clvr_tir[0])
    clvr_tir = pd.concat([clvr_tir.drop(columns=[0]), metrics_df], axis=1)
    clvr_tir = clvr_tir[['PtID', 'treatment_arm', 'Visit', 'TIR']]
    print(clvr_tir)

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_tir,
        x='Visit', y='TIR',
        hue='treatment_arm',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    treatments = ['HCL','Non-HCL']

    mean_tbl = (
        clvr_tir.groupby(['Visit','treatment_arm'], sort=False)['TIR']
        .mean()
        .unstack('treatment_arm')
        .reindex(index=visits, columns=treatments)
    )
    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = mean_tbl.loc[v, t]
            if pd.isna(y):
                continue
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)
            xs.append(x); ys.append(y)
        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('Time in Target glucose range')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('Time in glucose range\nof 3.9-100 mmol/l, %')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/tir_raw.png')
    plt.close()

def testing_clvr_final():
    '''AUC for C-Peptide Level'''
    clvr_auc = df_clvr[['id','visit', 'insulin_delivery', 'cpep_auc']].compute()
    clvr_auc['cpep_auc'] = clvr_auc['cpep_auc']
    clvr_auc = clvr_auc.drop_duplicates(['id','visit']).copy()
    visits = ['Randomization', '13 Week', '26 Week', '39 Week', '52 Week']
    clvr_auc = clvr_auc[clvr_auc['visit'].isin(visits)]
    
    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_auc,
        x='visit', y='cpep_auc',
        hue='insulin_delivery',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    means = (clvr_auc.groupby(['visit','insulin_delivery'])['cpep_auc']
             .mean().reset_index())

    treatments = ['HCL','Non-HCL']

    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = means.query("visit == @v and insulin_delivery == @t")['cpep_auc'].item()
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            xs.append(x); ys.append(y)
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)

        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('C-peptide AUC by treatment group')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('C-peptide AUC')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/auc_final.png')
    plt.close()

    '''Peak C-peptide levels'''
    clvr_auc = df_clvr[['id','visit', 'insulin_delivery', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].compute()
    clvr_auc['peak_cpep'] = clvr_auc[['cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].max(axis=1)
    clvr_auc = clvr_auc.drop_duplicates(['id','visit']).copy()
    clvr_auc = clvr_auc[clvr_auc['visit'].isin(visits)]

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_auc,
        x='visit', y='peak_cpep',
        hue='insulin_delivery',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    means = (clvr_auc.groupby(['visit','insulin_delivery'])['peak_cpep']
             .mean().reset_index())

    treatments = ['HCL','Non-HCL']

    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = means.query("visit == @v and insulin_delivery == @t")['peak_cpep'].item()
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            xs.append(x); ys.append(y)
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)

        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('Peak C-peptide levels')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('Peak C-peptide, nmol/l')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/peak_cpep_final.png')
    plt.close()

    ''' Time in target glucose range '''
    visits = ['6 Week', '13 Week', '26 Week', '39 Week', '52 Week']
    clvr_cgm = df_clvr[['id', 'timestamp', 'visit', 'insulin_delivery', 'glucose mmol/l']].compute()
    clvr_cgm.loc[clvr_cgm['visit'] == 'Randomization', 'visit'] = '6 Week'

    clvr_tir = pd.DataFrame(clvr_cgm.groupby(['id', 'insulin_delivery', 'visit'], sort=False).apply(data_analysis.get_cgm_core_endpoints_per_patient)).reset_index()
    metrics_df = pd.json_normalize(clvr_tir[0])
    clvr_tir = pd.concat([clvr_tir.drop(columns=[0]), metrics_df], axis=1)
    clvr_tir = clvr_tir[['id', 'insulin_delivery', 'visit', 'TIR']]

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=clvr_tir,
        x='visit', y='TIR',
        hue='insulin_delivery',
        hue_order=['HCL','Non-HCL'],
        palette={'Non-HCL': "#008bfc", 'HCL': "#f79400"}
    )
    palette={'Non-HCL': "#003458", 'HCL': "#ff7700bc"}
    treatments = ['HCL','Non-HCL']

    mean_tbl = (
        clvr_tir.groupby(['visit','insulin_delivery'], sort=False)['TIR']
        .mean()
        .unstack('insulin_delivery')
        .reindex(index=visits, columns=treatments)
    )
    box_width = 0.8
    k = len(treatments)
    spacing = box_width / k

    for t in treatments:
        xs, ys = [], []
        for i, v in enumerate(visits):
            y = mean_tbl.loc[v, t]
            if pd.isna(y):
                continue
            x = i - box_width/2 + spacing/2 + (treatments.index(t))*spacing
            ax.scatter(x, y, s=80, color=palette[t], edgecolor='black', zorder=5)
            xs.append(x); ys.append(y)
        ax.plot(xs, ys, linewidth=2, color=palette[t], zorder=4)

    ax.set_title('Time in Target glucose range')
    ax.set_xlabel('Time Since Diabetes Diagnosis (weeks)')
    ax.set_ylabel('Time in glucose range\nof 3.9-100 mmol/l, %')

    ax.legend(title='Treatment')
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/clvr/tir_final.png')
    plt.close()

def testing_defend_raw():
    '''Stimulated C-peptide mean AUC WHOLE COHORT'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12']
    defend_auc = pd.read_csv('./data/studies/defend/original/extra_features.csv')
    defend_auc = defend_auc[['PtID', 'time_bins', 'plcb', 'cpep0', 'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    defend_auc = defend_auc[defend_auc['time_bins'].isin(visits)]
    defend_auc.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'cpep0': 'cpep_0_min','cpep15': 'cpep_15_min',
                               'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min','cpep90': 'cpep_90_min',
                               'cpep120': 'cpep_120_min', 'time_bins': 'time_bin'}, inplace=True)
    defend_auc['cpep_auc'] = defend_auc.apply(get_cpep_auc, axis=1)*0.331
    defend_auc['VisitNum'] = (
        defend_auc['time_bin']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    defend_auc = defend_auc.sort_values('VisitNum').reset_index(drop=True)
    mean_auc = (
        defend_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Otelixizumab", offset)

    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Stimulated C-peptide mean area under\nthe curve (nmol/l)')
    # plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/defend/total_auc_raw.png')
    plt.close()

    '''Stimulated C-peptide mean AUC ADOLESCENTS ONLY'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12']
    defend_auc = pd.read_csv('./data/studies/defend/original/extra_features.csv')
    defend_auc = defend_auc[['PtID', 'age', 'time_bins', 'plcb', 'cpep0', 'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    defend_auc = defend_auc[defend_auc['time_bins'].isin(visits)]
    defend_auc = defend_auc[(defend_auc['age'] > 12) & (defend_auc['age'] <= 17)]
    defend_auc.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'cpep0': 'cpep_0_min','cpep15': 'cpep_15_min',
                               'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min','cpep90': 'cpep_90_min',
                               'cpep120': 'cpep_120_min', 'time_bins': 'time_bin'}, inplace=True)
    defend_auc['cpep_auc'] = defend_auc.apply(get_cpep_auc, axis=1)*0.331
    defend_auc['VisitNum'] = (
        defend_auc['time_bin']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    defend_auc = defend_auc.sort_values('VisitNum').reset_index(drop=True)
    mean_auc = (
        defend_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Otelixizumab", offset)

    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Stimulated C-peptide mean area under\nthe curve (nmol/l)')
    # plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/defend/adolescence_auc_raw.png')
    plt.close()

def testing_defend_final():
    '''Stimulated C-peptide mean AUC WHOLE COHORT'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12']
    defend_auc = df_defend[['id', 'time_bin', 'treatment_arm', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc']].compute()
    defend_auc = defend_auc.drop_duplicates(['id','time_bin']).copy()
    defend_auc = defend_auc[defend_auc['time_bin'].isin(visits)]
    defend_auc['VisitNum'] = (
        defend_auc['time_bin']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    defend_auc = defend_auc.sort_values('VisitNum').reset_index(drop=True)
    mean_auc = (
        defend_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Otelixizumab", offset)

    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Stimulated C-peptide mean area under\nthe curve (nmol/l)')
    # plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/defend/total_auc_final.png')
    plt.close()

    '''Stimulated C-peptide mean AUC ADOLESCENTS ONLY'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12']
    defend_auc = df_defend[['id', 'time_bin', 'age', 'treatment_arm', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc']].compute()
    defend_auc = defend_auc.drop_duplicates(['id','time_bin']).copy()
    defend_auc = defend_auc[defend_auc['time_bin'].isin(visits)]
    defend_auc = defend_auc[(defend_auc['age'] > 12) & (defend_auc['age'] <= 17)]
    defend_auc['cpep_auc'] = defend_auc['cpep_auc']*0.331
    defend_auc['VisitNum'] = (
        defend_auc['time_bin']
        .str.extract(r'(\d+)')        # extract number from the string
        .astype(float)                # convert to numeric (NaN for "Base")
        .fillna(0)                    # treat 'Base' as 0
    )
    defend_auc = defend_auc.sort_values('VisitNum').reset_index(drop=True)
    mean_auc = (
        defend_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Otelixizumab", offset)

    plt.xlabel('Month Since Diagnosis')
    plt.ylabel('Stimulated C-peptide mean area under\nthe curve (nmol/l)')
    # plt.ylim(0, 1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/defend/adolescence_auc_final.png')
    plt.close()

def testing_diagnode_raw():
    ''' Change in AUC C-peptide (FAS) '''
    visits = ['Baseline', 'Month 6', 'Month 15']
    diagnode_auc = pd.read_csv('./data/studies/diagnode/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')
    diagnode_auc = diagnode_auc[['id', 'visit_name', 'treatment', 'cpep0', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    diagnode_auc.rename(columns={'treatment': 'treatment_arm', 'visit_name': 'visit', 'cpep0': 'cpep_0_min', 'cpep30': 'cpep_30_min',
                                'cpep60': 'cpep_60_min', 'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min'}, inplace=True)
    diagnode_auc = diagnode_auc[diagnode_auc['visit'].isin(visits)]
    diagnode_auc['cpep_auc'] = diagnode_auc.apply(get_cpep_auc, axis=1)
    # Keep only observations with positive AUC to support log-scale summaries
    diagnode_auc = diagnode_auc[diagnode_auc['cpep_auc'] > 0]

    # Baseline mapping per patient
    baseline_values = diagnode_auc.loc[diagnode_auc['visit'] == 'Baseline', ['id', 'cpep_auc']]
    baseline_dict = dict(zip(baseline_values['id'], baseline_values['cpep_auc']))
    diagnode_auc['baseline_cpep_auc'] = diagnode_auc['id'].map(baseline_dict)

    # Relative change (ratio to own baseline)
    diagnode_auc['relative_change'] = diagnode_auc['cpep_auc'] / diagnode_auc['baseline_cpep_auc']
    diagnode_auc = diagnode_auc.dropna(subset=['baseline_cpep_auc', 'relative_change'])

    # Mean and 95% CI of relative change per treatment and visit
    summary = (
        diagnode_auc
        .groupby(['treatment_arm', 'visit'], sort=False)['relative_change']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci_95'] = summary['sem'] * 1.96  # normal approx

    base_positions = np.arange(len(visits))
    offset = 0.08
    palette = {'Placebo': "#f57600", 'Diamyd': "#1000f5"}
    legend_labels = {'Placebo': 'Placebo', 'Diamyd': 'GAD-alum'}

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean']
        yerr = tdf['ci_95']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    for arm, shift in [('Placebo', -offset), ('Diamyd', offset)]:
        arm_df = summary[summary['treatment_arm'] == arm].set_index('visit').reindex(visits)
        add_errorbar(arm_df, palette.get(arm, 'black'), legend_labels.get(arm, arm), shift)

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylabel('Change in AUC C-peptide')
    plt.ylim(0,1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/diagnode/general_auc_raw.png')
    plt.close()

    ''' Change in AUC C-peptide (DR3-QD2) '''
    visits = ['Baseline', 'Month 6', 'Month 15']
    diagnode_auc = pd.read_csv('./data/studies/diagnode/original/ALJC_clean_DIAGNODE_analysis_dataset_perch.csv')
    diagnode_auc = diagnode_auc[['id', 'dr3_dq2', 'visit_name', 'treatment', 'cpep0', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    diagnode_auc.rename(columns={'treatment': 'treatment_arm', 'visit_name': 'visit', 'cpep0': 'cpep_0_min', 'cpep30': 'cpep_30_min',
                                'cpep60': 'cpep_60_min', 'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min'}, inplace=True)
    diagnode_auc = diagnode_auc[diagnode_auc['visit'].isin(visits)]
    diagnode_auc = diagnode_auc[diagnode_auc['dr3_dq2'] == 1]
    dr3_ids = 'Diagnode_' + diagnode_auc['id'].unique()
    diagnode_auc['cpep_auc'] = diagnode_auc.apply(get_cpep_auc, axis=1)
    # Keep only observations with positive AUC to support log-scale summaries
    diagnode_auc = diagnode_auc[diagnode_auc['cpep_auc'] > 0]

    # Baseline mapping per patient
    baseline_values = diagnode_auc.loc[diagnode_auc['visit'] == 'Baseline', ['id', 'cpep_auc']]
    baseline_dict = dict(zip(baseline_values['id'], baseline_values['cpep_auc']))
    diagnode_auc['baseline_cpep_auc'] = diagnode_auc['id'].map(baseline_dict)

    # Relative change (ratio to own baseline)
    diagnode_auc['relative_change'] = diagnode_auc['cpep_auc'] / diagnode_auc['baseline_cpep_auc']
    diagnode_auc = diagnode_auc.dropna(subset=['baseline_cpep_auc', 'relative_change'])

    # Mean and 95% CI of relative change per treatment and visit
    summary = (
        diagnode_auc
        .groupby(['treatment_arm', 'visit'], sort=False)['relative_change']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci_95'] = summary['sem'] * 1.96  # normal approx

    base_positions = np.arange(len(visits))
    offset = 0.08
    palette = {'Placebo': "#f57600", 'Diamyd': "#1000f5"}
    legend_labels = {'Placebo': 'Placebo', 'Diamyd': 'GAD-alum'}

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean']
        yerr = tdf['ci_95']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    for arm, shift in [('Placebo', -offset), ('Diamyd', offset)]:
        arm_df = summary[summary['treatment_arm'] == arm].set_index('visit').reindex(visits)
        add_errorbar(arm_df, palette.get(arm, 'black'), legend_labels.get(arm, arm), shift)

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylabel('Change in AUC C-peptide')
    plt.ylim(0,1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/diagnode/dr3_auc_raw.png')
    plt.close()

    ''' IDAAC, HbA1C, and Insulin Dose Forest Plots '''
    #TODO

    return dr3_ids

def testing_diagnode_final(dr3_ids):
    ''' Change in AUC C-peptide (FAS) '''
    visits = ['Baseline', 'Month 6', 'Month 15']
    diagnode_auc = df_diagnode[['id', 'time_bin', 'treatment_arm', 'cpep_0_min', 'cpep_30_min',
                                'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc', 'cpep_auc_preservation']].compute()
    diagnode_auc = diagnode_auc.drop_duplicates(['id','time_bin']).copy()
    diagnode_auc['cpep_auc_preservation'] = diagnode_auc['cpep_auc_preservation']/100
    diagnode_auc['time_bin'] = diagnode_auc['time_bin'].str.strip()
    diagnode_auc = diagnode_auc[diagnode_auc['time_bin'].isin(visits)]

    # Mean and 95% CI of relative change per treatment and visit
    summary = (
        diagnode_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc_preservation']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )

    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci_95'] = summary['sem'] * 1.96  # normal approx

    base_positions = np.arange(len(visits))
    offset = 0.08
    palette = {'Placebo': "#f57600", 'Diamyd': "#1000f5"}
    legend_labels = {'Placebo': 'Placebo', 'Diamyd': 'GAD-alum'}

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean']
        yerr = tdf['ci_95']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    for arm, shift in [('Placebo', -offset), ('Diamyd', offset)]:
        arm_df = summary[summary['treatment_arm'] == arm].set_index('time_bin').reindex(visits)
        add_errorbar(arm_df, palette.get(arm, 'black'), legend_labels.get(arm, arm), shift)

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylabel('Change in AUC C-peptide')
    plt.ylim(0,1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/diagnode/general_auc_final.png')
    plt.close()

    ''' Change in AUC C-peptide (DR3-QD2) '''
    visits = ['Baseline', 'Month 6', 'Month 15']
    diagnode_auc = df_diagnode[['id', 'time_bin', 'treatment_arm', 'cpep_0_min', 'cpep_30_min',
                                'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc', 'cpep_auc_preservation']].compute()
    diagnode_auc = diagnode_auc.drop_duplicates(['id','time_bin']).copy()
    diagnode_auc['cpep_auc_preservation'] = diagnode_auc['cpep_auc_preservation']/100
    diagnode_auc['time_bin'] = diagnode_auc['time_bin'].str.strip()
    diagnode_auc = diagnode_auc[diagnode_auc['time_bin'].isin(visits)]
    diagnode_auc = diagnode_auc[diagnode_auc['id'].isin(dr3_ids)]

    summary = (
        diagnode_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc_preservation']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )

    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci_95'] = summary['sem'] * 1.96  # normal approx

    base_positions = np.arange(len(visits))
    offset = 0.08
    palette = {'Placebo': "#f57600", 'Diamyd': "#1000f5"}
    legend_labels = {'Placebo': 'Placebo', 'Diamyd': 'GAD-alum'}

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean']
        yerr = tdf['ci_95']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    for arm, shift in [('Placebo', -offset), ('Diamyd', offset)]:
        arm_df = summary[summary['treatment_arm'] == arm].set_index('time_bin').reindex(visits)
        add_errorbar(arm_df, palette.get(arm, 'black'), legend_labels.get(arm, arm), shift)

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylabel('Change in AUC C-peptide')
    plt.ylim(0,1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/graphs/testing/diagnode/dr3_auc_final.png')
    plt.close()

    ''' IDAAC, HbA1C, and Insulin Dose Forest Plots '''
    #TODO

    pass

def testing_gskalb_raw():
    ''' Mean +- SE C-peptide AUC'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12', 'Month 15']
    gskalb_auc = pd.read_csv('./data/studies/gskalb/original/extra_features.csv')
    gskalb_auc = gskalb_auc[['PtID', 'plcb', 'DaysFromEnroll', 'time_bins', 'cpep0', 'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    gskalb_auc.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'DaysFromEnroll': 'dy', 'cpep0': 'cpep_0_min',
                               'cpep15': 'cpep_15_min', 'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min',
                               'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min', 'time_bins': 'time_bin'}, inplace=True)
    gskalb_auc['cpep_auc'] = gskalb_auc.apply(get_cpep_auc, axis=1)*0.331
    gskalb_auc.loc[(gskalb_auc['dy'] >= 390) & (gskalb_auc['dy'] < 480), 'time_bin'] = 'Month 15'

    mean_auc = (
        gskalb_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Albiglutide", offset)

    plt.xlabel('Month')
    plt.ylabel('Mean ± SE C-peptide AUC (nmol/l)')
    plt.ylim(0.2, 0.75)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/gskalb/auc_raw.png')
    plt.close()

    ''' Mean +- SE Maximum Simulated C-peptide'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12', 'Month 15']
    gskalb_peak_cpep = pd.read_csv('./data/studies/gskalb/original/extra_features.csv')
    gskalb_peak_cpep = gskalb_peak_cpep[['PtID', 'plcb', 'DaysFromEnroll', 'time_bins', 'cpep0', 'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    gskalb_peak_cpep.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'DaysFromEnroll': 'dy', 'cpep0': 'cpep_0_min',
                               'cpep15': 'cpep_15_min', 'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min',
                               'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min', 'time_bins': 'time_bin'}, inplace=True)
    gskalb_peak_cpep.loc[(gskalb_peak_cpep['dy'] >= 390) & (gskalb_peak_cpep['dy'] < 480), 'time_bin'] = 'Month 15'
    gskalb_peak_cpep['peak_cpep'] = gskalb_peak_cpep[['cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].max(axis=1)*0.331

    mean_peak_cpep = (
        gskalb_peak_cpep
        .groupby(['treatment_arm', 'time_bin'], sort=False)['peak_cpep']
        .agg(mean_peak_cpep='mean', std='std', count='count')
        .reset_index()
    )
    mean_peak_cpep['sem'] = mean_peak_cpep['std'] / np.sqrt(mean_peak_cpep['count'])

    treatment_0 = mean_peak_cpep[mean_peak_cpep['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_peak_cpep[mean_peak_cpep['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)
    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_peak_cpep']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Albiglutide", offset)

    plt.xlabel('Month')
    plt.ylabel('Mean ± SE Maximum Simulated\nC-peptide (nmol/l)')
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylim(0.4,1.2)
    plt.legend()
    plt.savefig('./data/graphs/testing/gskalb/peak_cpep_raw.png')
    plt.close()

    ''' Mean Change from Baseline in Time Spent (%)'''
    #TODO

def testing_gskalb_final():
    ''' Mean +- SE C-peptide AUC'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12', 'Month 15']
    gskalb_auc = df_gskalb[['id', 'time_bin', 'dy', 'treatment_arm', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc']].compute()
    gskalb_auc = gskalb_auc.drop_duplicates(['id','time_bin']).copy()

    mean_auc = (
        gskalb_auc
        .groupby(['treatment_arm', 'time_bin'], sort=False)['cpep_auc']
        .agg(mean_cpep_auc='mean', std='std', count='count')
        .reset_index()
    )
    mean_auc['sem'] = mean_auc['std'] / np.sqrt(mean_auc['count'])

    treatment_0 = mean_auc[mean_auc['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_auc[mean_auc['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)

    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_cpep_auc']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Albiglutide", offset)

    plt.xlabel('Month')
    plt.ylabel('Mean ± SE C-peptide AUC (nmol/l)')
    plt.ylim(0.2, 0.75)
    plt.xticks(ticks=base_positions, labels=visits)
    plt.legend()
    plt.savefig('./data/graphs/testing/gskalb/auc_final.png')
    plt.close()

    ''' Mean +- SE Maximum Simulated C-peptide'''
    visits = ['Baseline', 'Month 3', 'Month 6', 'Month 12', 'Month 15']
    gskalb_peak_cpep = df_gskalb[['id', 'time_bin', 'dy', 'treatment_arm', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].compute()

    gskalb_peak_cpep['peak_cpep'] = gskalb_peak_cpep[['cpep_0_min', 'cpep_15_min', 'cpep_30_min', 'cpep_60_min', 'cpep_90_min', 'cpep_120_min']].max(axis=1)
    gskalb_peak_cpep = gskalb_peak_cpep.drop_duplicates(['id','time_bin']).copy()

    mean_peak_cpep = (
        gskalb_peak_cpep
        .groupby(['treatment_arm', 'time_bin'], sort=False)['peak_cpep']
        .agg(mean_peak_cpep='mean', std='std', count='count')
        .reset_index()
    )
    mean_peak_cpep['sem'] = mean_peak_cpep['std'] / np.sqrt(mean_peak_cpep['count'])

    treatment_0 = mean_peak_cpep[mean_peak_cpep['treatment_arm'] == 'Control'].set_index('time_bin').reindex(visits)
    treatment_1 = mean_peak_cpep[mean_peak_cpep['treatment_arm'] == 'Active'].set_index('time_bin').reindex(visits)
    base_positions = np.arange(len(visits))
    offset = 0.08

    def add_errorbar(tdf, color, label, shift):
        y = tdf['mean_peak_cpep']
        yerr = tdf['sem']
        plt.errorbar(
            base_positions + shift, y, yerr=yerr,
            label=label, color=color, fmt='-o', capsize=5, linewidth=2, markersize=6
        )

    add_errorbar(treatment_0, "#1000f5", "Placebo", -offset)
    add_errorbar(treatment_1, "#f57600", "Albiglutide", offset)

    plt.xlabel('Month')
    plt.ylabel('Mean ± SE Maximum Simulated\nC-peptide (nmol/l)')
    plt.xticks(ticks=base_positions, labels=visits)
    plt.ylim(0.4,1.2)
    plt.legend()
    plt.savefig('./data/graphs/testing/gskalb/peak_cpep_final.png')
    plt.close()

    ''' Mean Change from Baseline in Time Spent (%)'''
    #TODO

def _style_jaeb_boxplot(ax, hue_order, fill_colors):
    """Apply grayscale styling with solid black outlines like the reference figure."""
    for i, patch in enumerate(ax.artists):
        hue = hue_order[i % len(hue_order)]
        patch.set_facecolor(fill_colors[hue])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        patch.set_alpha(1)
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(1.2)

def _plot_jaeb_boxplot(df, visits, value_col, output_path, ylabel, ylim_a, ylim_b):
    hue_order = ['Control', 'Active']
    fill_colors = {'Control': '#ffffff', 'Active': '#c0c0c0'}
    marker_faces = {'Control': '#ffffff', 'Active': '#000000'}

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df,
        x='time_bin', y=value_col,
        order=visits,
        hue='treatment_arm',
        hue_order=hue_order,
        palette=fill_colors,
        width=0.65,
        linewidth=1.4,
        whis=0,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.3},
        whiskerprops={'linewidth': 0},
        capprops={'linewidth': 0}
    )
    _style_jaeb_boxplot(ax, hue_order, fill_colors)

    mean_tbl = (
        df.groupby(['time_bin', 'treatment_arm'], sort=False)[value_col]
        .mean()
        .unstack('treatment_arm')
        .reindex(index=visits, columns=hue_order)
    )
    box_width = 0.65
    spacing = box_width / len(hue_order)

    for t in hue_order:
        xs, ys = [], []
        for i, v in enumerate(visits):
            if v not in mean_tbl.index:
                continue
            y = mean_tbl.loc[v, t]
            if pd.isna(y):
                continue
            x = i - box_width/2 + spacing/2 + (hue_order.index(t))*spacing
            ax.scatter(
                x, y, s=70, facecolors=marker_faces[t],
                edgecolors='black', linewidths=1.2, zorder=5
            )
            xs.append(x); ys.append(y)
        if xs:
            ax.plot(xs, ys, color='black', linewidth=1.2, zorder=4)

    ax.set_xlabel('Visit')
    ax.set_ylabel(ylabel)
    if ax.legend_:
        ax.legend_.remove()
    legend_handles = [
        Patch(facecolor=fill_colors[t], edgecolor='black', label=t)
        for t in hue_order
    ]
    ax.legend(handles=legend_handles, title='Treatment', loc='upper left', frameon=False)
    plt.tight_layout()
    plt.ylim(ylim_a, ylim_b)
    plt.savefig(output_path)
    plt.close()

def testing_jaeb_raw():
    ''' C-peptide AUC '''
    visits = ['Baseline', 'Week 6', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
    jaeb_auc = pd.read_csv('./data/studies/jaeb_t1d/original/extra_features.csv')
    jaeb_auc = jaeb_auc[['PtID', 'plcb', 'DaysFromEnroll', 'time_bins', 'cpep0', 'cpep15', 'cpep30', 'cpep60', 'cpep90', 'cpep120']]
    jaeb_auc.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'DaysFromEnroll': 'dy', 'cpep0': 'cpep_0_min',
                               'cpep15': 'cpep_15_min', 'cpep30': 'cpep_30_min','cpep60': 'cpep_60_min',
                               'cpep90': 'cpep_90_min', 'cpep120': 'cpep_120_min', 'time_bins': 'time_bin'}, inplace=True)
    jaeb_auc['cpep_auc'] = jaeb_auc.apply(get_cpep_auc, axis=1)*0.331
    jaeb_auc.loc[jaeb_auc['dy'] < 30 , 'time_bin'] = 'Baseline'
    jaeb_auc.loc[(jaeb_auc['dy'] > 30) & (jaeb_auc['dy'] < 60) , 'time_bin'] = 'Week 6'
    jaeb_auc = jaeb_auc[jaeb_auc['time_bin'].isin(visits)].copy()

    _plot_jaeb_boxplot(jaeb_auc, visits, 'cpep_auc', './data/graphs/testing/jaeb_t1d/auc_raw.png', 'C-peptide AUC (%)', ylim_a=0, ylim_b=1.2)

    ''' HbA1C (%)'''
    visits = ['Week 6', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
    jaeb_a1c = pd.read_csv('./data/studies/jaeb_t1d/original/extra_features.csv')
    jaeb_a1c = jaeb_a1c[['PtID', 'plcb', 'DaysFromEnroll', 'time_bins', 'hba1c']]
    jaeb_a1c.rename(columns={'PtID': 'id', 'plcb': 'treatment_arm', 'DaysFromEnroll': 'dy', 'hba1c': 'hb_a1c', 'time_bins': 'time_bin'}, inplace=True)
    jaeb_a1c.loc[(jaeb_a1c['dy'] > 30) & (jaeb_a1c['dy'] < 60) , 'time_bin'] = 'Week 6'
    jaeb_a1c = jaeb_a1c[jaeb_a1c['time_bin'].isin(visits)].copy()

    _plot_jaeb_boxplot(jaeb_a1c, visits, 'hb_a1c', './data/graphs/testing/jaeb_t1d/a1c_raw.png', 'HbA1C (%)', ylim_a=5, ylim_b=9)

def testing_jaeb_final():
    
    ''' Number of median CGM days avaialble'''
    visits = ['Baseline', 'Week 6', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
    cgm_dfs = _to_pandas(df_jaeb_t1d[['id', 'timestamp', 'time_bin', 'treatment_arm', 'glucose mmol/l']])
    for time_bin in visits:
        print('\n',time_bin)
        cgm_df = cgm_dfs[(cgm_dfs['time_bin'] == time_bin)&(cgm_dfs['treatment_arm'] == 'Active')].dropna(subset=['timestamp']).copy()
        if not cgm_df.empty:
            cgm_df['timestamp'] = pd.to_datetime(cgm_df['timestamp'])
            coverage_days = (
                cgm_df.groupby('id')['timestamp']
                .agg(lambda s: s.dt.normalize().nunique())  # count distinct calendar days with CGM readings
            )
            median_days = coverage_days.median()
            print(f"Median CGM coverage days (distinct days with data) among CGM users: {median_days:.1f}")
        else:
            print("Median CGM coverage days (time span): no CGM data in this time bin")

    ''' C-peptide AUC '''
    visits = ['Baseline', 'Week 6', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
    jaeb_auc = _to_pandas(df_jaeb_t1d[['id', 'time_bin', 'dy', 'treatment_arm', 'cpep_0_min', 'cpep_15_min', 'cpep_30_min',
                            'cpep_60_min', 'cpep_90_min', 'cpep_120_min', 'cpep_auc']])
    jaeb_auc.loc[jaeb_auc['dy'] < 30 , 'time_bin'] = 'Baseline'
    jaeb_auc.loc[(jaeb_auc['dy'] > 30) & (jaeb_auc['dy'] < 60) , 'time_bin'] = 'Week 6'
    jaeb_auc = jaeb_auc.drop_duplicates(['id', 'time_bin']).copy()
    jaeb_auc = jaeb_auc[jaeb_auc['time_bin'].isin(visits)].copy()

    _plot_jaeb_boxplot(jaeb_auc, visits, 'cpep_auc', './data/graphs/testing/jaeb_t1d/auc_final.png', 'C-peptide AUC (%)', ylim_a=0, ylim_b=3)

    ''' HbA1C (%)'''
    visits = ['Week 6', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
    jaeb_a1c = _to_pandas(df_jaeb_t1d[['id', 'time_bin', 'dy', 'treatment_arm', 'hb_a1c']])
    jaeb_a1c.loc[(jaeb_a1c['dy'] > 30) & (jaeb_auc['dy'] < 60) , 'time_bin'] = 'Week 6'
    jaeb_a1c = jaeb_a1c.drop_duplicates(['id', 'time_bin']).copy()
    jaeb_a1c = jaeb_a1c[jaeb_a1c['time_bin'].isin(visits)].copy()

    _plot_jaeb_boxplot(jaeb_a1c, visits, 'hb_a1c', './data/graphs/testing/jaeb_t1d/a1c_final.png', 'HbA1C (%)', ylim_a=5, ylim_b=9 )

if __name__ == '__main__':
    # testing_cloud_raw()
    # testing_cloud_final()

    # testing_clvr_raw()
    # testing_clvr_final()

    # testing_defend_raw()
    # testing_defend_final()

    # dr3_ids = testing_diagnode_raw()
    # testing_diagnode_final(dr3_ids)

    # testing_gskalb_raw()
    # testing_gskalb_final()

    testing_jaeb_raw()
    testing_jaeb_final()
