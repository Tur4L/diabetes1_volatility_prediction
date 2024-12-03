import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold, train_test_split
from utils import *
import pickle

#pytorch stuff:
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

MAX_ENCODER_LENGTH = 4
MAX_PREDICTION_LENGTH = 1
FOLD = 5

#Reading data
df_path = "../data/data_processed.csv"  # uncomment for main data
df = pd.read_csv(df_path) 
df = transformer_data(df, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)

#create sine and cosine TOD:
df['time_sin'] = np.sin(2 * np.pi * df['RESULT_TOD']/24)
df['time_cos'] = np.cos(2 * np.pi * df['RESULT_TOD']/24)
df = df.drop('RESULT_TOD',axis=1)

# Normalize or standardize relevant features
features_to_scale = ['WEIGHT_KG', 'HEIGHT_CM', 'AGE', 'TIME_DELTA_LAST', 'TIME_DELTA_NEXT']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Adding a sequential time index within each patient's encounter
df['time_idx'] = df.groupby(['STUDY_ID', 'ENCOUNTER_NUM']).cumcount()

#creating test and train data
df["study_encounter_id"] = df['STUDY_ID'].astype(str) + "-" + df['ENCOUNTER_NUM'].astype(str)
unique_groups = df['study_encounter_id'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.1, random_state=42)
train_df = df[df['study_encounter_id'].isin(train_groups)]
test_df = df[df['study_encounter_id'].isin(test_groups)]

train_df = train_df.drop(columns=['study_encounter_id'])
test_df = test_df.drop(columns=['study_encounter_id'])
df = df.drop(columns=['study_encounter_id'])

train_df.to_csv("./models/tft/train.csv",index=False)
test_df.to_csv("./models/tft/test.csv",index=False)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Creating KFolds:
groups = df['STUDY_ID'].astype(str) + "-" + df['ENCOUNTER_NUM'].astype(str)
groups_kf = GroupKFold(n_splits=FOLD)

for fold_idx, (train_idx, val_idx) in enumerate(groups_kf.split(df, groups=groups)):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

# groups = np.load("./models/rnn/folds.npy", allow_pickle=True)

# for fold_idx, (train_idx, val_idx) in enumerate(groups):
#     train_df = train_df[train_df['STUDY_ID'].isin(train_idx)]
#     val_df = train_df[train_df['STUDY_ID'].isin(val_idx)]

    #Time Series Data Set creating for TFT
    static_reals = ["AGE", "WEIGHT_KG", "HEIGHT_CM","SEX_Male"]
    time_varying_known_reals = ["TIME_DELTA_LAST", "TIME_DELTA_NEXT","time_sin","time_cos","MEAL_bedtime", "MEAL_breakfast","MEAL_lunch","MEAL_supper"]
    time_varying_unknown_reals = ["GLUCOSE (mmol/L)","4A0","A31","A41","A49","B37","B44","B46","B49","B96","B99","C34","C80","D15","D21","D64","D72","E87","I05","I07","I08","I20","I21","I24","I25","I26","I31","I33","I34","I35","I38","I42","I45","I47","I48","I49","I50","I51","I60","I62","I63","I64","I70","I71","I77","J15","J18","J44","J81","J84","J86","J95","J96","J98","K81","L03","M46","M86","N39","Q21","Q22","Q23","Q24","Q89","R00","R04","R06","R07","R09","R10","R32","R41","R50","R55","R57","R58","R94","S21","S22","S81","S90","T14","T17","T79","T81","T82","T84","U82","U83","Y71","Y83","Y84","Y95","Z13","Z22","Z45","Z76","Z82","Z92","Z94","Z95","Z98","Z99","A10A","A10B","H02A","H02B","J01A","J01C","J01D","J01E","J01F","J01G","J01M","J01X","J02A","J04B",
                                "MEDICATION_ATC_A10A","MEDICATION_ATC_A10AB01","MEDICATION_ATC_A10AB04","MEDICATION_ATC_A10AC01","MEDICATION_ATC_A10AD04","MEDICATION_ATC_A10AE04","MEDICATION_ATC_A10AE05","MEDICATION_ATC_A10AE06","MEDICATION_ATC_A10B","MEDICATION_ATC_B05B","MEDICATION_ATC_H02A","MEDICATION_ATC_J01A","MEDICATION_ATC_J01C","MEDICATION_ATC_J01D","MEDICATION_ATC_J01E","MEDICATION_ATC_J01F","MEDICATION_ATC_J01M","MEDICATION_ATC_J01X","MEDICATION_ATC_J02A","MEDICATION_ATC_J04A","MEDICATION_ATC_V06D","MEDICATION_ATC_V07A",
                                "OR_PROC_ID_10700068","OR_PROC_ID_10700075","OR_PROC_ID_10700085","OR_PROC_ID_10700094","OR_PROC_ID_10700099","OR_PROC_ID_10700104","OR_PROC_ID_10700105","OR_PROC_ID_10700109","OR_PROC_ID_10700110","OR_PROC_ID_10700115","OR_PROC_ID_10700116","OR_PROC_ID_10700122","OR_PROC_ID_10700125","OR_PROC_ID_10700132","OR_PROC_ID_10700136","OR_PROC_ID_10700202","OR_PROC_ID_10700204","OR_PROC_ID_10700213","OR_PROC_ID_10700237","OR_PROC_ID_10700247","OR_PROC_ID_10702735","OR_PROC_ID_10703149","OR_PROC_ID_10703335","OR_PROC_ID_10704331","OR_PROC_ID_1180000001","OR_PROC_ID_1180000003","OR_PROC_ID_1180000014","OR_PROC_ID_1180000046","OR_PROC_ID_1180000065","OR_PROC_ID_1180000070","OR_PROC_ID_1180000087","OR_PROC_ID_1180100004","OR_PROC_ID_1180100006",
                                "COMPONENT_ID_140","COMPONENT_ID_847","COMPONENT_ID_864","COMPONENT_ID_882","COMPONENT_ID_1534435","COMPONENT_ID_1577876","COMPONENT_ID_1740200","COMPONENT_ID_1231000011","COMPONENT_ID_1231000018","COMPONENT_ID_1239904222","COMPONENT_ID_1239916328","COMPONENT_ID_1239917011","COMPONENT_ID_1239917188","COMPONENT_ID_12371534435","COMPONENT_ID_12371577876"]


    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="GLUCOSE (mmol/L)",
        group_ids=["STUDY_ID", "ENCOUNTER_NUM"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        categorical_encoders={"STUDY_ID": NaNLabelEncoder(add_nan=True), "ENCOUNTER_NUM": NaNLabelEncoder(add_nan=True)},
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals
        # add additional parameters as necessary
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict= True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers = 0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=320, num_workers = 0)
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("./models/tft/transformer_model/lightning_logs")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.33790950450435064,
        # limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback],
        logger= logger
)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0018992869325045636,
        hidden_size=121,
        attention_head_size=2,
        dropout=0.13339642919448905,
        hidden_continuous_size=105,
        loss=MAE(),
        log_interval=10,
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Tuning Hyperparameters
    # # create study
    # study = optimize_hyperparameters(
    #     train_dataloader,
    #     val_dataloader,
    #     model_path="optuna_test",
    #     n_trials=200,
    #     max_epochs=50,
    #     gradient_clip_val_range=(0.01, 1.0),
    #     hidden_size_range=(8, 128),
    #     hidden_continuous_size_range=(8, 128),
    #     attention_head_size_range=(1, 4),
    #     learning_rate_range=(0.001, 0.1),
    #     dropout_range=(0.1, 0.3),
    #     trainer_kwargs=dict(limit_train_batches=30),
    #     reduce_on_plateau_patience=4,
    #     use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    # )

    # # save study results - also we can resume tuning at a later point in time
    # with open("test_study.pkl", "wb") as fout:
    #     pickle.dump(study, fout)

    # # show best hyperparameters
    # print(study.best_trial.params)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Evaluate data:
    best_model_path = trainer.checkpoint_callback.best_model_path

    # best_model_path = "./models/tft/transformer_models/lightning_logs/lightning_logs/version_14/checkpoints/epoch=19-step=9020.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_15/checkpoints/epoch=19-step=9020.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_16/checkpoints/epoch=19-step=9020.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_17/checkpoints/epoch=15-step=7216.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_18/checkpoints/epoch=19-step=9020.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_20/checkpoints/epoch=19-step=8300.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_21/checkpoints/epoch=19-step=8300.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_22/checkpoints/epoch=19-step=8300.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_23/checkpoints/epoch=12-step=5395.ckpt"
    # best_model_path = "./models/tft/transformer_model/lightning_logs/lightning_logs/version_24/checkpoints/epoch=19-step=8300.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    #Calculate errors
    group_ids = test_df.groupby(["STUDY_ID","ENCOUNTER_NUM"])
    y_predictions = np.array([])
    y_actuals = np.array([])

    #loop over each patient-encounter
    for (sid,enum), group in group_ids:
        test = test_df.loc[(test_df['STUDY_ID'] == sid) & (test_df['ENCOUNTER_NUM'] == enum)]
        index = 0
        y_prediction = np.array([])
        y_actual = np.array([])

        #loop for each patient
        for i in range(test.shape[0] - 5):
            test_df_part = test.iloc[index:index+5]
            testing = TimeSeriesDataSet.from_dataset(training, test_df_part, predict= True, stop_randomization=True)
            test_dataloader = testing.to_dataloader(train=False, batch_size=320, num_workers = 0)
            predictions = best_tft.predict(test_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
            y_prediction = np.concatenate((y_prediction,np.array(predictions.output.T.tolist()[0])))
            y_actual = np.concatenate((y_actual,np.array(predictions.y[0].T.tolist()[0])))
            index += 1
        y_predictions = np.concatenate((y_predictions,y_prediction))
        y_actuals = np.concatenate((y_actuals,y_actual))

    i_MAE,i_gMAE,i_RMSE,i_gRMSE,i_MAPE,i_MARD,i_gMARD = find_confidence_errors_w_intervals(np.array(list(y_predictions)),np.array(list(y_actuals)))
    ce = clark_error_perc(list(y_actuals),list(y_predictions))
    print(f"i_MAE: {i_MAE[0]} +- {i_MAE[1]}")
    print(f"i_gMAE: {i_gMAE[0]} +- {i_gMAE[1]}")
    print(f"i_RMSE: {i_RMSE[0]} +- {i_RMSE[1]}")
    print(f"i_gRMSE: {i_gRMSE[0]} +- {i_gRMSE[1]}")
    print(f"i_MAPE: {i_MAPE[0]} +- {i_MAPE[1]}")
    print(f"i_MARD: {i_MARD[0]} +- {i_MARD[1]}")
    print(f"i_gMARD: {i_gMARD[0]} +- {i_gMARD[1]}")
    print(f"Clark Error Grid: {ce}")

    # #Plots:
    # raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)
    # prediction = 0
    # for idx in range(10):  # plot 10 examples
    #     best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True,).savefig(f'./models/tft/transformer_model/predictions/graph{prediction}.png')
    #     prediction+= 1
    # interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")

    # interpretations = 0
    # interpretations_dict = best_tft.plot_interpretation(interpretation)
    # for graph in interpretations_dict:
    #     interpretations_dict[graph].savefig(f'./models/tft/transformer_model/interpretations/graph{interpretations}.png')
    #     interpretations+=1