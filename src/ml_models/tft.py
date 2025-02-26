import sys
import os
import warnings
import logging
import wandb

warnings.filterwarnings("ignore")
wandb.init(entity = 't1d_uofa', project='t1d_tft')

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
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

MAX_ENCODER_LENGTH = 12
MAX_PREDICTION_LENGTH = 6
TFT_RANGE = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
N_SPLIT = 5

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

#Reading data
df_path = "./data/normal/db_final.csv"
df = pd.read_csv(df_path) 
tft_data = transformer_data(df, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)

#Creating KFolds:
gkf = GroupKFold(n_splits=N_SPLIT)
groups = tft_data['PtID']

t_MAE,t_gMAE,t_RMSE,t_gRMSE,t_MAPE,t_MARD,t_gMARD = [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(tft_data, groups=groups)):
    train_val_df = tft_data.iloc[train_idx]
    train_groups, val_groups = train_test_split(train_val_df["PtID"].unique(), test_size=0.3, random_state=42)

    train_df = train_val_df[train_val_df["PtID"].isin(train_groups)]
    val_df = train_val_df[train_val_df["PtID"].isin(val_groups)]
    test_df = tft_data.iloc[test_idx]

    #Time Series Data Set creating for TFT
    static_reals = ["AgeAsOfEnrollDt", "Weight", "Height","HbA1c",'Gender']
    time_varying_known_reals = []
    time_varying_unknown_reals = ['Value', 'Scaled_Value']
    group_ids = ['PtID']


    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Value",
        group_ids=group_ids,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        categorical_encoders={"PtID": NaNLabelEncoder(add_nan=True)},
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx = True
        # add additional parameters as necessary
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict= True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers = 0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=320, num_workers = 0)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    hparams = {
        "learning_rate": 0.011876282198155926,
        "hidden_size": 126,
        "attention_head_size": 3,
        "dropout": 0.2714312556856915,
        "hidden_continuous_size": 34,
        "batch_size": 32,
        "gradient_clip_val": 0.041158135741519074,
        "optimizer": "Adam",
        "reduce_on_plateau_patience": 4
    }


    #WandB logger:
    wandb_logger = WandbLogger(log_model="all", project="t1d_tft")
    wandb_logger.log_hyperparams(hparams)

    #Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    tensorboard_logger = TensorBoardLogger(save_dir="./data/normal/predictions/tft")
    checkpoint_callback = ModelCheckpoint(dirpath="./data/normal/predictions/tft/checkpoints",
                                        filename="best_model",
                                        save_top_k=1,
                                        monitor="val_loss",
                                        mode="min",
    )

    #Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=hparams["gradient_clip_val"],
        # limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=[wandb_logger, tensorboard_logger]
    )

    #TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=hparams["learning_rate"],
        hidden_size=hparams["hidden_size"],
        attention_head_size=hparams["attention_head_size"],
        dropout=hparams["dropout"],
        hidden_continuous_size=hparams["hidden_continuous_size"],
        loss=MAE(),
        log_interval=10,
        optimizer=hparams["optimizer"],
        reduce_on_plateau_patience=hparams["reduce_on_plateau_patience"],
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    #Fit network
    try:
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    except Exception as e:
        wandb_logger.finalize(status="failed")  # Explicitly pass the status argument
        wandb.finish()  # Properly close WandB session
        raise e  # Reraise exception for debuggi

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
    # best_model_path = "./data/normal/predictions/tft/lightning_logs/version_9/checkpoints/epoch=9-step=65350.ckpt"

    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Calculate errors
    pt_num = 0
    grouped_test = test_df.groupby(['PtID'])
    pt_length = len(grouped_test)
    y_predictions = np.array([])
    y_actuals = np.array([])

    #looping over each patient:
    for ptid, group in grouped_test:
        test_ptid = test_df.loc[test_df['PtID'] == ptid]
        y_predictions_ptid = np.array([])
        y_actuals_ptid = np.array([])

        for i in range(test_ptid.shape[0] - TFT_RANGE):
            test_ptid_part = test_ptid.iloc[i : i + TFT_RANGE]
            testing = TimeSeriesDataSet.from_dataset(training, test_ptid_part, predict=True, stop_randomization= True)
            test_dataloader = testing.to_dataloader(train=False, batch_size=320)
            ptid_predictions = best_tft.predict(test_dataloader, return_y= True, trainer_kwargs={'logger':False})
            y_predictions_ptid = np.concatenate((y_predictions_ptid, np.array(ptid_predictions.output.T.tolist()[5])), axis=None)
            y_actuals_ptid = np.concatenate((y_actuals_ptid, np.array(ptid_predictions.y[0].T.tolist()[5])), axis=None)
        
        pt_num+=1

        y_predictions = np.concatenate((y_predictions, y_predictions_ptid), axis=None)
        y_actuals = np.concatenate((y_actuals,y_actuals_ptid), axis=None)
        print(f'Testing data gathered: {pt_num}/{pt_length}')

    i_MAE,i_gMAE,i_RMSE,i_gRMSE,i_MAPE,i_MARD,i_gMARD = find_confidence_errors_w_intervals(y_predictions,y_actuals)
    # ce = clark_error_perc(list(y_actuals),list(y_predictions))

    t_MAE[0] += i_MAE[0]
    t_MAE[1] += i_MAE[1]

    t_gMAE[0] += i_gMAE[0]
    t_gMAE[1] += i_gMAE[1]

    t_RMSE[0] += i_RMSE[0]
    t_RMSE[1] += i_RMSE[1]

    t_gRMSE[0] += i_gRMSE[0]
    t_gRMSE[1] += i_gRMSE[1]

    t_MAPE[0] += i_MAPE[0]
    t_MAPE[1] += i_MAPE[1]

    t_MARD[0] += i_MARD[0]
    t_MARD[1] += i_MARD[1]

    t_gMARD[0] += i_gMARD[0]
    t_gMARD[1] += i_gMARD[1]

#Metrics:
metrics = {
    'MAE': f'{t_MAE[0]/N_SPLIT} +- {t_MAE[1]/N_SPLIT}',
    't_gMAE': f'{t_gMAE[0]/N_SPLIT} +- {t_gMAE[1]/N_SPLIT}',
    't_RMSE': f'{t_RMSE[0]/N_SPLIT} +- {t_RMSE[1]/N_SPLIT}',
    't_gRMSE': f'{t_gRMSE[0]/N_SPLIT} +- {t_gRMSE[1]/N_SPLIT}',
    't_MAPE': f'{t_MAPE[0]/N_SPLIT} +- {t_MAPE[1]/N_SPLIT}',
    't_MARD': f'{t_MARD[0]/N_SPLIT} +- {t_MARD[1]/N_SPLIT}',
    't_gMARD': f'{t_gMARD[0]/N_SPLIT} +- {t_gMARD[1]/N_SPLIT}'
}
wandb.log(metrics)

metrics_df = pd.DataFrame(metrics,index=[0])
metrics_df.to_csv('./data/normal/predictions/tft/metrics.csv', index=False)


print(f"t_MAE: {t_MAE[0]/N_SPLIT} +- {t_MAE[1]/N_SPLIT}")
print(f"t_gMAE: {t_gMAE[0]/N_SPLIT} +- {t_gMAE[1]/N_SPLIT}")
print(f"t_RMSE: {t_RMSE[0]/N_SPLIT} +- {t_RMSE[1]/N_SPLIT}")
print(f"t_gRMSE: {t_gRMSE[0]/N_SPLIT} +- {t_gRMSE[1]/N_SPLIT}")
print(f"t_MAPE: {t_MAPE[0]/N_SPLIT} +- {t_MAPE[1]/N_SPLIT}")
print(f"t_MARD: {t_MARD[0]/N_SPLIT} +- {t_MARD[1]/N_SPLIT}")
print(f"t_gMARD: {t_gMARD[0]/N_SPLIT} +- {t_gMARD[1]/N_SPLIT}")
# print(f"Clark Error Grid: {ce}")

#Plots:
raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)
prediction = 0
best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True,).savefig(f'./data/normal/predictions/tft/graph{prediction}.png')

interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
interpretations = 0
interpretations_dict = best_tft.plot_interpretation(interpretation)
for graph in interpretations_dict:
    interpretations_dict[graph].savefig(f'./data/normal/predictions/tft/interpretation{interpretations}.png')
    interpretations+=1

