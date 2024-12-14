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
MAX_ENCODER_LENGTH = 12
MAX_PREDICTION_LENGTH = 6
TFT_RANGE = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
N_SPLIT = 5

#Reading data
df_path = "./data/normal/db_final.csv"
df = pd.read_csv(df_path) 
tft_data = transformer_data(df, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
tft_data.to_csv('./data/normal/db_tft.csv', index=False)

#Creating KFolds:
gkf = GroupKFold(n_splits=N_SPLIT)
groups = tft_data['PtID']

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(tft_data, groups=groups)):
    train_val_df = tft_data.iloc[train_idx]
    train_groups, val_groups = train_test_split(train_val_df["PtID"].unique(), test_size=0.3, random_state=42)

    train_df = train_val_df[train_val_df["PtID"].isin(train_groups)]
    val_df = train_val_df[train_val_df["PtID"].isin(val_groups)]
    test_df = tft_data.iloc[test_idx]

    train_df.to_csv('./data/db_train.csv', index=False)
    val_df.to_csv('./data/db_val.csv', index=False)
    test_df.to_csv('./data/db_test.csv', index=False)


    #Time Series Data Set creating for TFT
    static_reals = ["AgeAsOfEnrollDt", "Weight", "Height","HbA1c",'Gender']
    time_varying_known_reals = []
    time_varying_unknown_reals = ['Value', 'Scaled_Value']


    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Value",
        group_ids=["PtID"],
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
    
    print(f"Validation Dataset Length: {len(validation)}")
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("./data/normal/predictions/tft")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.041158135741519074,
        # limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback],
        logger= logger
)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.011876282198155926,
        hidden_size=126,
        attention_head_size=3,
        dropout=0.2714312556856915,
        hidden_continuous_size=34,
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

    # Calculate errors
    grouped_test_data = test_df.groupby(["PtID"])
    y_predictions = np.array([])
    y_actuals = np.array([])

    #loop over each patient-encounter
    for ptid, patient_data in grouped_test_data:
        index = 0
        y_prediction = np.array([])
        y_actual = np.array([])

        #loop for each patient
        for i in range(patient_data.shape[0] - TFT_RANGE):
            test_df_part = patient_data.iloc[index:index+TFT_RANGE]
            testing = TimeSeriesDataSet.from_dataset(training, test_df_part, predict= True, stop_randomization=True)
            test_dataloader = testing.to_dataloader(train=False, batch_size=320, num_workers = 0)
            predictions = best_tft.predict(test_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
            y_prediction = np.concatenate((y_prediction,np.array(predictions.output.T.tolist()[0])))
            y_actual = np.concatenate((y_actual,np.array(predictions.y[0].T.tolist()[0])))
            index += 1

        y_predictions = np.concatenate((y_predictions,y_prediction))
        y_actuals = np.concatenate((y_actuals,y_actual))

    print(y_predictions)
    print(y_actuals)

    # i_MAE,i_gMAE,i_RMSE,i_gRMSE,i_MAPE,i_MARD,i_gMARD = find_confidence_errors_w_intervals(np.array(list(y_predictions)),np.array(list(y_actuals)))
    # ce = clark_error_perc(list(y_actuals),list(y_predictions))
    # print(f"i_MAE: {i_MAE[0]} +- {i_MAE[1]}")
    # print(f"i_gMAE: {i_gMAE[0]} +- {i_gMAE[1]}")
    # print(f"i_RMSE: {i_RMSE[0]} +- {i_RMSE[1]}")
    # print(f"i_gRMSE: {i_gRMSE[0]} +- {i_gRMSE[1]}")
    # print(f"i_MAPE: {i_MAPE[0]} +- {i_MAPE[1]}")
    # print(f"i_MARD: {i_MARD[0]} +- {i_MARD[1]}")
    # print(f"i_gMARD: {i_gMARD[0]} +- {i_gMARD[1]}")
    # print(f"Clark Error Grid: {ce}")

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