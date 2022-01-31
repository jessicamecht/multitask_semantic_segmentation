#%%
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from models.deeplab import Deeplab
from models.freezer import *
from utils import *
from file_utils import *
import sys
import time
from factory import select_dataloader
from experiment import *
from pytorch_lightning.callbacks import ModelCheckpoint
import os


#%%
if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    resume_from_checkpoint = bool(sys.argv[2] == "1") if len(sys.argv) > 2 else None
    fast_dev_run = int(sys.argv[3]) if len(sys.argv) > 3 else False
    #%%
    config_dict = read_file(exp_name)
    hparams = dict_to_args(config_dict)
    #%%
    experiment_folder = (
        "experiment_nyu"
        if "nyu" in config_dict.get("dataset", "hypersim")
        else "experiment_data"
    )
    checkpoint_path = hparams.checkpoint_path if "checkpoint_path" in hparams else False
    if resume_from_checkpoint and not checkpoint_path:
        path = f"./{experiment_folder}/{config_dict['experiment_name']}/v_{config_dict['version_number']}/checkpoints"
        try:
            checkpoint = sorted(
                [file for file in os.listdir(path) if "ckpt" in file],
                key=lambda s: int(s.split("=")[1].split("-")[0]),
                reverse=True,
            )[0]
            checkpoint_path = os.path.join(path, checkpoint)
            print(f"RESUMING FROM {checkpoint_path}")
        except:
            pass

    #%%

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=f"./{experiment_folder}",
        name=f'{str(config_dict["experiment_name"])}',  # This will create different subfolders for your models
        version=f'v_{str(config_dict["version_number"])}',
        default_hp_metric=False,
    )
    #%%
    print("-" * 80)
    print("RUNNING FOLLOWING EXPERIMENT")
    for key, value in config_dict.items():
        print(f"\t{key}: {value}")
    print("-" * 80)
    #%%
    if resume_from_checkpoint or checkpoint_path:
        exp = Experiment.load_from_checkpoint(checkpoint_path, hparams=hparams)
    else:
        exp = Experiment(hparams)

    if config_dict.get("epoch_milestones"):
        print("Using Callback")
        if hparams.epoch_milestones == "1":
            trainer = pl.Trainer(
                fast_dev_run=fast_dev_run,
                resume_from_checkpoint=checkpoint_path,
                gpus=1,
                max_epochs=hparams.num_epochs,
                progress_bar_refresh_rate=False,
                logger=tb_logger,
                callbacks=UnfreezeCallbackOneEpoch(),
            )

        else:
            trainer = pl.Trainer(
                fast_dev_run=fast_dev_run,
                resume_from_checkpoint=checkpoint_path,
                gpus=1,
                max_epochs=hparams.num_epochs,
                progress_bar_refresh_rate=False,
                logger=tb_logger,
                callbacks=UnfreezeCallback(),
            )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mIoU" if hparams.model_type != "depth" else "val_rel_err",
            mode="max" if hparams.model_type != "depth" else "min",
        )
        trainer = pl.Trainer(
            fast_dev_run=fast_dev_run,
            resume_from_checkpoint=checkpoint_path,
            gpus=1,
            max_epochs=hparams.num_epochs,
            progress_bar_refresh_rate=False,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
        )
    trainer.fit(exp)
    if not fast_dev_run:
        trainer.test()
#%%