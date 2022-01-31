#%%
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from models.deeplab import Deeplab
from utils import *
from file_utils import *
import sys
import time
from factory import select_model
from experiment import *
import os

#%%
exp_name = sys.argv[1] if len(sys.argv) > 1 else "default"
#%%
config_dict = read_file(exp_name)
config_dict

#%%
experiment_folder = (
    "experiment_nyu"
    if "nyu" in config_dict.get("dataset", "hypersim")
    else "experiment_data"
)
folder_path = f"./{experiment_folder}/{config_dict['experiment_name']}/v_{config_dict['version_number']}"
path = folder_path + "/checkpoints"
checkpoint = sorted(
    [file for file in os.listdir(path) if "ckpt" in file], reverse=True
)[0]
checkpoint_path = os.path.join(path, checkpoint)
print(f"RESUMING FROM {checkpoint_path}")


#%%
hparams = dict_to_args(config_dict)
#%%
print("-" * 80)
print("RUNNING FOLLOWING EXPERIMENT")
for key, value in config_dict.items():
    print(f"\t{key}: {value}")
print("-" * 80)
#%%
exp = Experiment.load_from_checkpoint(checkpoint_path, hparams=hparams)
trainer = pl.Trainer(
    resume_from_checkpoint=checkpoint_path,
    gpus=1,
    max_epochs=hparams.num_epochs,
    progress_bar_refresh_rate=False,
    logger=False,
)
#%%
start = time.time()
trainer.test(exp)
print(f"Finish in {time.time() - start} seconds")
#%%