#%%
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from utils import *
from file_utils import *
import sys
import time
from experiments.experiment_unet_multitask import *


# from IPython import get_ipython

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")

#%%
exp_name = "experiment_unet_multitask"
#%%
if len(sys.argv) > 1:
    exp_name = sys.argv[1]
#%%
train_data = HypersimDataset("./data/train_data.csv")
val_data = HypersimDataset("./data/val_data.csv")
test_data = HypersimDataset("./data/test_data.csv")
#%%
print(f"train size: {len(train_data)}")
print(f"val size: {len(val_data)}")
print(f"test size: {len(test_data)}")
#%%
config_dict = read_file(exp_name)
config_dict
#%%

#%%
tb_logger = pl.loggers.TensorBoardLogger(
    save_dir="./",
    name=f'{str(config_dict["experiment_name"])}',  # This will create different subfolders for your models
    version=f'v_{str(config_dict["version_number"])}',
)
#%%
hparams = dict_to_args(config_dict)
#%%
exp = UNet_Multitask_lightning(hparams, 3, 40 + 1)
trainer = pl.Trainer(
    fast_dev_run=False,
    gpus=1,
    max_epochs=hparams.num_epochs,
    progress_bar_refresh_rate=5,
    logger=tb_logger,
)
#%%
trainer.fit(exp)
trainer.test(exp)
#%%