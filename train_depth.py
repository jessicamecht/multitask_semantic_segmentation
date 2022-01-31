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
from experiments.experiment_depth_dl_aug import *


# from IPython import get_ipython

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")


#%%
exp_name = "default_depth_dl_aug"
#%%
if len(sys.argv) > 1:
    exp_name = sys.argv[1]
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
exp = DepthExperiment(hparams)
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
