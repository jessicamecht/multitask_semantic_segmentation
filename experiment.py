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
from factory import *
from model_type import *


#%%
class Experiment(pl.LightningModule):
    def __init__(self, hparams, verbose=True):
        super().__init__()
        self.hparams = hparams
        self.fine_tune = self.hparams.fine_tune
        self.model = select_model(hparams)
        self.model_type = select_model_type(hparams)
        self.dataset = self.hparams.dataset if "dataset" in self.hparams else "hypersim"
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.verbose = verbose

        if self.hparams.get("epoch_milestones"):
            self.epoch_milestones = hparams.epoch_milestones
            self.train_seg = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        # print(self.model_type.loss)
        if "recon" in self.hparams.model_type:
            targets = (semseg, depth, img)
            loss = self.model_type.loss(output, targets)
        else:
            loss = self.model_type.loss(output, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.verbose and batch_idx % 20 == 0:
            print(
                f"EPOCH {self.current_epoch}   iter:{batch_idx}   loss: {loss.item()}"
            )
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        if "recon" in self.hparams.model_type:
            targets = (semseg, depth, img)
        data = self.model_type.get_batch_data(output, targets, self.on_gpu)
        return data

    def validation_epoch_end(self, outputs):
        data = self.model_type.get_epoch_data(outputs)
        data = {f"val_{key}": value for key, value in data.items()}
        for key, value in data.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if "combine" in self.hparams.loss_type:
            params = self.model_type.params
            self.log("log_var_seg", params[0], sync_dist=True)
            self.log("log_var_depth", params[1], sync_dist=True)

            if self.current_epoch == self.hparams.fix_epoch:
                self.model_type.params = torch.nn.Parameter(
                    self.model_type.params.detach()
                )

        if self.verbose:
            print("-" * 80)
            print(f"EPOCH {self.current_epoch} VALIDATION RESULT")
            for key, value in data.items():
                print(f"\t{key}: {value}")

            print("-" * 80)

    def test_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        data = self.model_type.get_batch_data(output, targets, self.on_gpu)
        return data

    def test_epoch_end(self, outputs):
        data = self.model_type.get_epoch_data(outputs)
        data = {f"test_{key}": value for key, value in data.items()}
        for key, value in data.items():
            self.log(key, value, sync_dist=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return optim

    def train_dataloader(self):
        return select_dataloader(self.hparams, "train")

    def val_dataloader(self):
        return select_dataloader(self.hparams, "validation")

    def test_dataloader(self):
        return select_dataloader(self.hparams, "test")
