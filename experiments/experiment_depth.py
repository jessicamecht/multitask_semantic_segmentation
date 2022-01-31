#%%
import sys

sys.path.append("..")
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import pytorch_lightning as pl
from models.deeplab import Deeplab
from utils import *
from file_utils import *
import time


class DepthExperiment(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.fine_tune = self.hparams.fine_tune
        self.model = Deeplab(self.fine_tune, depth=True)
        self.loss = self.customl1loss
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.train_data = HypersimDataset(hparams.train_data)
        self.val_data = HypersimDataset(hparams.val_data)
        self.test_data = HypersimDataset(hparams.test_data)
        self.num_workers = hparams.num_workers

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, depth, _ = batch
        output = self(img)
        loss = self.loss(output, depth)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, _ = batch
        output = self(img)
        loss = self.loss(output, depth)
        rel = rel_err(output, depth, self.on_gpu)
        rms = rms_err(output, depth, self.on_gpu)
        log10 = log10_err(output, depth, self.on_gpu)
        return {"val_loss": loss, "val_rel": rel, "val_rms": rms, "val_log10": log10}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        rel = torch.stack([x["val_rel"] for x in outputs]).mean()
        rms = torch.stack([x["val_rms"] for x in outputs]).mean()
        log10 = torch.stack([x["val_log10"] for x in outputs]).mean()
        self.log("val_loss", avg_loss.item(), prog_bar=True)
        self.log("val_rel", rel.item(), prog_bar=True)
        self.log("val_rms", rms.item(), prog_bar=True)
        self.log("val_log10", log10.item(), prog_bar=True)

    def on_epoch_start(self):
        print("\n")

    def test_step(self, batch, batch_idx):
        img, depth, _ = batch
        output = self(img)
        loss = self.loss(output, depth)
        rel = rel_err(output, depth, self.on_gpu)
        rms = rms_err(output, depth, self.on_gpu)
        log10 = log10_err(output, depth, self.on_gpu)
        return {
            "test_loss": loss,
            "test_rel": rel,
            "test_rms": rms,
            "test_log10": log10,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        rel = torch.stack([x["test_rel"] for x in outputs]).mean()
        rms = torch.stack([x["test_rms"] for x in outputs]).mean()
        log10 = torch.stack([x["test_log10"] for x in outputs]).mean()
        self.log("test_loss", avg_loss.item())
        self.log("test_rel_err", rel.item())
        self.log("test_rms_err", rms.item())
        self.log("test_log10_err", log10.item())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optim

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return test_loader

    def customl1loss(self, output, target):
        mask = torch.isnan(target)
        return F.l1_loss(output[~mask], target[~mask])
