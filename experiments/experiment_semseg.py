#%%
import sys 
sys.path.append('..')
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



class SegmentationExperiment(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.fine_tune = self.hparams.fine_tune
        self.model = Deeplab(self.fine_tune)
        self.loss = nn.CrossEntropyLoss()
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.train_data = HypersimDataset(hparams.train_data)
        self.val_data = HypersimDataset(hparams.val_data)
        self.test_data = HypersimDataset(hparams.test_data)
        self.num_workers = hparams.num_workers

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, depth, semseg = batch
        output = self(img)
        loss = self.loss(output, semseg)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, semseg = batch
        output = self(img)
        pred = output.argmax(1)
        loss = self.loss(output, semseg)
        IoUs = iou(pred, semseg, self.on_gpu)
        acc = pixel_acc(pred, semseg)
        return {"val_loss": loss, "val_ious": IoUs, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        IoUs = torch.stack([x["val_ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()
        self.log("val_loss", avg_loss.item(), prog_bar=True)
        self.log("val_mIoU", mIoU.item(), prog_bar=True)
        self.log("val_acc", acc.item(), prog_bar=True)

    def on_epoch_start(self):
        print("\n")

    def test_step(self, batch, batch_idx):
        img, depth, semseg = batch
        output = self(img)
        pred = output.argmax(1)
        loss = self.loss(output, semseg)
        IoUs = iou(pred, semseg, self.on_gpu)
        acc = pixel_acc(pred, semseg)
        return {"test_loss": loss, "test_ious": IoUs, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        IoUs = torch.stack([x["test_ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()
        self.log("test_loss", avg_loss.item())
        self.log("test_mIoU", mIoU.item())
        self.log("test_acc", acc.item())

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

