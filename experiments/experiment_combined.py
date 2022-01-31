#%%
import sys

sys.path.append("..")
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import pytorch_lightning as pl
from models.deeplab_multitask import DeeplabMultiTask
from utils import *
from file_utils import *
import time


class MultiTaskExperiment(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.fine_tune = self.hparams.fine_tune
        self.fine_tune_all = self.hparams.fine_tune_all
        self.model = DeeplabMultiTask(
            fine_tune=self.fine_tune, depth=True, fine_tune_all=self.fine_tune_all
        )
        self.seg_loss = nn.CrossEntropyLoss()
        self.depth_loss = self.customl1loss
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.train_data = HypersimDataset(hparams.train_data)
        self.val_data = HypersimDataset(hparams.val_data)
        self.test_data = HypersimDataset(hparams.test_data)
        self.num_workers = hparams.num_workers

        self.log_var_seg = torch.zeros((1,), requires_grad=True)
        self.log_var_depth = torch.zeros((1,), requires_grad=True)

    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        img, depth, semseg = batch
        seg_out, depth_out = self(img)
        seg_loss, depth_loss = self.seg_loss(seg_out, semseg), self.depth_loss(
            depth_out, depth
        )
        loss = self.combined_loss(
            (seg_loss, depth_loss),
            (
                self.log_var_seg.type_as(seg_loss),
                self.log_var_depth.type_as(depth_loss),
            ),
        )
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, semseg = batch
        seg_out, depth_out = self(img)
        seg_out, depth_out = self(img)
        seg_loss, depth_loss = self.seg_loss(seg_out, semseg), self.depth_loss(
            depth_out, depth
        )
        loss = self.combined_loss(
            (seg_loss, depth_loss),
            (
                self.log_var_seg.type_as(seg_loss),
                self.log_var_depth.type_as(depth_loss),
            ),
        )

        pred = seg_out.argmax(1)
        IoUs = iou(pred, semseg, self.on_gpu)
        acc = pixel_acc(pred, semseg)
        rel = rel_err(depth_out, depth, self.on_gpu)
        rms = rms_err(depth_out, depth, self.on_gpu)
        log10 = log10_err(depth_out, depth, self.on_gpu)

        return {
            "val_loss": loss,
            "val_rel": rel,
            "val_rms": rms,
            "val_log10": log10,
            "val_ious": IoUs,
            "val_acc": acc,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        rel = torch.stack([x["val_rel"] for x in outputs]).mean()
        rms = torch.stack([x["val_rms"] for x in outputs]).mean()
        log10 = torch.stack([x["val_log10"] for x in outputs]).mean()
        IoUs = torch.stack([x["val_ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()

        self.log("val_loss", avg_loss.item(), prog_bar=True)
        self.log("val_rel", rel.item(), prog_bar=True)
        self.log("val_mIoU", mIoU.item(), prog_bar=True)
        self.log("val_rms", rms.item(), prog_bar=True)
        self.log("val_acc", acc.item(), prog_bar=True)
        self.log("val_log10", log10.item(), prog_bar=True)

    def on_epoch_start(self):
        print("\n")

    def test_step(self, batch, batch_idx):
        img, depth, semseg = batch
        seg_out, depth_out = self(img)
        seg_loss, depth_loss = self.seg_loss(seg_out, semseg), self.depth_loss(
            depth_out, depth
        )
        loss = self.combined_loss(
            (seg_loss, depth_loss),
            (
                self.log_var_seg.type_as(seg_loss),
                self.log_var_depth.type_as(depth_loss),
            ),
        )

        pred = seg_out.argmax(1)
        IoUs = iou(pred, semseg, self.on_gpu)
        acc = pixel_acc(pred, semseg)
        rel = rel_err(depth_out, depth, self.on_gpu)
        rms = rms_err(depth_out, depth, self.on_gpu)
        log10 = log10_err(depth_out, depth, self.on_gpu)
        return {
            "test_loss": loss,
            "test_rel": rel,
            "test_rms": rms,
            "test_log10": log10,
            "test_ious": IoUs,
            "test_acc": acc,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        rel = torch.stack([x["test_rel"] for x in outputs]).mean()
        rms = torch.stack([x["test_rms"] for x in outputs]).mean()
        log10 = torch.stack([x["test_log10"] for x in outputs]).mean()
        IoUs = torch.stack([x["test_ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()

        self.log("test_loss", avg_loss.item(), prog_bar=True)
        self.log("test_rel", rel.item(), prog_bar=True)
        self.log("test_mIoU", mIoU.item(), prog_bar=True)
        self.log("test_rms", rms.item(), prog_bar=True)
        self.log("test_acc", acc.item(), prog_bar=True)
        self.log("test_log10", log10.item(), prog_bar=True)

    def configure_optimizers(self):
        params = (
            [p for p in self.model.parameters()]
            + [self.log_var_seg]
            + [self.log_var_depth]
        )
        optim = torch.optim.Adam(params, lr=self.lr)
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

    def combined_loss(self, losses, log_vars):
        assert len(losses) == len(log_vars)
        loss = 0
        for i in range(len(losses)):
            p = torch.exp(-log_vars[i])
            loss += p * losses[i] + log_vars[i]

        return loss
