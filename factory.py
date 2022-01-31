#%%
from models import (
    deeplab,
    deeplab_multitask,
    deeplab_multitask_single_decoder,
    deeplab_multitask_single_decoder_deeper,
    deeplab_multitask_single_decoder_shallow,
)
from unet import multitask_unet
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from model_type import *
from dataloader import *

# from NYU_dataloader import *
from torch.utils.data import Dataset, DataLoader


def select_model(hparams):
    config = vars(hparams)
    model_type = config.get("model_type")
    model = config.get("model")
    fine_tune = config.get("fine_tune", False)
    fine_tune_decoder = config.get("fine_tune_decoder", False)
    fine_tune_all = config.get("fine_tune_all", False)
    transpose = config.get("transpose", False)
    reconstruction = config.get("reconstruction", False)

    resnext = config.get("resnext", False)
    if "seg" in model_type:
        if model == "deeplab":
            return deeplab.Deeplab(fine_tune=fine_tune, resnext=resnext)
    elif "depth" in model_type:
        if model == "deeplab":
            return deeplab.Deeplab(fine_tune=fine_tune, depth=True, resnext=resnext)
    elif "multi" in model_type:
        if model == "deeplab":
            return deeplab_multitask.DeeplabMultiTask(
                fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
            )
        elif model == "unet":
            return multitask_unet.UNet_Multitask()
        elif model == "deeplab_single_decoder":
            return deeplab_multitask_single_decoder.DeeplabMultiTaskSingleDecoder(
                fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
            )
        elif model == "deeplab_single_decoder_deeper":
            return (
                deeplab_multitask_single_decoder_deeper.DeeplabMultiTaskSingleDecoder(
                    fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
                )
            )
        elif model == "deeplab_shared_decoder_deeper":
            return (
                deeplab_multitask_single_decoder_deeper.DeeplabMultiTaskSharedDecoder(
                    fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
                )
            )
        elif model == "deeplab_shared_decoder_shallow":
            return (
                deeplab_multitask_single_decoder_shallow.DeeplabMultiTaskSharedDecoder(
                    fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
                )
            )
        elif model == "deeplab_shared_decoder":
            return deeplab_multitask_single_decoder.DeeplabMultiTaskSharedDecoder(
                fine_tune=fine_tune,
                fine_tune_all=fine_tune_all,
                resnext=resnext,
                fine_tune_decoder=fine_tune_decoder,
                transpose=transpose,
                reconstruction=reconstruction,
            )
    elif "recon" in model_type:
        if model == "deeplab_shared_decoder":
            return deeplab_multitask_single_decoder.DeeplabMultiTaskSharedDecoder(
                fine_tune=fine_tune,
                fine_tune_all=fine_tune_all,
                resnext=resnext,
                fine_tune_decoder=fine_tune_decoder,
                transpose=transpose,
                reconstruction=reconstruction,
            )
    else:
        raise ValueError(f"Unknown model type {model_type} {model}")


def select_model_type(hparams):
    config = vars(hparams)
    model_type = hparams.model_type
    loss_type = hparams.loss_type
    alpha = config.get("naive_alpha", 0)
    if "seg" in model_type:
        return Segmentation(loss_type)
    elif "depth" in model_type:
        return Depth(loss_type)
    elif "multi" in model_type:
        return MultiTask(loss_type, alpha)
    elif "recon" in model_type:
        return MultiTaskRecon(loss_type, alpha)


def select_dataset(hparams):
    dataset = hparams.dataset if "dataset" in hparams else "hypersim"
    resolution = (hparams.img_size_h, hparams.img_size_w)
    use_transform = hparams.get("use_transform", False)
    if "hypersim" in dataset:
        return (
            HypersimDataset(
                hparams.train_data, out_size=resolution, use_transform=use_transform
            ),
            HypersimDataset(hparams.val_data, out_size=resolution, use_transform=False),
            HypersimDataset(
                hparams.test_data, out_size=resolution, use_transform=False
            ),
        )
    elif "nyu" in dataset:
        return (
            NYUDataset(
                dataset_type="train", out_size=resolution, use_transform=use_transform
            ),
            NYUDataset(dataset_type="test", out_size=resolution, use_transform=False),
            NYUDataset(dataset_type="test", out_size=resolution, use_transform=False),
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def select_dataloader(hparams, loader_type="train"):
    dataset = hparams.dataset if "dataset" in hparams else "hypersim"
    batch_size = hparams.batch_size
    num_workers = hparams.num_workers
    train_data, val_data, test_data = select_dataset(hparams)

    if loader_type.lower() == "train":
        return DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    elif loader_type.lower() in ("validation", "val"):
        return DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    elif loader_type.lower() == "test":
        return DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
