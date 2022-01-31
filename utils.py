#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


def iou(pred, target, gpu=False):
    """

    Args:
        pred ([type]): [description]
        target ([type]): [description]

    Returns:
        [type]: [description]
    """
    # pred and target has the format of
    # N, H, W
    device = "cuda" if gpu else "cpu"
    n_class = 41

    ious = torch.zeros((2, n_class)).to(device)
    for i in range(1, n_class):
        predMask = pred == i
        targetMask = target == i
        intersection = (predMask & targetMask).sum()
        union = (predMask | targetMask).sum()
        # intersection of each class
        ious[0, i] += intersection.float()
        # union of each class
        ious[1, i] += union.float()
    return ious

    # index = torch.arange(1, n_class)[None].to(device)  # do not count class 0
    # return torch.stack(
    #     [
    #         ((pred[:, :, :, None] == index) & (target[:, :, :, None] == index)).sum(
    #             axis=(0, 1, 2)
    #         ),
    #         ((pred[:, :, :, None] == index) | (target[:, :, :, None] == index)).sum(
    #             axis=(0, 1, 2)
    #         ),
    #     ]
    # ).float()


def pixel_acc(pred, target):
    # res = (pred == target).to(dtype=torch.float).mean()
    # do not count class 0

    res = (pred[target != 0] == target[target != 0]).to(dtype=torch.float).mean()
    return res


def rel_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return (torch.abs(target - preds) / target).sum() / torch.numel(target)


def rms_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.sqrt(((target - preds) ** 2).sum() / torch.numel(target))


def log10_err(preds, target, gpu=False):
    device = "cuda" if gpu else "cpu"
    mask = torch.isnan(target)
    target, preds = target[~mask], preds[~mask]
    return torch.abs(torch.log10(target) - torch.log10(preds)).sum() / torch.numel(
        target
    )
