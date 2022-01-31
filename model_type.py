import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class ModelType(nn.Module):
    def __init__(self, loss_type):
        super(ModelType, self).__init__()
        self.loss_type = loss_type.lower()
        self.params = []

    def loss(self):
        raise NotImplementedError()

    def get_batch_data(self, output, targets):
        raise NotImplementedError()

    def get_epoch_data(self, outputs):
        raise NotImplementedError()


class Segmentation(ModelType):
    def loss(self, output, targets):
        if "entropy" in self.loss_type:
            return F.cross_entropy(output, targets[0])
        else:
            raise ValueError(f"UNKNOWN LOSS {self.loss_type}")

    def get_batch_data(self, output, targets, on_gpu):
        loss = self.loss(output, targets)
        semseg_out = output
        semseg, depth = targets
        # calculate semantic segmentation data
        pred = semseg_out.argmax(1)
        IoUs = iou(pred, semseg, on_gpu)
        acc = pixel_acc(pred, semseg)
        return {
            "loss": loss,
            "ious": IoUs,
            "acc": acc,
        }

    def get_epoch_data(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        IoUs = torch.stack([x["ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()
        return {
            "loss": loss,
            "mIoU": mIoU,
            "acc": acc,
        }


class Depth(ModelType):
    def loss(self, output, targets):
        if "l1" in self.loss_type:
            return customl1loss(output, targets[1])
        else:
            raise ValueError(f"UNKNOWN LOSS {self.loss_type}")

    def get_batch_data(self, output, targets, on_gpu):
        loss = self.loss(output, targets)
        depth_out = output
        semseg, depth = targets
        # calculate depth data
        rel = rel_err(depth_out, depth, on_gpu)
        rms = rms_err(depth_out, depth, on_gpu)
        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
        }

    def get_epoch_data(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        rel = torch.stack([x["rel_err"] for x in outputs]).mean()
        rms = torch.stack([x["rms_err"] for x in outputs]).mean()
        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
        }


class MultiTask(ModelType):
    def __init__(self, loss_type, alpha=0):
        super().__init__(loss_type)
        if "combine" in self.loss_type:
            self.params = nn.Parameter(torch.tensor([-1.0, 0.0]), requires_grad=True)
        elif "naive" in self.loss_type:
            self.alpha = alpha
        elif "add" in self.loss_type:
            self.train_seg = "seg"

    def loss(self, output, targets):
        if "combine" in self.loss_type:
            return combined_loss(
                output,
                targets,
                self.params,
            )
        elif "naive" in self.loss_type:
            return naive_multiloss(output, targets, self.alpha)
        elif "add" in self.loss_type:
            return add_loss(output, targets, self.train_seg)
        else:
            raise ValueError(f"UNKNOWN LOSS {self.loss_type}")

    def get_batch_data(self, outputs, targets, on_gpu):
        loss = self.loss(outputs, targets)
        semseg_out, depth_out = outputs
        semseg, depth = targets
        # calculate depth data
        rel = rel_err(depth_out, depth, on_gpu)
        rms = rms_err(depth_out, depth, on_gpu)
        # calculate semantic segmentation data
        pred = semseg_out.argmax(1)
        IoUs = iou(pred, semseg, on_gpu)
        acc = pixel_acc(pred, semseg)
        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
            "ious": IoUs,
            "acc": acc,
        }

    def get_epoch_data(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        rel = torch.stack([x["rel_err"] for x in outputs]).mean()
        rms = torch.stack([x["rms_err"] for x in outputs]).mean()
        IoUs = torch.stack([x["ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()

        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
            "mIoU": mIoU,
            "acc": acc,
        }

class MultiTaskRecon(ModelType):
    def __init__(self, loss_type, alpha=0):
        super().__init__(loss_type)
        if "combine" in self.loss_type:
            self.params = nn.Parameter(torch.tensor([-1.0, 0.0]), requires_grad=True)
        elif "naive" in self.loss_type:
            self.alpha = alpha
        elif "add" in self.loss_type:
            self.train_seg = "seg"

    def loss(self, output, targets):
        if "combine" in self.loss_type:
            return combined_loss(
                output,
                targets,
                self.params,
            )
        elif "naive" in self.loss_type:
            return naive_multiloss_recon(output, targets, self.alpha)
        elif "add" in self.loss_type:
            return add_loss(output, targets, self.train_seg)
        else:
            raise ValueError(f"UNKNOWN LOSS {self.loss_type}")

    def get_batch_data(self, outputs, targets, on_gpu):
        loss = self.loss(outputs, targets)
        semseg_out, depth_out, _ = outputs
        semseg, depth, _ = targets
        # calculate depth data
        rel = rel_err(depth_out, depth, on_gpu)
        rms = rms_err(depth_out, depth, on_gpu)
        # calculate semantic segmentation data
        pred = semseg_out.argmax(1)
        IoUs = iou(pred, semseg, on_gpu)
        acc = pixel_acc(pred, semseg)
        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
            "ious": IoUs,
            "acc": acc,
        }

    def get_epoch_data(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        rel = torch.stack([x["rel_err"] for x in outputs]).mean()
        rms = torch.stack([x["rms_err"] for x in outputs]).mean()
        IoUs = torch.stack([x["ious"] for x in outputs], dim=0).sum(dim=0)
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        mIoU = IoUs[0].sum() / IoUs[1].sum()

        return {
            "loss": loss,
            "rel_err": rel,
            "rms_err": rms,
            "mIoU": mIoU,
            "acc": acc,
        }


def naive_multiloss_recon(outputs, targets, alpha):
    # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    seg_out, depth_out, recon_out = outputs
    # print(len(targets))

    semseg, depth, recon = targets
    seg_loss, depth_loss = F.cross_entropy(seg_out, semseg), customl1loss(
        depth_out, depth
    )
    recon_loss = mseloss(recon, recon_out)
    return alpha * seg_loss + (1 - alpha)//2 * depth_loss + (1 - alpha)//2 * recon_loss


def mseloss(output, target):
    return F.mse_loss(output, target)

def customl1loss(output, target):
    mask = torch.isnan(target)
    return F.l1_loss(output[~mask], target[~mask])


def naive_multiloss(outputs, targets, alpha):
    seg_out, depth_out = outputs
    semseg, depth = targets
    seg_loss, depth_loss = F.cross_entropy(seg_out, semseg), customl1loss(
        depth_out, depth
    )
    return alpha * seg_loss + (1 - alpha) * depth_loss


def add_loss(outputs, targets, train_seg=None):
    seg_out, depth_out = outputs
    semseg, depth = targets
    seg_loss, depth_loss = F.cross_entropy(seg_out, semseg), customl1loss(
        depth_out, depth
    )
    loss = 0
    if train_seg == "seg":
        loss += seg_loss
        with torch.no_grad():
            loss += depth_loss
    elif train_seg == "depth":
        loss += depth_loss
        with torch.no_grad():
            loss += seg_loss
    else:
        loss += seg_loss + depth_loss
    return loss


def combined_loss(outputs, targets, params):
    seg_out, depth_out = outputs
    semseg, depth = targets
    seg_loss, depth_loss = F.cross_entropy(seg_out, semseg), customl1loss(
        depth_out, depth
    )

    losses, log_vars = (seg_loss, depth_loss), params
    assert len(losses) == len(log_vars)
    loss = 0
    for i in range(len(losses)):
        p = torch.exp(-log_vars[i])
        loss += p * losses[i] + log_vars[i]

    return loss
