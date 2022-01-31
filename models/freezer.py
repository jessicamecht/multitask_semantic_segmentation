# Selectively Yanked from : https://gist.github.com/jbschiratti/e93f1ff9cc518a93769101044160d64d
import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from itertools import chain

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)


def filter_params(module, train_bn=True):
    """Yield the trainable parameters of a given module.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
    Returns
    -------
    generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            filter_params(module=child, train_bn=train_bn)


class UnfreezeCallback(Callback):
    """Unfreeze feature extractor callback."""

    def on_epoch_start(self, trainer, pl_module):
        print(trainer.current_epoch, pl_module.hparams.epoch_milestones[0])
        # print(pl_module.hparams)
        if str(trainer.current_epoch) in pl_module.hparams.epoch_milestones.split(", "):

            model = trainer.get_model()
            optimizer = trainer.optimizers[0]
            _current_lr = optimizer.param_groups[0]["lr"]
            if model.train_seg:
                print("\n----Training Seg-----\n")
                _make_trainable(model.model.decoder.seg_classifier)

                optimizer.add_param_group(
                    {
                        "params": filter_params(
                            module=model.model.decoder.seg_classifier
                        ),
                        "lr": _current_lr,
                    }
                )
                freeze(model.model.decoder.depth_classifier)
                model.train_seg = False
            else:
                print("----Training Depth----")
                _make_trainable(model.model.decoder.depth_classifier)

                optimizer.add_param_group(
                    {
                        "params": filter_params(
                            module=model.model.decoder.depth_classifier
                        ),
                        "lr": _current_lr,
                    }
                )
                freeze(model.model.decoder.seg_classifier)
                model.train_seg = True


class UnfreezeCallbackOneEpoch(Callback):
    """Unfreeze feature extractor callback."""

    def on_epoch_start(self, trainer, pl_module):
        model = trainer.get_model()
        optimizer = trainer.optimizers[0]
        _current_lr = optimizer.param_groups[0]["lr"]

        if "add" in model.model_type.loss_type:
            model.model_type.train_seg = "seg" if model.train_seg else "depth"
        if model.train_seg:
            print("----Training Seg-----")
            # freeze(model.model.decoder.depth_feat)
            freeze(model.model.decoder.depth_classifier)
            freeze(model.model.decoder.seg_to_dep)
            freeze(model.model.decoder.shared_to_dep)

            # _make_trainable(model.model.decoder.seg_feat)
            # REFER TO theta_dep
            _make_trainable(model.model.decoder.seg_classifier)
            # REFER TO theta_d-2
            _make_trainable(model.model.decoder.dep_to_seg)
            # REFER TO theta 3d
            _make_trainable(model.model.decoder.shared_to_seg)

            optimizer.add_param_group(
                {
                    "params": chain(
                        # filter_params(module=model.model.decoder.seg_feat),
                        filter_params(module=model.model.decoder.seg_classifier),
                        filter_params(module=model.model.decoder.dep_to_seg),
                        filter_params(module=model.model.decoder.shared_to_seg),
                    ),
                    "lr": _current_lr,
                }
            )
            model.train_seg = False
        else:
            print("----Training Depth----")
            # freeze(model.model.decoder.seg_feat)
            freeze(model.model.decoder.seg_classifier)
            freeze(model.model.decoder.dep_to_seg)
            freeze(model.model.decoder.shared_to_seg)

            # _make_trainable(model.model.decoder.depth_feat)
            _make_trainable(model.model.decoder.depth_classifier)
            _make_trainable(model.model.decoder.seg_to_dep)
            _make_trainable(model.model.decoder.shared_to_dep)

            optimizer.add_param_group(
                {
                    "params": chain(
                        # filter_params(module=model.model.decoder.depth_feat),
                        filter_params(module=model.model.decoder.depth_classifier),
                        filter_params(module=model.model.decoder.seg_to_dep),
                        filter_params(module=model.model.decoder.shared_to_dep),
                    ),
                    "lr": _current_lr,
                }
            )
            model.train_seg = True
