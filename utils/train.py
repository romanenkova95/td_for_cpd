"""Module for trainig Change Point Detection models."""

import pytorch_lightning as pl
import torch.nn as nn

from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from . import models_v2 as models

from .models import fix_seeds


def train_model(model: models.CPD_model,
                experiments_name: str,
                max_epochs: int = 100,
                patience: int = None,
                gpus: int = 0,
                gradient_clip_val: float = 0.0,
                seed: int = 0,
                monitor: str = "val_loss",
                min_delta: float = 0.0,
                check_val_every_n_epoch: int = 1) -> models.CPD_model:
    """Initialize logger, callbacks, trainer and TRAIN CPD model
    
    :param model: CPD_model for training
    :param experiments_name: name of the conducted experiment
    :param max_epochs: maximum # of epochs to train (default=100)
    :param patience: # of epochs to eait before early stopping (no early stopping if patience=None)
    :param gpus: # of available GPUs for trainer (default=0 -> train on CPU)
    :param gradient_clip_val: parameter for Gradient Clipping (if 0, no Gradient Clipping)
    
    :return: trained model
    """
    callbacks = []

    # initialize TensorBoard logger
    logger = TensorBoardLogger(save_dir='logs/', name=experiments_name)

    # initialize Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/{}'.format(experiments_name),
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}')
    callbacks.append(checkpoint_callback)

    if patience is not None:
        # initialize EarlyStopping callback
        early_stop_callback = EarlyStopping(monitor=monitor,
                                            min_delta=min_delta,
                                            patience=patience,
                                            verbose=True,
                                            mode="min")
        callbacks.append(early_stop_callback)

    # fixing all the seeds
    fix_seeds(seed)

    trainer = pl.Trainer(max_epochs=max_epochs,
                         gpus=gpus,
                         benchmark=True,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gradient_clip_val=gradient_clip_val,
                         logger=logger,
                         callbacks=callbacks)

    trainer.fit(model)

    return model