from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import pytorch_lightning

try:
    from pytorch_lightning.loggers import Logger
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase

    Logger = LightningLoggerBase

from src import utils
log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best weights achieved during training.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    device = torch.device(
        f"cuda:{config.trainer.gpus[0]}" if torch.cuda.is_available() else "cpu"
    )

    # Init lightning datamodule
    log.info(
        f"Instantiating datamodule <{config.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule
    )

    datamodule.setup()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
    ).to(device)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model, datamodule)

    ckpt_path = Path(config.original_work_dir) / "final.pth"
    torch.save(model.state_dict(), ckpt_path)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train"):  # or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    return config.original_work_dir
