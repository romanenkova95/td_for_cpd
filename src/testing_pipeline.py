from typing import List, Optional

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import os

try:
    from pytorch_lightning.loggers import Logger
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase

    Logger = LightningLoggerBase

from src import utils
log = utils.get_logger(__name__)

def test(config: DictConfig) -> Optional[float]:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(
        f"Instantiating datamodule <{config.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule
    )

    datamodule.setup()


    device = torch.device(
        f"cuda:{config.trainer.gpus[0]}" if torch.cuda.is_available() else "cpu"
    )

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
    ).to(device)

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, _convert_="partial"
    )

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    # Make sure everything closed properly
    log.info("Finalizing!")
