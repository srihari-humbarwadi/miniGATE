import os
import pathlib
from typing import List, Optional

import hydra
import pytorch_lightning
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner
from rich.traceback import install
from wandb.util import generate_id

from minigate.base.utils.loggers import get_logger
from minigate.datamodules.base import DataModule
from minigate.train_eval_agents.base import TrainingEvaluationAgent

log = get_logger(__name__)

install(show_locals=False, word_wrap=True, width=350)


def checkpoint_setup(config):
    checkpoint_path = None

    if config.resume:

        log.info("Continue from existing checkpoint")

        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

        checkpoint_path = f"{config.current_experiment_dir}/checkpoints/last.ckpt"

        if not pathlib.Path(checkpoint_path).exists():
            checkpoint_path = None

        log.info(checkpoint_path)

    else:

        log.info("Starting from scratch")
        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

    return checkpoint_path


def debug(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    log.info("Starting debug mode")
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # --------------------------------------------------------------------------------
    # Create or recover checkpoint path to resume from
    checkpoint_path = checkpoint_setup(config)
    # --------------------------------------------------------------------------------
    # Instantiate Lightning DataModule for task
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    # log information regarding data module to be instantiated -- particularly the class name that is stored in _target_
    datamodule: DataModule = instantiate(config.datamodule, _recursive_=False)
    # List in comments all possible datamodules/datamodule configs
    datamodule.setup(stage="fit")
    # # datamodule_pretty_dict_tree = generate_config_tree(
    # #     config=datamodule.__dict__, resolve=True
    # # )
    # log.info(
    #     f"Datamodule <{config.datamodule._target_}> instantiated, "
    #     f"with attributes {datamodule.__dict__}"
    # )
    # --------------------------------------------------------------------------------
