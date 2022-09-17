from dataclasses import dataclass
from typing import Any

from minigate.configs import get_module_import_path
from minigate.configs.trainer.base import BaseTrainer
from pytorch_lightning.plugins import DDPPlugin


@dataclass
class DDPPlugin:
    _target_: str = get_module_import_path(DDPPlugin)
    find_unused_parameters: bool = False


@dataclass
class DDPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    gpus: int = 1
    strategy: Any = None
    replace_sampler_ddp: bool = True
    sync_batchnorm: bool = True
    auto_scale_batch_size: bool = False
    plugins: Any = DDPPlugin()


@dataclass
class DPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: str = "dp"
    gpus: int = 1
    auto_scale_batch_size: bool = False
