from typing import Optional, Union

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DictConfig,
        data_loader_config: DictConfig,
    ):
        super(DataModule, self).__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.dataset_root = dataset_config.dataset_root
        self.seed = data_loader_config.seed
        self.dataset_config = dataset_config
        self.data_loader_config = data_loader_config

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def dummy_batch(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
