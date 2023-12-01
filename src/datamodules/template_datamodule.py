from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)
from src.datamodules.components.rotation_dataloader import RotationDataset, collate


class RotationDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        file_list_train: str,
        file_list_val: str = None,
        file_list_test: str = None,
        max_frames: int = None,
        smooth_output: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = RotationDataset(
                self.hparams.file_list_train,
                max_frames=self.hparams.max_frames,
                smooth_output=self.hparams.smooth_output,
            )
            if self.hparams.file_list_val:
                self.data_val = RotationDataset(
                    self.hparams.file_list_val,
                    max_frames=self.hparams.max_frames,
                    smooth_output=self.hparams.smooth_output,
                )
            if self.hparams.file_list_test:
                self.data_test = RotationDataset(
                    self.hparams.file_list_test,
                    max_frames=self.hparams.max_frames,
                    smooth_output=self.hparams.smooth_output,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                collate_fn=collate,
                shuffle=False,
            )
        else:
            return None

    def test_dataloader(self):
        if self.data_test:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                collate_fn=collate,
                shuffle=False,
            )
        else:
            return None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "rotation_datamodule.yaml")
    # cfg.data_dir = str(root / "data")
    print(cfg)
    data = hydra.utils.instantiate(cfg)
    data.prepare_data()
    data.setup()
