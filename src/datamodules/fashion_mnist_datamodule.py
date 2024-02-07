from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from src.datamodules.components.fashion_mnist_dataset import CustomFashionMNIST


class FashionMNISTDataModule(LightningDataModule):
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
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transform = transforms.Compose([transforms.ToTensor()])  # Add any additional transformations here

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None  # Uncomment if you have a test dataset

    def prepare_data(self):
        # Download the data (if not already present)
        CustomFashionMNIST(self.hparams.data_dir, train=True, download=True)
        CustomFashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if self.data_train is None:
            self.data_train = CustomFashionMNIST(self.hparams.data_dir, train=True, transform=self.transform)
        if self.data_val is None:
            self.data_val = CustomFashionMNIST(self.hparams.data_dir, train=False, transform=self.transform)
        # if self.data_test is None:
        #     self.data_test = CustomFashionMNIST(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
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
                shuffle=False,
            )
        else:
            return None

    def test_dataloader(self):
        if self.data_val:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                shuffle=False,
            )
        else:
            return None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from PIL import Image

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "fashion_mnist_datamodule.yaml")

    # Small hack to allow running this script without staring main config
    del cfg["defaults"]
    cfg["batch_size"] = 8

    print(cfg)
    data = hydra.utils.instantiate(cfg)
    data.prepare_data()
    data.setup()

    # Check the dataset loads correctly
    image, label = next(iter(data.train_dataloader()))
    print("Image train shape:", image.shape)
    print("Image train range:", image.min(), image.max())
    print("Label train:", label)
    print("Total number of images:", len(data.data_train))

    # Save one image to check it looks OK
    image = (image[0].squeeze().numpy() * 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(f"{root}/logs/data_check/image_with_label_{label[0]}.png")

    image, label = next(iter(data.val_dataloader()))
    print("Image val shape:", image.shape)
    print("Image val range:", image.min(), image.max())
    print("Label val:", label)
    print("Total number of images:", len(data.data_val))
    image = (image[0].squeeze().numpy() * 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(f"{root}/logs/data_check/image_val_with_label_{label[0]}.png")
