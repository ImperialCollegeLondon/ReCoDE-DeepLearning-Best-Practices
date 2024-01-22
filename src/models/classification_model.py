from typing import Any, List, Union

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, Metric
from src.utils.utils import instantiate_from_config


class ClassificationModel(LightningModule):
    """Example of LightningModule for FashionMNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        metrics: List[Union[Metric, Metric]],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = criterion()

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = {}
        for train_metric in metrics:
            name = train_metric["name"]
            train_metric = instantiate_from_config(train_metric)
            self.train_metrics[f"train/{name}"] = train_metric
        self.val_metrics = {}
        for val_metric in metrics:
            name = val_metric["name"]
            val_metric = instantiate_from_config(val_metric)
            self.val_metrics[f"val/{name}"] = val_metric
        self.test_metrics = {}
        for test_metric in metrics:
            name = test_metric["name"]
            test_metric = instantiate_from_config(test_metric)
            self.test_metrics[f"test/{name}"] = test_metric

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        for train_metric_name in self.train_metrics:
            out_metric = self.train_metrics[train_metric_name](preds.cpu(), targets.cpu())
            self.log(train_metric_name, out_metric, prog_bar=True)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        for val_metric_name in self.val_metrics:
            out_metric = self.val_metrics[val_metric_name](preds.cpu(), targets.cpu())
            self.log(val_metric_name, out_metric, prog_bar=True)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        for test_metric_name in self.test_metrics:
            out_metric = self.test_metrics[test_metric_name](preds.cpu(), targets.cpu())
            self.log(test_metric_name, out_metric, prog_bar=True)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None and self.hparams.scheduler.func is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
