import torch
import torchmetrics
from typing import Any
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)


class DeepLabV3(pl.LightningModule):
    def __init__(
        self,
        num_batches,
        epochs,
        num_classes,
        has_aux_loss,
        aux_importance_ratio=0.5,
        lr=0.0001,
        momentum=0.9,
        weight_decay=1e-4,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.num_batches = num_batches
        self.epochs = epochs
        self.num_classes = num_classes
        self.aux_importance_ratio = aux_importance_ratio
        self.model = self.prepare_model(
            num_classes=num_classes, has_aux_loss=has_aux_loss
        )
        self.kwargs = kwargs
        self.train_mean_iou_metric = torchmetrics.JaccardIndex(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_mean_iou_metric = torchmetrics.JaccardIndex(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.test_mean_iou_metric = torchmetrics.JaccardIndex(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.save_hyperparameters()

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        metrics = self.train_mean_iou_metric(output["out"].argmax(1), target)
        self.log(
            "train_mean_iou",
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.val_mean_iou_metric.update(output["out"], target)
        metrics = self.val_mean_iou_metric(output["out"].argmax(1), target)
        self.log(
            "val_mean_iou",
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    # def on_validation_epoch_end(self) -> None:
    #     output = self.val_mean_iou_metric.compute()
    #     self.log(
    #         "val_mean_iou",
    #         output,
    #         prog_bar=True,
    #         logger=True,
    #     )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # self.test_mean_iou_metric.update(output["out"], target)
        metrics = self.test_mean_iou_metric(output["out"].argmax(1), target)
        self.log(
            "test_mean_iou",
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    # def on_test_epoch_end(self) -> None:
    #     output = self.test_mean_iou_metric.compute()
    #     self.log(
    #         "test_mean_iou",
    #         output,
    #         prog_bar=True,
    #         logger=True,
    #     )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = self.create_lr_scheduler(
            optimizer, self.num_batches, self.epochs, warmup=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def criterion(self, output, target):
        loss = {}
        for key in output.keys():
            loss[key] = F.cross_entropy(output[key], target, ignore_index=255)

        if len(loss) == 1:
            return loss["out"]

        return loss["out"] + self.aux_importance_ratio * loss["aux"]

    def create_lr_scheduler(
        self,
        optimizer,
        num_step: int,
        epochs: int,
        warmup=True,
        warmup_epochs=1,
        warmup_factor=1e-3,
    ):
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0

        def f(x):
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                return warmup_factor * (1 - alpha) + alpha
            else:
                return (
                    1
                    - (x - warmup_epochs * num_step)
                    / ((epochs - warmup_epochs) * num_step)
                ) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

    def prepare_model(self, num_classes, has_aux_loss):
        model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT, aux_loss=has_aux_loss
        )
        model.classifier[-1] = torch.nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        if has_aux_loss:
            model.aux_classifier[-1] = torch.nn.Conv2d(
                256, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
        return model
