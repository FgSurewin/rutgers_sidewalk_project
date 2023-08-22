import os
import torch
import json
from datasets.data_module import DataModule
from models.deeplabv3_model import DeepLabV3
from pytorch_lightning import Trainer
from torchvision.transforms import transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(
    DATA_DIR,
    NUM_CLASSES=25,
    DEV_RUN=False,
    BATCH_SIZE=8,
    EPOCHS=10,
    MODEL_NAME="models",
):
    # Create data module
    data_module = DataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=10,
        random_seed=42,
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    num_batches_train = len(train_loader)

    # Create model
    model = DeepLabV3(
        num_batches=num_batches_train,
        epochs=EPOCHS,
        num_classes=NUM_CLASSES,
        has_aux_loss=True,
    )

    # Create callbacks
    model_folder_path = f"checkpoint/{MODEL_NAME}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder_path,
        monitor="val_mean_iou",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        filename="{epoch:02d}-{val_loss:.4f}-{val_mean_iou:.4f}",
        verbose=False,
        mode="max",
    )

    # Set torch float precision
    torch.set_float32_matmul_precision("medium")
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger("tb_logs", name=MODEL_NAME)
    # Train model
    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=EPOCHS,
        log_every_n_steps=2,
        fast_dev_run=DEV_RUN,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=data_module)

    # Test model
    if not DEV_RUN:
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    data_dir = os.path.join("data", "sidewalk")
    # Read labels
    with open(os.path.join(data_dir, "label_info.json"), "r") as f:
        label_info = json.load(f)

    num_classes = len(label_info["label_to_id"])

    epochs = 50
    batch_size = 8
    model_name = f"v2_augment_sidewalk_deeplabv3_epoch_{epochs}"
    dev_run = False
    main(
        DATA_DIR=data_dir,
        NUM_CLASSES=num_classes,
        DEV_RUN=dev_run,
        EPOCHS=epochs,
        BATCH_SIZE=batch_size,
        MODEL_NAME=model_name,
    )
