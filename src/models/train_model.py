import logging
import os
import pickle
from typing import Tuple

import hydra
import torch
from model import MyAwesomeConvNext
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class dataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = images
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg) -> None:
    log.info("Training day and night")
    model_hparams = cfg.model
    train_hparams = cfg.training

    log.info(train_hparams.hyperparameters.lr)
    torch.manual_seed(train_hparams.hyperparameters.seed)

    model = MyAwesomeConvNext(
        model_name=model_hparams.hyperparameters.model_name,
        pretrained=model_hparams.hyperparameters.pretrained,
        in_chans=model_hparams.hyperparameters.in_chans,
        num_classes=model_hparams.hyperparameters.num_classes,
        lr=train_hparams.hyperparameters.lr
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="train_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=10, verbose=True, mode="min"
    )
    accelerator = "gpu" if train_hparams.hyperparameters.cuda else "cpu"
    trainer = Trainer(
        devices=1,
        accelerator=accelerator,
        max_epochs=train_hparams.hyperparameters.epochs,
        limit_train_batches=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(project="first train test", entity="dtu-mlopsproject"),
        # precision=16,
    )

    log.info(f"device (accelerator): {accelerator}")

    with open(train_hparams.hyperparameters.train_data_path, 'rb') as handle:
        image_data, images_labels = pickle.load(handle)
    log.info("Data loaded")
    data = dataset(image_data, images_labels.long())
    train_loader = DataLoader(
        data,
        batch_size=train_hparams.hyperparameters.batch_size
    )
    log.info("Dataset created")
    trainer.fit(model, train_dataloaders=train_loader)
    torch.save(model, f"{os.getcwd()}/trained_model.pt")


if __name__ == "__main__":
    train()
