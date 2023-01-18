import logging
import pickle
from typing import Tuple
import wandb
import hydra
import torch
from model import MyAwesomeConvNext
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

wandb.login(key='5b7c4dfaaa3458ff59ee371774798a737933dfa9')


class dataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = images
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def train(cfg) -> None:
    log.info("Training day and night")
    model_hparams = cfg.model
    train_hparams = cfg.training

    # print(cfg.training)

    # log.info("lr:", train_hparams.hyperparameters.lr)
    # log.info("batch size:", train_hparams.hyperparameters.batch_size)
    torch.manual_seed(train_hparams.hyperparameters.seed)

    model = MyAwesomeConvNext(
        model_name=model_hparams.hyperparameters.model_name,
        pretrained=model_hparams.hyperparameters.pretrained,
        in_chans=model_hparams.hyperparameters.in_chans,
        num_classes=model_hparams.hyperparameters.num_classes,
        lr=train_hparams.hyperparameters.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=train_hparams.hyperparameters.patience,
        verbose=True,
        mode="min"
    )
    accelerator = "gpu" if train_hparams.hyperparameters.cuda else "cpu"
    wandb_logger = WandbLogger(
        project="Final-Project", entity="dtu-mlopsproject", log_model="all"
    )
    for key, val in train_hparams.hyperparameters.items():
        wandb_logger.experiment.config[key] = val
    trainer = Trainer(
        devices="auto",
        accelerator=accelerator,
        max_epochs=train_hparams.hyperparameters.epochs,
        limit_train_batches=train_hparams.hyperparameters.limit_train_batches,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        precision=16
    )

    log.info(f"device (accelerator): {accelerator}")

    with open(train_hparams.hyperparameters.train_data_path, "rb") as handle:
        train_image_data, train_images_labels = pickle.load(handle)
    train_data = dataset(train_image_data, train_images_labels.long())
    train_loader = DataLoader(
        train_data,
        batch_size=train_hparams.hyperparameters.batch_size,
        num_workers=1,
        shuffle=True
    )

    with open(train_hparams.hyperparameters.val_data_path, "rb") as handle:
        val_image_data, val_images_labels = pickle.load(handle)

    val_data = dataset(val_image_data, val_images_labels.long())
    val_loader = DataLoader(
        val_data,
        batch_size=train_hparams.hyperparameters.batch_size,
        num_workers=1
    )

    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    # torch.save(model, f"{os.getcwd()}/trained_model.pt")


if __name__ == "__main__":
    train()
