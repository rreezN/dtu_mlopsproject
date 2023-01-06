import timm
import torch

import pickle
import os
import click
import matplotlib.pyplot as plt
import torch
import hydra
import logging
log = logging.getLogger(__name__)

from torch.utils.data import DataLoader, Dataset


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg) -> None:
    log.info("Training day and night")
    model_hparams = cfg.model
    training_hparams = cfg.training

    log.info(training_hparams.hyperparameters.lr)
    torch.manual_seed(training_hparams.hyperparameters.seed)

    model = timm.create_model(model_hparams.hyperparameters.model_name,
                              pretrained=model_hparams.hyperparameters.pretrained,
                              in_chans=model_hparams.hyperparameters.in_chans,
                              num_classes=model_hparams.hyperparameters.num_classes)

    device = "cuda" if training_hparams.hyperparameters.cuda else "cpu"
    log.info(f"device: {device}")
    model = model.to(device)

    with open(training_hparams.hyperparameters.train_data_path, 'rb') as handle:
        raw_data = pickle.load(handle)

    data = dataset(raw_data['images'], raw_data['labels'])
    dataloader = DataLoader(data, batch_size=training_hparams.hyperparameters.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_hparams.hyperparameters.lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = training_hparams.hyperparameters.epochs
    loss_tracker = []

    for epoch in range(n_epoch):
        running_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_tracker.append(running_loss/len(dataloader))
        log.info(f"Epoch {epoch + 1}/{n_epoch}. Loss: {running_loss/len(dataloader)}")
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    plt.plot(loss_tracker, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig(f"{os.getcwd()}/training_curve.png")


if __name__ == "__main__":
    train()
