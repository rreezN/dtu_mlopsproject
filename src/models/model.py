import torch
from pytorch_lightning import LightningModule
from torch import nn
import timm
import wandb


class MyAwesomeConvNext(LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        in_chans: int,
        num_classes: int,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3:
            raise ValueError("Expected dim 1 of sample to have shape {3}")
        elif x.shape[2] != 224:
            raise ValueError("Expected dim 2 of sample to have shape {224}")
        elif x.shape[3] != 224:
            raise ValueError("Expected dim 3 of sample to have shape {224}")

        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        # on_epoch=True by default in `validation_step`,
        # so it is not necessary to specify
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return [preds, target]

    def validation_epoch_end(self, outs):
        preds, targets = list(zip(*outs))
        preds = torch.cat(preds).cpu().argmax(dim=1).numpy()
        targets = torch.cat(targets).cpu().numpy()
        self.logger.experiment.log({f"conf_mat_e{self.current_epoch}": wandb.plot.confusion_matrix(
                                                     probs=None,
                                                     y_true=targets, preds=preds)})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
