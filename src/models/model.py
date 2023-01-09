import torch
from pytorch_lightning import LightningModule
from torch import nn
import timm


class MyAwesomeConvNext(LightningModule):
    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_chans: int,
            num_classes: int,
            lr: float = 1e-3):
        super().__init__()
        self.model = timm.create_model(
                            model_name,
                            pretrained=pretrained,
                            in_chans=in_chans,
                            num_classes=num_classes)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3 or x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError("Expected each sample to have shape [3, 224, 224]")

        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
