import pickle

import timm
import click
import torch
import yaml
from yaml.loader import SafeLoader
from torch.utils.data import DataLoader, Dataset


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
def evaluate(model_filepath, test_filepath):
    print("Evaluating model")

    with open(model_filepath + ".hydra/config.yaml") as f:
        params = yaml.load(f, Loader=SafeLoader)

    model = timm.create_model(
        params.model.hyperparameters.model_name,
        pretrained=params.model.hyperparameters.pretrained,
        in_chans=params.model.hyperparameters.in_chans,
        num_classes=params.model.hyperparameters.num_classes,
    )

    model.load_state_dict(torch.load(model_filepath + "/trained_model.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with open(test_filepath, "rb") as handle:
        image_data, image_labels = pickle.load(handle)

    data = dataset(image_data, image_labels.long())
    dataloader = DataLoader(data, batch_size=100)

    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch

        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)

        correct += (preds == y.to(device)).sum().item()
        total += y.numel()

    print(f"Test set accuracy {correct / total}")


if __name__ == "__main__":
    evaluate()
