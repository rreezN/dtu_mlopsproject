import torch
# from pytorch_lightning import LightningModule
from model import MyAwesomeConvNext
import click


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("model_out", type=click.Path(exists=False))
def convert(model_path, model_out):
    print("Converting model")

    model = MyAwesomeConvNext.load_from_checkpoint(model_path)
    script = model.to_torchscript()
    torch.jit.save(script, model_out)

    print(f"Converted {model_path} to jit and saved at {model_out}")


if __name__ == "__main__":
    convert()
