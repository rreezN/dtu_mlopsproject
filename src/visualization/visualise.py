import timm
import torch
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

def visualize_model():
    model = timm.create_model('convnext_atto', pretrained=True, in_chans=3, num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    make_dot(y.mean(), params=dict(model.named_parameters())).view()

if __name__ == "__main__":
    # Can't generate PDF if one already exists.
    visualize_model()
