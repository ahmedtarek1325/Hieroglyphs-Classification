
from torchvision.models import resnet50 
import torch


def resnet50_model(path):
    model = resnet50()
    model.fc = torch.nn.Linear(2048, 134)
    model.load_state_dict(torch.load(path))
    return model

