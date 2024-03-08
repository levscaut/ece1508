from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16

class SimCLRCNN(nn.Module):

    def __init__(self, backbone, out_dim):
        super(SimCLRCNN, self).__init__()
        backbone_dict = {
            "resnet18": resnet18,
            "vgg16": vgg16
        }
        self.backbone = backbone_dict[backbone](num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
    
    def forward(self, x):
        return self.backbone(x)
