from torch import nn
from torchvision.models import resnet18, resnet50, vgg16

class SimCLRCNN(nn.Module):

    def __init__(self, backbone, out_dim, mod=True):
        super(SimCLRCNN, self).__init__()
        backbone_dict = {
            "resnet18": resnet18,
            "resnet50": resnet50,
            "vgg16": vgg16
        }
        self.backbone = backbone_dict[backbone](num_classes=out_dim)
        if mod:
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
    
    def forward(self, x):
        return self.backbone(x)
