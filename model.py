from torch import nn
from torchvision.models import resnet18, resnet50, vgg16


class SimCLRCNN(nn.Module):
    def __init__(self, backbone, out_dim, mod=True):
        super(SimCLRCNN, self).__init__()
        backbone_dict = {"resnet18": resnet18, "resnet50": resnet50, "vgg16": vgg16}
        self.backbone = backbone_dict[backbone](num_classes=out_dim)
        self.out_dim = out_dim
        self.classifer = None
        if mod:
            # line 14-17 provides much improvement, but cost much more VRAM
            conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = conv1
            self.backbone.maxpool = nn.Identity()
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
            )

    def forward(self, x):
        x = self.backbone(x)
        if self.classifer is not None:
            x = self.classifer(x)
        return x

    def finetune(self, num_classes):
        # Freeze the parameters of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifer = nn.Linear(self.out_dim, num_classes)
