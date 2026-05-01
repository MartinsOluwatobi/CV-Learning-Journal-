import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self,x):
        return self.backbone(x) # B,3, 224, 224 → B, 2048, 7,7