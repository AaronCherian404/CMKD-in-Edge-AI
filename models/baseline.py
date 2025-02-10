import torch.nn as nn
import torchvision.models as models

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Using same architecture as student but without knowledge distillation
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[-1] = nn.Linear(576, num_classes)
    
    def forward(self, x):
        return self.backbone(x)