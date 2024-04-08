import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet_module(nn.Module):
    def __init__(self, embed_size = 512):
        super(ResNet_module, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_size)

        self.classifier = nn.Linear(embed_size, 2)

    def forward(self, x):
        embed = self.backbone(x)
        out = self.classifier(embed)
        return out

ps_keys = ['dopu', 'optic', 'retard', 'oct']
class PSOCT_module(nn.Module):
    def __init__(self, embed_size = 512, mode='dup_backbone', ps_keys = ps_keys):
        super(PSOCT_module, self).__init__()
        self.mode = mode
        self.ps_keys = ps_keys
        if mode == 'dup_backbone':
            self.backbone = nn.ModuleDict()
            for key in ps_keys:
                self.backbone[key] = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.backbone[key].fc = nn.Linear(self.backbone[key].fc.in_features, embed_size)
            
            self.aggregation = nn.Conv1d(in_channels=len(ps_keys), out_channels=1, kernel_size=1)
        elif mode == 'oct_only':
            self.backbone_oct = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone_oct.fc = nn.Linear(self.backbone_oct.fc.in_features, embed_size)
        else:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone.conv1 = nn.Conv2d(len(self.ps_keys * 3), self.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_size)

        self.classifier = nn.Linear(embed_size, 2)

    def forward(self, x):
        if self.mode == 'dup_backbone':
            embed = {}
            for key in self.ps_keys:
                embed[key] = self.backbone[key](x[key])
            embed = self.aggregation(torch.stack(list(embed.values()), dim=1))
        elif self.mode == 'oct_only':
            embed = self.backbone_oct(x['oct'])
        else:
            embed = self.backbone(x)
        out = self.classifier(embed).squeeze()
        return out