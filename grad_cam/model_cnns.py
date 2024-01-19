import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet, Bottleneck

class DenseNet121(DenseNet):
    def __init__(self, num_classes=1000):
        super(DenseNet121, self).__init__(block_config=(6, 12, 24, 16), num_classes=num_classes)
    
    def set_specific_id(self, specific_id):
        self.specific_id = specific_id

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out[:, self.specific_id]
    
class ResNet50(ResNet):
    def __init__(self, num_classes=14):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    def set_specific_id(self, specific_id):
        self.specific_id = specific_id

    def forward(self, x):
        return self._forward_impl(x)[:, self.specific_id]