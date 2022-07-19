from abc import *
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.model_zoo import load_url

URL_DICT = {
    'resnet18': "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    'resnet101': "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
}


class SimCLRResNet(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 simclr_dim=128,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
            ) -> None:
        super(SimCLRResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                        groups, width_per_group, replace_stride_with_dilation,
                                        norm_layer)
        last_dim = 512 * block.expansion

        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

    def penultimate(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        return feature

    def forward(self, inputs, penultimate=False, simclr=False, freeze=False):
        aux = {}
        assert penultimate or simclr

        features = self.penultimate(inputs)
        if freeze: features = features.detach()

        if penultimate:
            aux['penultimate'] = features

        if simclr:
            aux['simclr'] = self.simclr_layer(features)

        return aux


def ResNet18(num_classes, simclr_dim):
    net = SimCLRResNet(BasicBlock, [2, 2, 2, 2], simclr_dim)
    url = URL_DICT['resnet18']
    checkpoint = load_url(url)
    net.load_state_dict(checkpoint, strict=False)
    net = modify_last_layer(net, num_classes)
    print(f'Load {url}')
    return net

def ResNet50(num_classes, simclr_dim):
    net = SimCLRResNet(Bottleneck, [3, 4, 6, 3], simclr_dim)
    url = URL_DICT['resnet50']
    checkpoint = load_url(url)
    net.load_state_dict(checkpoint, strict=False)
    net = modify_last_layer(net, num_classes)
    print(f'Load {url}')
    return net

def Build_ResNet(base_model, num_classes, simclr_dim):
    if base_model == 'resnet18':
        return ResNet18(num_classes, simclr_dim)
    elif base_model == 'resnet50':
        return ResNet50(num_classes, simclr_dim)
    else:
        return NotImplementedError

def modify_last_layer(net, n_classes=10):
    d = net.fc.in_features
    net.fc = nn.Linear(d, n_classes)
    return net
