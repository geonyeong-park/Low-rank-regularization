import torch
import torch.nn as nn
from munch import Munch
from models.resnet_simclr import ResNet18, ResNet50
import torchvision.models as models
from models.resnet_simclr import modify_last_layer
from models.simple_cnn import ConvNet

num_classes = {
    'celebA': 2,
    'UTKFace': 2,
    'bffhq': 2,
    'stl10mnist': 10,
    'imagenet': 9,
    'MIMIC_CXR': 2
}

arch = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'conv': ConvNet
}

arch_ERM = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
}

last_dim = {
    'resnet18': 512,
    'resnet50': 2048,
    'conv': 256,
}


class FC(nn.Module):
    def __init__(self, last_dim, num_classes=10):
        super(FC, self).__init__()
        self.fc = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_model(args):
    n_classes = num_classes[args.data]

    encoder = arch[args.arch](n_classes, args.simclr_dim, pretrain=True if args.data != 'imagenet' else False)
    classifier= FC(last_dim[args.arch], n_classes)

    nets = Munch(encoder=encoder,
                classifier=classifier)

    #classifier = arch_ERM[args.arch](pretrained=True if args.data != 'imagenet' else False)
    #classifier = modify_last_layer(classifier, n_classes=n_classes)

    #nets = Munch(classifier=classifier)
    return nets
