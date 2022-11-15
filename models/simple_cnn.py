from abc import *
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, simclr_dim=128, num_classes=2, pretrain=False):
        super().__init__()

        last_dim = 256

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(last_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_dim, num_classes)

        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

        ###########
        # Simsiam
        ###########
        prj_dim = last_dim * 8
        pred_dim = last_dim * 4
        self.simsiam_prj_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim, bias=False),
            # nn.BatchNorm1d(last_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(last_dim, last_dim, bias=False),
            nn.BatchNorm1d(last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(last_dim, prj_dim, bias=False),
            nn.BatchNorm1d(prj_dim, affine=False)
        )

        self.simsiam_pred_layer = nn.Sequential(
            nn.Linear(prj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, prj_dim)
        )

        ###########
        # vicReg
        ###########
        dim = 4096
        mlp_spec = [last_dim, dim, dim, dim]
        temp_layers = []
        for i in range(mlp_spec.__len__()-2):
            temp_layers.append(nn.Linear(mlp_spec[i], mlp_spec[i+1]))
            temp_layers.append(nn.BatchNorm1d(mlp_spec[i + 1]))
            temp_layers.append(nn.ReLU(True))
        temp_layers.append(nn.Linear(mlp_spec[-2], mlp_spec[-1], bias=False))
        self.vicReg_layer = nn.Sequential(*temp_layers)

    def penultimate(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        return feature

    def forward(self, inputs, penultimate=False, simclr=False, simsiam=False, vicReg=False, freeze=False):
        aux = {}
        assert penultimate or simclr

        features = self.penultimate(inputs)
        if freeze: features = features.detach()

        if penultimate:
            aux['penultimate'] = features

        if simclr:
            aux['simclr'] = self.simclr_layer(features)

        if simsiam:
            prj = self.simsiam_prj_layer(features)
            pred = self.simsiam_pred_layer(prj)
            aux['simsiam_prj'] = prj
            aux['simsiam_pred'] = pred

        if vicReg:
            aux['vicReg'] = self.vicReg_layer(features)

        return aux


def SimpleConvNet(num_classes, simclr_dim, pretrain=False):
    net = ConvNet(simclr_dim, num_classes)
    return net
