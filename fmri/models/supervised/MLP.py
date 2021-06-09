import random
import argparse
from abc import ABC

import torch.nn as nn
import torch


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


class MLP(nn.Module, ABC):
    def __init__(self,
                 n_neurons,
                 n_classes,
                 activation=torch.nn.ReLU,
                 ):
        super().__init__()
        self.activation = activation()
        self.dense1 = torch.nn.Linear(in_features=n_neurons[0], out_features=n_neurons[1])
        self.dense2 = torch.nn.Linear(in_features=n_neurons[1], out_features=n_classes)
        self.dense1_bn = nn.BatchNorm1d(num_features=n_neurons[1])
        self.dropout = nn.Dropout(0.5)

    def random_init(self, init_method=nn.init.xavier_uniform_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense1_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

class LinearClassifier(nn.Module, ABC):
    def __init__(self,
                 n_inputs,
                 n_classes,
                 activation=torch.nn.ReLU,
                 ):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_features=n_inputs, out_features=n_classes)

    def random_init(self, init_method=nn.init.xavier_uniform_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        return x

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
