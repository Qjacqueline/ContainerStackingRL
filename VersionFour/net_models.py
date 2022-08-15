# -*- coding = utf-8 -*-
# @Time : 2022/5/20 09:55
# @Author : Jacqueline
# @File : net_models.py
from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 0,
                 hidden_sizes: Sequence[int] = (256, 256),
                 activation: nn.Module = nn.LeakyReLU()
                 ) -> None:
        """
        Multilayer Perceptron

        Args:
            input_dim: dimension of input is [batch_size, input_dim]
            output_dim: dimension of output is [batch_size, output_dim]
            hidden_sizes: a sequence consisting of number of neurons per hidden layer
            activation: activation function
        """
        super().__init__()
        net = nn.Sequential()
        dim_last_layer = input_dim
        for i, num_neurons in enumerate(hidden_sizes):
            net.add_module(f'fc{i}', nn.Linear(dim_last_layer, num_neurons))
            net.add_module(f'act{i}', activation)
            dim_last_layer = num_neurons
        net.add_module('output layer', nn.Linear(dim_last_layer, output_dim))
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FlattenMlp(MLP):
    """
    Flatten inputs along dimension 1 ([u_action, u_state]).
    """

    def forward(self, *inputs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs)


class QNet(nn.Module):
    def __init__(self,
                 BAY_S: int,
                 BAY_T: int,
                 device: torch.device):
        super(QNet, self).__init__()
        self.device = device
        model = nn.ModuleDict()
        model['mlp'] = FlattenMlp(input_dim=BAY_S * BAY_T + 1, hidden_sizes=(256, 256, 256), output_dim=BAY_S)
        self.model = model.to(self.device)

    def forward(self, s: torch.tensor) -> torch.Tensor:
        x = self.model['mlp'](s.float())
        return x


class Dueling_DQN(nn.Module):
    def __init__(self,
                 BAY_S: int,
                 BAY_T: int,
                 device: torch.device):
        super(Dueling_DQN, self).__init__()
        self.device = device
        model = nn.ModuleDict()
        model['adv_mlp'] = FlattenMlp(input_dim=BAY_S * BAY_T + 1, hidden_sizes=(256, 256, 256), output_dim=BAY_S)
        model['val_mlp'] = FlattenMlp(input_dim=BAY_S * BAY_T + 1, hidden_sizes=(256, 256, 256), output_dim=1)
        self.model = model.to(self.device)

    def forward(self, s: torch.tensor) -> torch.Tensor:
        adv = self.model['adv_mlp'](s.float())
        value = self.model['val_mlp'](s.float()).repeat(1, adv.shape[1])
        Q = value + adv - adv.mean(1, keepdim=True)
        return Q


class Attention_QNet(nn.Module):
    def __init__(self,
                 BAY_S: int,
                 BAY_T: int,
                 device: torch.device):
        super(Attention_QNet, self).__init__()
        self.device = device
        model = nn.ModuleDict()
        model['mlp'] = FlattenMlp(input_dim=BAY_S * BAY_T + 1, hidden_sizes=(256, 256, 256), output_dim=BAY_S)
        self.model = model.to(self.device)

    def forward(self, s: torch.tensor) -> torch.Tensor:
        x = self.model['mlp'](s.float())
        return x
