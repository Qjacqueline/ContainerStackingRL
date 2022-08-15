# -*- coding = utf-8 -*-
# @Time : 2022/5/20 09:55
# @Author : Jacqueline
# @File : net_models.py
from typing import Sequence, Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


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


class Policy(nn.Module):
    def __init__(self,
                 BAY_S: int,
                 BAY_T: int,
                 device: torch.device):
        super(Policy, self).__init__()
        self.s = BAY_S
        self.t = BAY_T
        self.device = device
        model = nn.ModuleDict()
        model['mlp'] = FlattenMlp(input_dim=BAY_T + 1, hidden_sizes=(256, 256, 256), output_dim=1)
        self.model = model.to(self.device)

    def forward(self, s: torch.tensor, action_flag: torch.tensor) -> Tuple[Any, Any]:
        s_chunk = torch.split(s.float(), split_size_or_sections=self.t, dim=1)
        ls = []
        for i in range(self.s):
            tmp_stack = torch.cat((s_chunk[i], s_chunk[-1]), dim=1)
            goal_stack = self.model['mlp'](tmp_stack)
            ls.append(goal_stack)
        x = torch.cat(ls, 1)
        pi = F.softmax(x, dim=1) * action_flag
        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        return action, log_pi
