# -*- codeing = utf-8 -*-
# @Time : 2022/5/20 09:53
# @Author : Jacqueline
# @File : RL_utils.py
from typing import List, Tuple, Any

import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import Dataset

from utils.common_utils import Logger

logger = Logger().get_logger()


def soft_update(tgt: nn.Module, src: nn.Module, tau: float) -> None:
    """
    Update target net
    """
    for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
        tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)


def print_result(field_name: List, value: List) -> None:
    res_table = PrettyTable()
    res_table.field_names = field_name
    res_table.add_row(value)
    print('\n', res_table)


class Buffer(Dataset):
    def __init__(self, buffer_size: int = 30000) -> None:
        super(Buffer, self).__init__()
        logger.info("创建data buffer")
        self.s = []
        self.action = []
        self.action_flag = []
        self.action_flag_ = []
        self.s_ = []
        self.reward = []
        self.done = []
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, index: int) -> Tuple:
        return (self.s[index],
                self.action[index],
                self.action_flag[index],
                self.action_flag_[index],
                self.s_[index],
                self.reward[index],
                self.done[index]
                )

    def append(self, s, ac, af, af_, s_, r, done):
        if len(self.done) == self.buffer_size:
            self.pop()
        self.s.append(s)
        self.action.append(ac)
        self.action_flag.append(af)
        self.action_flag_.append(af_)
        self.s_.append(s_)
        self.reward.append(r)
        self.done.append(done)

    def pop(self):
        self.s.pop(0)
        self.action.pop(0)
        self.action_flag.pop(0)
        self.action_flag_.pop(0)
        self.s_.pop(0)
        self.reward.pop(0)
        self.done.pop(0)


def collate_fn(raw_batch: List) -> Any:
    zip_raw_batch = zip(*raw_batch)
    return {'S': torch.cat(next(zip_raw_batch), dim=0),
            'action': torch.tensor(next(zip_raw_batch), dtype=torch.int64).reshape(-1, 1),
            'action_flag': torch.cat(next(zip_raw_batch), dim=0),
            'action_flag_': torch.cat(next(zip_raw_batch), dim=0),
            's_': torch.cat(next(zip_raw_batch), dim=0),
            'reward': torch.tensor(next(zip_raw_batch), dtype=torch.int64).reshape(-1, 1),
            'done': torch.tensor(next(zip_raw_batch), dtype=torch.int64).reshape(-1, 1)
            }


class PolicyBuffer(Dataset):
    def __init__(self, buffer_size: int = 1) -> None:
        super(PolicyBuffer, self).__init__()
        logger.info("创建policy buffer")
        self.log_pi = []
        self.reward = []
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.log_pi)

    def __getitem__(self, index: int) -> Tuple:
        return (self.log_pi[index],
                self.reward[index]
                )

    def append(self, log_pi, r):
        if len(self.log_pi) == self.buffer_size:
            self.pop()
        self.log_pi.append(log_pi)
        self.reward.append(r)

    def pop(self):
        self.log_pi.pop(0)
        self.reward.pop(0)


def collate_fn_policy(raw_batch: List) -> Any:
    zip_raw_batch = zip(*raw_batch)
    return {
            'log_pi': torch.tensor(next(zip_raw_batch)).reshape(-1, 1),
            'reward': torch.tensor(next(zip_raw_batch)).reshape(-1, 1)
            }
