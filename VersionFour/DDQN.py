# -*- codeing = utf-8 -*-
# @Time : 2022/5/20 09:37
# @Author : Jacqueline
# @File : DDQN.py

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from env import Maze
from utils.RL_utils import soft_update, Buffer, print_result
from utils.common_utils import Logger

logger = Logger().get_logger()


class BaseAgent(ABC, nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()
        """
        """

    @abstractmethod
    def forward(self, **kwargs):
        """
        """

    @abstractmethod
    def update(self, **kwargs):
        """

        """

    def sync_weight(self) -> None:
        """        Soft-update the weight for the target network.        """
        soft_update(tgt=self.qf_target, src=self.qf, tau=self.update_tau)


class DDQN(BaseAgent):
    def __init__(self,
                 eval_net,
                 target_net,
                 dim_action: int,
                 device: torch.device,
                 gamma: float,
                 epsilon: float,
                 lr: float,
                 soft_update_tau: float = 0.05,
                 loss_fn: Callable = nn.MSELoss(),
                 ) -> None:
        """

        Args:
            eval_net:
            target_net:
            device:
            gamma:
            epsilon:
            soft_update_tau:
            loss_fn:
        """
        super(DDQN, self).__init__()
        logger.info("创建DDQN_agent")
        self.qf = eval_net.to(device)
        self.qf_target = target_net.to(device)
        self.optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        self.dim_action = dim_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.update_tau = soft_update_tau
        self.device = device
        self.loss_func = loss_fn
        self.train_count = 0

    def forward(self, state, action_flag, eval_tag=True):
        state = state.to(self.device)
        if eval_tag:
            if np.random.rand() <= self.epsilon:  # greedy policy
                x = self.qf.forward(state)
                action = torch.max(x, 1)[1].item()
            else:  # random policy
                action = np.random.choice(4)
        else:
            x = self.qf(state).to(self.device)
            action = torch.max(x, 1)[1].item()
        return action

    def update(self, batch: Dict[str, Any]):
        s = batch['S'].to(self.device)
        action = batch['action'].to(self.device)
        s_ = batch['s_'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)
        q_eval_value = self.qf.forward(s)
        q_next_value = self.qf_target.forward(s_)
        q_eval = q_eval_value.gather(1, action)
        q_next = q_next_value.gather(1, torch.max(q_next_value, 1)[1].unsqueeze(1))
        q_target = reward + self.gamma * q_next * (1.0 - done)
        loss = self.loss_func(q_eval, q_target.detach())
        # print(torch.mean(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_count % 10 == 0:
            self.sync_weight()
        self.train_count += 1
        return loss.detach().mean(), q_eval.detach().mean()
    # self.writer.add_scalar("loss", loss, global_step=self.learn_step_counter)


class Collector:
    def __init__(self,
                 env: Maze,
                 test_env: Maze,
                 data_buffer: Buffer,
                 dl_train: DataLoader,
                 agent: DDQN,
                 test_ls: List[List[int]],
                 rl_logger: SummaryWriter,
                 save_path: str):
        logger.info("创建data Collector")
        self.env = env
        self.test_env = test_env
        self.data_buffer = data_buffer
        self.dl_train = dl_train
        self.agent = agent
        self.best_result = 0
        self.test_ls = test_ls
        self.save_path = save_path
        self.train_time = 0
        self.rl_logger = rl_logger

    def collect(self):
        state, c_type = self.env.get_state_action()
        action_flag = torch.ones(self.env.BAY_S).reshape(1, -1)
        while True:
            action = self.agent.forward(state=state, action_flag=action_flag)
            reward, done, action_flag_ = self.env.step(action=action, c_type=c_type)
            if done == 0:
                new_state, c_type = self.env.get_state_action()
                self.data_buffer.append(s=state, ac=action, af=action_flag, af_=action_flag_, s_=new_state,
                                        r=reward, done=done)
                state = new_state
                action_flag = action_flag_
            else:
                self.data_buffer.append(s=state, ac=action, af=action_flag, af_=action_flag_, s_=state, r=reward,
                                        done=done)
                break
            # train & eval
            self.train()
            self.train_time = self.train_time + 1
        self.env.reset()

    def train(self, train_num: int = 1):
        total_loss = 0
        total_q_eval = 0
        for i in range(train_num):
            batch = next(iter(self.dl_train))
            loss, q_eval = self.agent.update(batch)  # s_mission, s_station, s_cross, s_yard, l_station, l_cross, l_yard
            total_loss += loss.data
            total_q_eval += q_eval.data
        if self.train_time % 20 == 0:
            reward = self.eval()
            # tensorboard
            self.rl_logger.add_scalar(tag=f'1.0_train/loss', scalar_value=total_loss,
                                      global_step=self.train_time)
            self.rl_logger.add_scalar(tag=f'1.0_train/reward', scalar_value=reward,
                                      global_step=self.train_time)
            # 画表
            field_name = ['Epoch', 'loss', 'loss/q', 'reward']
            value = [self.train_time, total_loss, torch.sqrt(total_loss) * train_num / total_q_eval, reward]
            print_result(field_name=field_name, value=value)

    def eval(self):
        with torch.no_grad():
            total_reward = 0
            for i in range(len(self.test_ls)):
                c_type = self.test_ls[i][0]
                state = self.test_env.get_state(c_type)
                action_flag = torch.ones(self.test_env.BAY_S).reshape(1, -1)
                for j in range(len(self.test_ls[i])):
                    action = self.agent.forward(state=state, eval_tag=False, action_flag=action_flag)
                    _, done, action_flag = self.test_env.step(action=action, c_type=c_type)
                    if done == 0:
                        c_type = self.test_ls[i][j]
                        new_state = self.test_env.get_state(c_type)
                        state = new_state
                    else:
                        total_reward += self.test_env.cal_LB1()
                        self.test_env.reset()

            if total_reward < self.best_result:
                self.best_result = total_reward
                torch.save(self.agent.qf,
                           self.save_path + '/' + str(self.env.BAY_S) + "_" + str(self.env.BAY_T) + "_" + str(
                               self.env.w) + "_" + 'eval_best.pkl')
                torch.save(self.agent.qf_target,
                           self.save_path + '/' + str(self.env.BAY_S) + "_" + str(self.env.BAY_T) + "_" + str(
                               self.env.w) + "_" + 'target_best.pkl')
            return total_reward

    def final_eval(self, test_ls):
        with torch.no_grad():
            total_reward = 0
            for i in range(len(test_ls)):
                c_type = test_ls[i][0]
                state = self.env.get_state(c_type)
                action_flag = torch.ones(self.env.BAY_S).reshape(1, -1)
                for j in range(len(test_ls[i])):
                    action = self.agent.forward(state=state, eval_tag=False, action_flag=action_flag)
                    _, done, action_flag = self.env.step(action=action, c_type=c_type)
                    if done == 0:
                        c_type = test_ls[i][j]
                        new_state = self.env.get_state(c_type)
                        state = new_state
                    else:
                        total_reward += self.env.cal_LB1()
                        self.env.reset()
            return total_reward


def l_train(train_time: int, epoch_num: int, dl_train: DataLoader, agent: DDQN, collector: Collector,
            rl_logger: SummaryWriter) -> None:
    for epoch in range(epoch_num):
        with tqdm(dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_loss = 0
            for batch in pbar:
                loss = agent.update(batch)  # s_mission, s_station, s_cross, s_yard, l_station, l_cross, l_yard
                total_loss += loss.data
            reward = collector.eval()
            # tensorboard
            rl_logger.add_scalar(tag=f'1.0_train/loss', scalar_value=total_loss / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'1.0_train/reward', scalar_value=reward,
                                 global_step=epoch + train_time * epoch_num)
            # 画表
            field_name = ['Epoch', 'loss', 'reward']
            value = [epoch, total_loss / len(pbar), reward]
            print_result(field_name=field_name, value=value)
