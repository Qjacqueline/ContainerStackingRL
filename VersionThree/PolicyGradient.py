# -*- codeing = utf-8 -*-
# @Time : 2022/5/20 09:37
# @Author : Jacqueline
# @File : DDQN.py

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List

import torch
from tensorboardX import SummaryWriter
from torch import nn

from env import Maze
from utils.RL_utils import soft_update, print_result, PolicyBuffer
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


class PG(BaseAgent):
    def __init__(self,
                 policy,
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
            policy:
            device:
            gamma:
            epsilon:
            soft_update_tau:
            loss_fn:
        """
        super(PG, self).__init__()
        logger.info("创建DDQN_agent")
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        self.dim_action = dim_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.update_tau = soft_update_tau
        self.device = device
        self.loss_func = loss_fn
        self.train_count = 0

    def forward(self, state, action_flag, eval_tag=True):
        action, log_pi = self.policy(state.to(self.device), action_flag.to(self.device))
        return action, log_pi

    def update(self, batch: Dict[str, Any]):
        # s = batch['S'].to(self.device)
        log_pi = batch['log_pi']
        # action_flag_ = batch['action_flag_'].to(self.device)
        # s_ = batch['s_'].to(self.device)
        reward = batch['reward'].to(self.device)
        # done = batch['done'].to(self.device)
        loss = -(log_pi * reward).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()


class PGCollector:
    def __init__(self,
                 env: Maze,
                 data_buffer: PolicyBuffer,
                 agent: PG,
                 test_ls: List[List[int]],
                 save_path: str):
        logger.info("创建data Collector")
        self.env = env
        self.data_buffer = data_buffer
        self.agent = agent
        self.best_result = 0
        self.test_ls = test_ls
        self.save_path = save_path

    def run(self):
        # logger.info("第" + str(i) + "轮收集RL交互数据")
        env_state, c_type = self.env.get_state_action()
        action_flag = torch.ones(self.env.BAY_S).reshape(1, -1)
        whole_env_loss = 0
        while True:
            batch = {}
            action, log_pi = self.agent.forward(state=env_state, action_flag=action_flag)
            reward, done, action_flag_ = self.env.step(action=action, c_type=c_type)
            if done == 0:
                new_state, c_type = self.env.get_state_action()
                batch['log_pi'] = log_pi
                batch['reward'] = torch.tensor(reward).float()
                env_state = new_state
                action_flag = action_flag_
            else:
                batch['log_pi'] = log_pi
                batch['reward'] = torch.tensor(reward).float()
                break
            loss = self.agent.update(batch=batch)
            whole_env_loss += loss.data
        self.env.reset()
        return whole_env_loss

    def eval(self):
        with torch.no_grad():
            total_reward = 0
            for i in range(len(self.test_ls)):
                c_type = self.test_ls[i][0]
                state = self.env.get_state(c_type)
                action_flag = torch.ones(self.env.BAY_S).reshape(1, -1)
                for j in range(len(self.test_ls[i])):
                    action, log_pi = self.agent.forward(state=state, eval_tag=False, action_flag=action_flag)
                    _, done, action_flag = self.env.step(action=action, c_type=c_type)
                    if done == 0:
                        c_type = self.test_ls[i][j]
                        new_state = self.env.get_state(c_type)
                        state = new_state
                    else:
                        total_reward += self.env.cal_LB1()
                        self.env.reset()
            if total_reward < self.best_result:
                self.best_result = total_reward
                torch.save(self.agent.policy,
                           self.save_path + '/' + str(self.env.BAY_S) + "_" + str(self.env.BAY_T) + "_" + str(
                               self.env.w) + "_" + 'eval_best.pkl')
            return total_reward

    def final_eval(self, test_ls):
        with torch.no_grad():
            total_reward = 0
            for i in range(len(test_ls)):
                c_type = test_ls[i][0]
                state = self.env.get_state(c_type)
                action_flag = torch.ones(self.env.BAY_S).reshape(1, -1)
                for j in range(len(test_ls[i])):
                    action, log_pi = self.agent.forward(state=state, eval_tag=False, action_flag=action_flag)
                    _, done, action_flag = self.env.step(action=action, c_type=c_type)
                    if done == 0:
                        c_type = test_ls[i][j]
                        new_state = self.env.get_state(c_type)
                        state = new_state
                    else:
                        total_reward += self.env.cal_LB1()
                        self.env.reset()
            return total_reward


def policy_train(train_time: int, epoch_num: int, collector: PGCollector,
                 rl_logger: SummaryWriter) -> None:
    total_loss = 0
    update_time = 100
    for epoch in range(epoch_num):
        whole_env_loss = collector.run()
        total_loss += whole_env_loss
        if epoch % update_time == 0:
            reward = collector.eval()
            # tensorboard
            rl_logger.add_scalar(tag=f'2.0_train/loss', scalar_value=total_loss / update_time,
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'2.0_train/reward', scalar_value=reward,
                                 global_step=epoch + train_time * epoch_num)
            # 画表
            field_name = ['Epoch', 'loss', 'reward']
            value = [epoch, total_loss / update_time, reward]
            print_result(field_name=field_name, value=value)
            total_loss = 0
