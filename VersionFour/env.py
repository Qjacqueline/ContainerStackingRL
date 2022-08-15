import sys

import numpy as np
import torch

if sys.version_info.major == 2:
    pass
else:
    pass

UNIT = 40  # pixels
dx = UNIT * 2
dy = UNIT


class Maze(object):
    def __init__(self,
                 s: int,
                 t: int,
                 w: int):
        super(Maze, self).__init__()
        self.BAY_S = s  # grid width
        self.BAY_T = t  # grid height
        self.w = w

        self.count = 0
        self.top = [0 for _ in range(s)]
        self.maze = [[0 for _ in range(t)] for _ in range(s)]
        self.action_flag = [1 for _ in range(s)]
        self.type_count = [s * t * 1.0 / w for _ in range(w)]  # fixme

    def reset(self):
        self.count = 0
        self.top = [0 for _ in range(self.BAY_S)]
        self.maze = [[0 for _ in range(self.BAY_T)] for _ in range(self.BAY_S)]
        self.action_flag = [1 for _ in range(self.BAY_S)]
        self.type_count = [self.BAY_S * self.BAY_T * 1.0 / self.w for _ in range(self.w)]  # fixme

    def get_state_action(self):
        # generate container
        p = np.array(self.type_count) / (self.BAY_T * self.BAY_S * 1.0 - self.count)
        index = np.random.choice([i for i in range(self.w)], p=p.ravel())
        # print(self.type_count)
        c_type = int(index) + 1
        self.type_count[index] -= 1
        self.count += 1

        s = torch.tensor(self.maze).reshape(1, -1)
        s = torch.cat((s, torch.tensor(c_type).reshape(1, -1)), 1)  # 当前贝位状态的矩阵表示+即将到达集装箱类型
        return s, c_type

    def get_state(self, c_type):
        # generate container
        s = torch.tensor(self.maze).reshape(1, -1)
        s = torch.cat((s, torch.tensor(c_type).reshape(1, -1)), 1)
        return s

    def step(self,
             action: int,
             c_type: int):
        # 如果动作超过限制
        if self.top[action] == self.BAY_T:
            done = 1
            reward = -100
            return reward, done, torch.tensor(self.action_flag).reshape(1, -1)
        else:
            self.maze[action][self.top[action]] = c_type
            self.top[action] += 1
            done = 0
            reward = self.cal_reward(action, c_type)
            if self.count == self.BAY_T * self.BAY_S:
                done = 1
            return reward, done, torch.tensor(self.action_flag).reshape(1, -1)

    def cal_LB1(self):
        block = 0
        for i in range(self.BAY_S):
            for j in range(1, self.BAY_T):
                flag = 0
                for k in range(0, j):
                    if self.maze[i][j] < self.maze[i][k]:
                        flag = 1
                if flag == 1:
                    block += 1
        return -block

    def cal_reward(self, action: int, c_type: int):
        block = 0
        for i in range(self.top[action] - 1):
            if self.maze[action][i] > c_type:
                block += 1
        return -block


if __name__ == '__main__':
    env = Maze(s=3, t=4, w=3)
    env.step(0, 2)
    env.step(0, 1)
    env.step(1, 1)
    env.step(1, 3)
