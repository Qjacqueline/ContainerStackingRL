"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import random


class QLearningTable:

    def __init__(self, n_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.7):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))  # a list
        self.forbid_actions = []
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation, :]
        dic = {}
        for i in range(len(state_action)):
            if i not in self.forbid_actions:
                dic[i]= state_action[i]
        sorted(dic.values())
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
             action = list(dic.keys())[0]
            # action =np.random.choice(np.max(list(dic.keys())))
            # some actions may have the same value, randomly choose on in these actions
            # action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = int(random.sample(dic.keys(), 1)[0])

        return action

    def learn(self, s, a, r, s_, end, top):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if top:
            self.forbid_actions.append(a)
        if not end:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            self.forbid_actions = []
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def printQlearningTable(self):
        print(self.q_table)