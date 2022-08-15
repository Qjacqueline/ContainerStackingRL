"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import sys
import time

import numpy as np

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # pixels
dx = UNIT * 2
dy = UNIT


class Maze(tk.Tk, object):
    def __init__(self, s, t, w, q):
        super(Maze, self).__init__()
        self.MAZE_S = s  # grid width
        self.MAZE_T = t  # grid height
        self.WN = w  # container classfication
        self.queue = q  # container arriving list

        self.position_space = np.zeros(self.MAZE_S)
        self.stepCount = 0
        self.maze = np.zeros(self.MAZE_S * self.MAZE_T + 1)
        self.maze[-1] = q[0]
        self.n_actions = self.MAZE_S

        self.top = np.zeros(self.MAZE_S, dtype=int, order='C')
        self.title('maze')
        self.geometry('{0}x{1}'.format((self.MAZE_S + 4) * UNIT, (self.MAZE_T + 4) * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=(self.MAZE_T + 4) * UNIT,
                                width=(self.MAZE_S + 4) * UNIT)
        self.canvas.create_text(UNIT * 1.5, (self.MAZE_T + 1.5) * UNIT, text='Stack')

        # create grids
        for c in range(dx, dx + (self.MAZE_S + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = c, dy, c, dy + self.MAZE_T * UNIT
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))
            if (c != (dx + self.MAZE_S * UNIT)):
                self.canvas.create_text(c + 0.5 * UNIT, (self.MAZE_T + 1.5) * UNIT, text=str(int((c - 1) / UNIT)))
                # print(str(x0) + " " + str(y0) + " " + str(x1) + " " + str(y1))

        for r in range(dy, dy + (self.MAZE_T + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = dx, r, dx + self.MAZE_S * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.top = np.zeros(self.MAZE_S, dtype=int, order='C')
        for i in range(self.stepCount):
            exec("self.canvas.delete('%sd')" % i)
            exec("self.canvas.delete('%st')" % i)
        self.stepCount = 0
        self.maze = np.zeros(self.MAZE_S * self.MAZE_T + 1)
        self.maze[-1] = self.queue[0]
        self.position_space = np.zeros(self.MAZE_S)

    def step(self, action):
        containerType = self.maze[-1]
        ## draw the newly container
        self.update()
        time.sleep(0.5)
        origin = np.array([dx, dy])
        hell_center = origin + np.array(
            [(action + 0.5) * UNIT, (self.MAZE_T - 0.5 - self.position_space[action]) * UNIT])
        exec(
            "self.hell%s=self.canvas.create_rectangle(hell_center[0] - 15, hell_center[1] - 15,hell_center[0] + 15, hell_center[1] + 15,fill='white',tag=str(self.stepCount)+'d')" % self.stepCount)
        # self.hell % S = self.canvas.create_rectangle(hell_center[0] - 15, hell_center[1] - 15, hell_center[0] + 15,hell_center[1] + 15, fill='white',tag=%S)
        self.canvas.create_text(hell_center[0], hell_center[1], text=str(containerType), tag=str(self.stepCount) + 't')
        # self.hell1 = self.canvas.create_rectangle(hell_center[0] - 15, hell_center[1] - 15, hell_center[0] + 15, hell_center[1] + 15, fill='black')

        ## update:stepCount/maze/position_space/end/top
        self.stepCount = self.stepCount + 1
        self.maze[action + int(self.position_space[action]) * self.MAZE_S] = containerType
        if self.stepCount == self.MAZE_T * self.MAZE_S:
            self.maze[-1] = -1
            end = True
        else:
            self.maze[-1] = self.queue[self.stepCount]
            end = False
        self.position_space[action] = self.position_space[action] + 1
        top = False  # 用于更新action可选范围
        if self.position_space[action] == self.MAZE_T:
            top = True

        ## calculate instant reward
        same = True
        reward = 0
        for i in range(int(self.position_space[action])):
            if (self.maze[action + i * self.MAZE_S] != containerType):
                same = False
                # if (self.maze[action + i * self.MAZE_S] > containerType):
                #     reward = reward - 1
        if same:
            reward = reward + 1

        ## calculate final reward
        s_ = self.maze
        block = 0
        if end:
            for i in range(self.MAZE_S):
                for j in range(1, self.MAZE_T):
                    temp_container = s_[j * self.MAZE_S + i]
                    for k in range(0, j):
                        compare_container = s_[k * self.MAZE_S + i]
                        if temp_container < compare_container:
                            block = block + 1
        reward = reward - block

        return s_, reward, end, top

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
