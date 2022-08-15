"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
from VersionOne.maze_env import Maze
from VersionOne.queue import generate_element, permuteUnique, writeIntoTxt
from VersionOne.RL_brain import QLearningTable


def update():
    for episode in range(80):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, end, top = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), end, top)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if end:
                break
        RL.printQlearningTable()
    # end of game1
    print(env.maze)
    print('game over')
    env.destroy()


if __name__ == "__main__":
    s, t, w = 3, 3, 3
    elements = generate_element(s, t, w)  # 生成集装箱分布
    res = permuteUnique(elements)  # 生成分布下的全排列队列
    writeIntoTxt(
        "/Users/jacqueline/Desktop/Tsing/Research/毕设/data & result/生成队列/S" + str(s) + "_t" + str(t) + "_w" + str(
            w) + ".txt", res)  # 写入txt文档
    env = Maze(s, t, w, res[60])
    RL = QLearningTable(env.n_actions)
    env.after(100, update)
    env.mainloop()
