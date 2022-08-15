# -*- codeing = utf-8 -*-
# @Time : 2021/10/11 17:37
# @Author : Jacqueline
# @File : queue.py
import os
from math import floor
from typing import List


def generate_element(s, t, wn):
    component = []
    if wn == 2:
        N1 = floor(s * t * 1 / 2)
        N2 = s * t - N1
        print(N1, N2)
        while N1 > 0:
            component.append(1)
            N1 -= 1
        while N2 > 0:
            component.append(2)
            N2 -= 1
    if wn == 3:
        N1 = floor(s * t * 1 / 3)
        N2 = floor(s * t * 1 / 3)
        N3 = s * t - N1 - N2
        # print(N1,N2,N3)
        while N1 > 0:
            component.append(1)
            N1 -= 1
        while N2 > 0:
            component.append(2)
            N2 -= 1
        while N3 > 0:
            component.append(3)
            N3 -= 1
    return component


def permuteUnique(nums: List[int]) -> List[List[int]]:
    nums.sort()
    queues = []
    temp = []

    def back(num, tmp):
        if not num:
            queues.append(tmp)
            return
        else:
            for i in range(len(num)):
                if i > 0 and num[i] == num[i - 1]:
                    continue
                back(num[:i] + num[i + 1:], tmp + [num[i]])

    back(nums, temp)
    return queues


def writeIntoTxt(path, ress):
    with open(str(path), "w") as f:
        for re in ress:
            for i in re:
                f.write(str(i))
            f.write("\n")


if __name__ == "__main__":
    ROOT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    S, T, W = 4, 3, 3
    QUEUE_PATH = os.path.join(ROOT_FOLDER_PATH, 'queue')
    if not os.path.exists(QUEUE_PATH):
        os.mkdir(QUEUE_PATH)
    elements = generate_element(S, T, W)  # 生成集装箱分布
    res = permuteUnique(elements)  # 生成分布下的全排列队列
    writeIntoTxt(QUEUE_PATH + "/" + str(S * T) + "_" + str(W) + ".txt", res)  # 写入txt文档
