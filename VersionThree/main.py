import argparse
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from VersionOne.queue import generate_element, permuteUnique
from VersionThree.PolicyGradient import PG, policy_train, PGCollector
from VersionThree.env import Maze
from VersionThree.net_models import Policy
from utils.RL_utils import PolicyBuffer
from utils.common_utils import Logger, exp_dir

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='V3')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--s', type=int, default=4)
    parser.add_argument('--t', type=int, default=3)
    parser.add_argument('--w', type=str, default=2)

    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gamma', type=float, default=0.99)  # 0.9
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=1)

    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--collect_epoch_num', type=int, default=1)

    parser.add_argument('--save_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
    command_list = []
    for key, value in kwargs.items():
        command_list.append(f'--{key}={value}')
    return parser.parse_args(command_list)


if __name__ == '__main__':
    # ==============  Create environment & buffer  =============
    args = get_args()
    exp_dir = exp_dir(desc=f'{args.task + "/" + str(args.s) + "_" + str(args.t) + "_" + str(args.w)}')
    rl_logger = SummaryWriter(exp_dir)
    rl_logger.add_text(tag='parameters', text_string=str(args))
    rl_logger.add_text(tag='characteristic',
                       text_string='init')  # 'debug'

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    write_txt = open(args.save_path + "/" + str(args.s) + "_" + str(args.t) + "_" + str(args.w) + ".txt", 'w+')
    print("worked", file=write_txt)
    # ========================= Policy ======================
    BAY_S = args.s
    BAY_T = args.t
    w = args.w
    env = Maze(s=BAY_S, t=BAY_T, w=w)

    # ========================= Policy ======================
    agent = PG(
        policy=Policy(BAY_S=BAY_S, BAY_T=BAY_T, device=args.device),
        dim_action=BAY_S,
        device=args.device,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr)

    # ======================== Data ==========================
    # 产生100条随机test序列
    elements = generate_element(BAY_S, BAY_T, w)  # 生成集装箱分布
    test_ls = permuteUnique(elements)  # 生成分布下的全排列队列
    array = np.array(test_ls)
    row_rand_array = np.arange(array.shape[0])
    np.random.shuffle(row_rand_array)
    row_rand = array[row_rand_array[0:1000]].tolist()

    data_buffer = PolicyBuffer(buffer_size=args.buffer_size)
    collector = PGCollector(env=env, agent=agent, data_buffer=data_buffer, save_path=args.save_path, test_ls=row_rand)

    # ===================== collect & train ====================
    eval_count = 0
    for i in range(30):
        logger.info("开始第" + str(i) + "训练")
        policy_train(train_time=i, epoch_num=1000, collector=collector, rl_logger=rl_logger)
        if i % 5 == 0:
            total_reward = collector.final_eval(test_ls)
            rl_logger.add_scalar(tag=f'2.0_train/loss', scalar_value=total_reward,
                                 global_step=eval_count)
            print("开始第" + str(eval_count) + "测试LB1为：" + str(total_reward), file=write_txt)
            print("***************************************************")
            eval_count += 1
    rl_logger.close()
    write_txt.close()
