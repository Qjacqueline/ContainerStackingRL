import argparse
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from DDQN import DDQN, Collector, l_train
from VersionOne.queue import generate_element, permuteUnique
from env import Maze
from net_models import Dueling_DQN
from utils.RL_utils import Buffer, collate_fn
from utils.common_utils import Logger, exp_dir

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='V2')
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

    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--buffer_size', type=int, default=300000)

    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--collect_epoch_num', type=int, default=2500)

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
                       text_string='dueling')  # 'debug'

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    write_txt = open(
        args.save_path + "/" + args.task + "_" + str(args.s) + "_" + str(args.t) + "_" + str(args.w) + ".txt", 'w+')
    print("worked", file=write_txt)

    # ========================= Policy ======================
    BAY_S = args.s
    BAY_T = args.t
    w = args.w
    env = Maze(s=BAY_S, t=BAY_T, w=w)
    test_env = Maze(s=BAY_S, t=BAY_T, w=w)

    # ========================= Policy ======================
    agent = DDQN(
        eval_net=Dueling_DQN(BAY_S=BAY_S, BAY_T=BAY_T, device=args.device),
        target_net=Dueling_DQN(BAY_S=BAY_S, BAY_T=BAY_T, device=args.device),
        dim_action=BAY_S,
        device=args.device,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr)

    # ======================== Data ==========================
    # ??????100?????????test??????
    elements = generate_element(BAY_S, BAY_T, w)  # ?????????????????????
    test_ls = permuteUnique(elements)  # ?????????????????????????????????
    array = np.array(test_ls)
    row_rand_array = np.arange(array.shape[0])
    np.random.shuffle(row_rand_array)
    row_rand = array[row_rand_array[0:1000]].tolist()

    data_buffer = Buffer(buffer_size=args.buffer_size)
    dl_train = DataLoader(dataset=data_buffer, batch_size=args.batch_size, collate_fn=collate_fn)
    collector = Collector(env=env, test_env=test_env, agent=agent, data_buffer=data_buffer, save_path=args.save_path,
                          test_ls=row_rand, dl_train=dl_train, rl_logger=rl_logger)

    # ===================== collect & train ====================
    eval_count = 0
    for i in range(args.epoch_num):
        collector.collect()
        # l_train(train_time=i, epoch_num=100, dl_train=dl_train, agent=agent, collector=collector, rl_logger=rl_logger)
        if i % 100 == 0:
            total_reward = collector.final_eval(test_ls)
            rl_logger.add_scalar(tag=f'1.0_train/loss', scalar_value=total_reward,
                                 global_step=eval_count)
            print("?????????" + str(eval_count) + "??????LB1??????" + str(total_reward), file=write_txt)
            eval_count += 1
    rl_logger.close()
    write_txt.close()
