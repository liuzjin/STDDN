
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict
import numpy as np
import random, torch
import setproctitle

from st_supervisor import Trainer

torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of stddn')
    parser.add_argument('--dataset', default='gc', choices=['gc', 'ucy','eth','hotel'])
    parser.add_argument('--mode', default='test', choices=['train', 'test'])
    return parser.parse_args() 


def main():
    args = parse_args()
    args.config = f"./configs/{args.dataset}.yaml"
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for k, v in vars(args).items():
        config[k] = v
    config["exp_name"] = f'{args.config.split("/")[-1].split(".")[0]}_{config["valid_steps"]}'
    config["dataset"] = args.dataset
    config['mode'] = args.mode
    print(config)

    config = EasyDict(config)
    if not config.finetune:
        config.valid_steps = 2
    agent = Trainer(config)

    if config['mode'] == 'train':
        agent.train()
    else:
        agent.simulate()


if __name__ == '__main__':
    main()
