import gym
import torch
import numpy as np
from utils import plot_rewards, save_args, save_results, make_dir, plot_losses
from utils import Normalization, RewardScaling
from ppo2 import PPO
from param import get_args
from func import train, test


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    if cfg.continuous:
        n_actions = env.action_space.shape[0]
    else:
        n_actions = env.action_space.n
    agent = PPO(n_states, n_actions, cfg)
    if seed != 0:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent


if __name__ == "__main__":
    cfg = get_args()

    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    save_args(cfg, path=cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(res_dic, tag='train', path=cfg.result_path)
    plot_rewards(res_dic['rewards'], path=cfg.result_path, tag="train")
    plot_losses(res_dic['loss'], path=cfg.result_path)

    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test', path=cfg.result_path)
    plot_rewards(res_dic['rewards'], path=cfg.result_path, tag="test")


