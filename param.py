import datetime
import argparse
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyper parameters")
    """
    以下为环境配置
    """
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--continuous', default=False, type=bool, help="if PPO is continuous")
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden layer node')
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    """
    以下为调试的超参数
    """
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--mini_batch_size', default=5, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=4, type=int, help='update epochs')

    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')

    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    """
    以下为文件路径
    """
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/models/')
    """
    以下为trick的使用
    """
    parser.add_argument('--use_tanh', default=False, type=bool, help="trick:use tanh function")
    args = parser.parse_args()
    return args