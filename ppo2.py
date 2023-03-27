import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, mini_batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.mini_batch_size = mini_batch_size

    def sample(self):
        batch_start = np.arange(0, len(self.states), self.mini_batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), mini_batches

    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        entropy = dist.entropy()
        return dist, entropy


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPO:
    def __init__(self, n_states, n_actions, cfg):
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device

        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.critic = Critic(n_states, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.memory = PPOMemory(cfg.mini_batch_size)
        self.loss = 0

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist, entropy = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, prob, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]

            # 计算GAE
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            # mini batch 更新
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                # 计算新的策略分布
                dist, entropy = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # actor_loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # critic_loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # 更新
                entropy_loss = entropy.mean()
                total_loss = actor_loss + 0.5 * critic_loss - entropy_loss * 0.01
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()
        return self.loss

    def save(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint))
        self.critic.load_state_dict(torch.load(critic_checkpoint))