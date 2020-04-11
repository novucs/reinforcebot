import argparse
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


def copy_params(origin, target, tau=1.0):
    for param, target_param in zip(origin.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, observation_space, action_space, max_size=128000):
        self.o = np.empty((max_size, *observation_space.shape), dtype=np.float32)
        self.a = np.empty((max_size, *action_space.shape), dtype=np.int8)
        self.r = np.empty((max_size,), dtype=np.float32)
        self.n = np.empty((max_size, *observation_space.shape), dtype=np.float32)
        self.d = np.empty((max_size,), dtype=np.bool)
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def write(self, o, a, r, n, d):
        self.o[self.index] = o
        self.a[self.index] = a
        self.r[self.index] = r
        self.n[self.index] = n
        self.d[self.index] = d
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size=8192):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.o[indices],
            self.a[indices],
            self.r[indices],
            self.n[indices],
            self.d[indices],
        )


class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self, observation_space, action_space,
                 alpha=0.001, epsilon=0.1, epsilon_decay=1.0, gamma=0.99, tau=0.005):
        self.action_space = action_space
        self.critic_target = Critic(2 + observation_space.shape[0])
        self.critic = Critic(2 + observation_space.shape[0])
        self.critic_criterion = nn.MSELoss()
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=alpha)
        copy_params(self.critic, self.critic_target)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

    def act(self, observation, epsilon=0.1):
        if epsilon > np.random.rand():
            return self.action_space.sample()

        a1 = self.critic_target(torch.from_numpy(np.concatenate((np.eye(2)[0], observation))).float())
        a2 = self.critic_target(torch.from_numpy(np.concatenate((np.eye(2)[1], observation))).float())
        action = np.array([a1, a2]).argmax()
        return action

    def train(self, experience):
        # Q(s,a)=Q(s,a)-alpha*(r+gamma*max_a(Q(st+1,a))-Q(s,a))
        o, a, r, n, d = experience
        with torch.no_grad():
            a1 = self.critic_target(torch.from_numpy(np.c_[np.tile(np.eye(2)[0], (len(n), 1)), n]).float())
            a2 = self.critic_target(torch.from_numpy(np.c_[np.tile(np.eye(2)[1], (len(n), 1)), n]).float())
            future_q = np.max(np.c_[a1, a2], axis=1)
            future_q[d] = 0
            labels = torch.reshape(torch.from_numpy(r + self.gamma * future_q), [len(a), 1])

        self.critic_optimiser.zero_grad()
        outputs = self.critic(torch.from_numpy(np.c_[np.eye(2)[a], o]).float())
        loss = self.critic_criterion(outputs, labels)
        loss.backward()
        self.critic_optimiser.step()
        self.epsilon *= self.epsilon_decay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', nargs='?', default='CartPole-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.seed(0)
    env._max_episode_steps = 200
    for _ in range(10):
        agent = Agent(env.observation_space, env.action_space)
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space)
        episode_count = 2000
        start = datetime.now()
        for i in range(episode_count):
            observation = env.reset()
            total_steps = 0
            while True:
                total_steps += 1
                # env.render()
                action = agent.act(observation)
                next_observation, reward, done, _ = env.step(action)
                reward = 0 if not done else 1 if total_steps == 200 else -1
                replay_buffer.write(observation, action, reward, next_observation, done)
                observation = next_observation
                if done:
                    print(f'episode {i}, total steps: {total_steps}')
                    experience = replay_buffer.read()
                    agent.train(experience)
                    with open(f'simple_training_{start}.csv', 'a') as file:
                        file.write(f'{i},{total_steps}\n')
                    if i % 5:
                        copy_params(agent.critic, agent.critic_target)
                    break
    env.close()


if __name__ == '__main__':
    main()
