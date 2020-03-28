import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ReplayBuffer:
    def __init__(self, observation_space, action_space, max_size=1024):
        self.o = np.empty((max_size, *observation_space.shape))
        self.a = np.empty((max_size, *action_space.shape))
        self.r = np.empty((max_size,))
        self.n = np.empty((max_size, *observation_space.shape))
        self.d = np.empty((max_size,))
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

    def read(self, batch_size=64):
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
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.critic = Critic(2 + observation_space.shape[0])

    def act(self, observation, epsilon=0.1):
        if epsilon > np.random.rand():
            return self.action_space.sample()

        a1 = self.critic(torch.from_numpy(np.concatenate((np.array([1, 0]), observation))).float())
        a2 = self.critic(torch.from_numpy(np.concatenate((np.array([0, 1]), observation))).float())
        action = np.array([a1, a2]).argmax()
        return action

    def train(self, experience, alpha=0.01, gamma=0.99):
        # Q(s,a)=Q(s,a)-alpha*(r+gamma*max_a(Q(st+1,a))-Q(s,a))
        o, a, r, n, d = experience
        # if d == True: max a of any Q for st+1 = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', nargs='?', default='CartPole-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.seed(0)
    agent = Agent(env.observation_space, env.action_space)
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space)
    episode_count = 100
    for i in range(episode_count):
        observation = env.reset()
        while True:
            env.render()
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            replay_buffer.write(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                experience = replay_buffer.read()
                agent.train(experience)
                break
    env.close()


if __name__ == '__main__':
    main()
