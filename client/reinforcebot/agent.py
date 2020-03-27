import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ReplayBuffer:
    def __init__(self, observation_space, action_space, max_size=1024):
        self.d = np.empty((max_size,))
        self.s = np.empty((max_size, *observation_space.shape))
        self.a = np.empty((max_size, *action_space.shape))
        self.r = np.empty((max_size,))
        self.sp = np.empty((max_size, *observation_space.shape))
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def write(self, done, s, a, r, sp):
        self.d[self.index] = done
        self.s[self.index] = s
        self.a[self.index] = a
        self.r[self.index] = r
        self.sp[self.index] = sp
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size=64):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.d[indices],
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.sp[indices],
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

    def train(self, experience):
        # Q(s,a)=Q(s,a)-a*(r+max_a(Q(st+1,a))-Q(s,a))
        pass


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
        state = env.reset()
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.write(done, state, action, reward, next_state)
            state = next_state
            if done:
                experience = replay_buffer.read()
                agent.train(experience)
                break
    env.close()


if __name__ == '__main__':
    main()
