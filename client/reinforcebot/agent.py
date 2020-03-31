import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim


def convert_pong_observation(observation):
    observation = np.array(Image.fromarray(observation).convert('L'))  # black and white (210x160)
    observation = observation[34:194]  # clip bounds (160x160)
    observation = observation[::2, ::2]  # downsize by skipping every other pixel (80x80)
    observation = observation / 256  # normalise
    return observation


def copy_params(origin, target, tau=1.0):
    for param, target_param in zip(origin.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, observation_space, action_space, max_size=int(2.5e5)):
        self.o = np.empty((max_size, *observation_space), dtype=np.float32)
        self.a = np.empty((max_size,), dtype=np.int8)
        self.r = np.empty((max_size,), dtype=np.float32)
        self.n = np.empty((max_size, *observation_space), dtype=np.float32)
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

    def read(self, batch_size=int(1e3)):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.o[indices],
            self.a[indices],
            self.r[indices],
            self.n[indices],
            self.d[indices],
        )


class Critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 200)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self, observation_space, action_space,
                 alpha=0.001, epsilon=0.1, epsilon_decay=1.0, gamma=0.99, tau=0.005):
        self.action_space = action_space
        self.critic_target = Critic(np.prod(observation_space), self.action_space)
        self.critic = Critic(np.prod(observation_space), self.action_space)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=alpha)
        copy_params(self.critic, self.critic_target)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

    def action(self, id_):
        """returns one-hot encoded action"""
        return np.eye(self.action_space)[id_]

    def act(self, observation, epsilon=0.1):
        if epsilon > np.random.rand():
            return np.random.randint(self.action_space)

        observation = observation.reshape(-1)
        a1, a2 = self.critic_target(torch.from_numpy(observation).float())

        if abs(a1 - a2) < 0.0015:
            print(f'confidence: {float(abs(a1 - a2)):.5f}')
            return np.random.randint(self.action_space)

        action = np.array([a1, a2]).argmax()
        print(action, f'up: {float(a1):.3f} down: {float(a2):.3f}')
        return action

    def train(self, experience):
        # Q(s,a)=Q(s,a)-alpha*(r+gamma*max_a(Q(st+1,a))-Q(s,a))
        o, a, r, n, d = experience
        o = o.reshape(-1, o.shape[0]).swapaxes(0, 1)
        n = n.reshape(-1, n.shape[0]).swapaxes(0, 1)
        r = r.reshape(-1, 1)
        with torch.no_grad():
            actions = torch.from_numpy(a).long().view(-1, 1)
            future_q = self.critic_target(torch.from_numpy(n).float())
            future_q = future_q.gather(1, actions)
            future_q[d] = 0
            targets = torch.from_numpy(r) + self.gamma * future_q

        self.critic_optimiser.zero_grad()
        outputs = self.critic(torch.from_numpy(o).float())
        targets = outputs.scatter(1, actions, targets)
        loss = self.critic_criterion(outputs, targets)
        loss.backward()
        self.critic_optimiser.step()
        self.epsilon *= self.epsilon_decay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', nargs='?', default='Pong-v0')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.seed(0)
    observation_space = (2, 80, 80)
    action_space = 2
    agent = Agent(observation_space, action_space)
    replay_buffer = ReplayBuffer(observation_space, action_space)
    episode_count = 10000
    for i in range(episode_count):
        previous_observation = np.zeros(observation_space[1:])
        observation = env.reset()
        observation = convert_pong_observation(observation)
        total_steps = 0
        while True:
            total_steps += 1
            env.render()
            stacked_observations = np.stack((previous_observation, observation))
            action = agent.act(stacked_observations)
            next_observation, reward, done, _ = env.step(2 if action == 0 else 3)
            next_observation = convert_pong_observation(next_observation)
            stacked_next_observations = np.stack((observation, next_observation))
            replay_buffer.write(stacked_observations, action, reward, stacked_next_observations, done)
            previous_observation, observation = observation, next_observation
            if done:
                print(f'episode {i}, total steps: {total_steps}')
                experience = replay_buffer.read()
                agent.train(experience)
                if i % 5:
                    copy_params(agent.critic, agent.critic_target)
                break
    env.close()


if __name__ == '__main__':
    main()
