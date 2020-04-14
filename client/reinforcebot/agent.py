import argparse
import json
import os

import gym
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision.transforms.functional import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_pong_observation(observation):
    observation = np.array(Image.fromarray(observation).convert('L'))  # black and white (210x160)
    observation = observation[34:194]  # clip bounds (160x160)
    observation = observation[::2, ::2]  # downsize by skipping every other pixel (80x80)
    observation = observation / 256  # normalise
    return observation


def convert_cartpole_observation(observation):
    observation = Image.fromarray(observation).convert('L')
    observation = resize(observation, (80, 80))
    return np.array(observation)


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

    def read(self, batch_size=128):
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
        steps, width, height = in_dim
        ks = 5
        stride = 2
        self.conv1 = nn.Conv2d(steps, 16, kernel_size=ks, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=ks, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=ks, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)

        def convolved_size(size, layers=3):
            return size if layers <= 0 else \
                convolved_size((size - ks) // stride + 1, layers - 1)

        linear_size = convolved_size(width) * convolved_size(height) * 32
        self.fc1 = nn.Linear(linear_size, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0), -1)).squeeze()


class Agent:
    def __init__(self, observation_space, action_space,
                 alpha=0.001, epsilon=0.1, epsilon_decay=1.0, gamma=0.99, tau=0.005):
        self.observation_space = observation_space
        self.action_space = action_space
        self.critic_target = Critic(observation_space, self.action_space).to(device)
        self.critic = Critic(observation_space, self.action_space).to(device)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=alpha)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

    def act(self, observation, epsilon=0.1):
        if epsilon > np.random.rand():
            return np.random.randint(self.action_space)

        q = self.critic(torch.from_numpy(observation).unsqueeze(0).float().to(device))
        return int(q.argmax())

    def train(self, experience):
        # Q(s,a)=Q(s,a)-alpha*(r+gamma*max_a(Q(st+1,a))-Q(s,a))
        o, a, r, n, d = experience
        with torch.no_grad():
            actions = torch.from_numpy(a).long().view(-1, 1).to(device)
            future_q = self.critic_target(torch.from_numpy(n).float().to(device))
            future_q = future_q.gather(1, actions)
            future_q[d] = 0
            targets = torch.from_numpy(r).to(device) + self.gamma * future_q

        self.critic_optimiser.zero_grad()
        outputs = self.critic(torch.from_numpy(o).float().to(device))
        targets = outputs.scatter(1, actions, targets)
        loss = self.critic_criterion(outputs, targets)
        loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimiser.step()

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'settings.json'), 'w') as settings_file:
            json.dump({
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'alpha': self.alpha,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'gamma': self.gamma,
                'tau': self.tau,
            }, settings_file, indent=2)

        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
        }, os.path.join(path, 'agent.pt'))

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'settings.json'), 'r') as settings_file:
            settings = json.load(settings_file)

        agent = Agent(settings['observation_space'], settings['action_space'],
                      settings['alpha'], settings['epsilon'],
                      settings['epsilon_decay'], settings['gamma'], settings['tau'])
        checkpoint = torch.load(os.path.join(path, 'agent.pt'))
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.critic.eval()
        agent.critic_target.eval()
        return agent


def main(env_id):
    env = gym.make(env_id)
    env.seed(0)
    observation_space = (2, 80, 80)
    action_space = 2
    agent = Agent(observation_space, action_space)
    replay_buffer = ReplayBuffer(observation_space, action_space)
    episode_count = 10000
    for i in range(episode_count):
        previous_observation = np.zeros(observation_space[1:])
        observation = env.reset()
        observation = env.render(mode='rgb_array')
        observation = convert_cartpole_observation(observation)
        total_steps = 0
        while True:
            total_steps += 1
            env.render()
            stacked_observations = np.stack((previous_observation, observation))
            action = agent.act(stacked_observations)
            # next_observation, reward, done, _ = env.step(2 if action == 0 else 3)
            next_observation, reward, done, _ = env.step(action)
            next_observation = env.render(mode='rgb_array')
            next_observation = convert_cartpole_observation(next_observation)
            stacked_next_observations = np.stack((observation, next_observation))
            replay_buffer.write(stacked_observations, action, reward, stacked_next_observations, done)
            previous_observation, observation = observation, next_observation
            experience = replay_buffer.read()
            agent.train(experience)
            if done:
                print(f'episode {i}, total steps: {total_steps}')
                torch.save(agent.critic.state_dict(), 'agent')
                if i % 5:
                    agent.critic_target.load_state_dict(agent.critic.state_dict())
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', nargs='?', default='CartPole-v0')
    args = parser.parse_args()
    main(args.env_id)
