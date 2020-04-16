import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def update_targets(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

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
