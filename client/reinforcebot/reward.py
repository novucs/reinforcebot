import torch
import torch.nn.functional as F
from torch import nn, optim


class Predictor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Predictor, self).__init__()
        steps, width, height = observation_space
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
        self.fc1 = nn.Linear(linear_size + action_space, 1)
        self.action_space = action_space

    def forward(self, x):
        observation, action = x
        action_encoded = torch.zeros(action.shape[0], self.action_space)
        action_encoded[torch.arange(action.shape[0]), action.long()] = 1

        x = F.relu(self.bn1(self.conv1(observation)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat((action_encoded, x.view(x.size(0), -1)), dim=1)
        return self.fc1(x).squeeze()


class Ensemble:
    def __init__(self, observation_space, action_space, size):
        self.predictors = [Predictor(observation_space, action_space) for _ in range(size)]

    def predict(self, observation, action):
        observation = torch.from_numpy(observation).float()
        action = torch.from_numpy(action)
        total_reward = 0
        for predictor in self.predictors:
            total_reward += predictor((observation, action))
        mean_reward = total_reward / len(self.predictors)
        return mean_reward

    def train(self, segment1, segment2, preference):
        for predictor in self.predictors:
            optimiser = optim.Adam(predictor.parameters())
            optimiser.zero_grad()
            loss = torch.zeros(1)

            for o1, a1, o2, a2, hp in zip(*segment1, *segment2, preference):
                hp = float(hp)
                s1 = predictor((torch.from_numpy(o1), torch.from_numpy(a1))).sum()
                s2 = predictor((torch.from_numpy(o2), torch.from_numpy(a2))).sum()
                p1 = torch.exp(s1) / (torch.exp(s1) + torch.exp(s2))  # prob. segment 1 > segment 2
                p2 = torch.exp(s2) / (torch.exp(s1) + torch.exp(s2))  # prob. segment 2 > segment 1
                loss += - (hp * p1 + (1 - hp) * p2)

            loss.backward()
            for param in predictor.parameters():
                param.grad.data.clamp_(-1, 1)
            optimiser.step()
