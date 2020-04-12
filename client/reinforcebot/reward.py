import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
        x = F.relu(self.bn1(self.conv1(observation)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        one_hot_actions = np.zeros((action.shape[0], self.action_space))
        one_hot_actions[np.arange(action.shape[0]), action.numpy()] = 1
        x = torch.from_numpy(np.c_[one_hot_actions, x.view(x.size(0), -1)]).type(torch.float)
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
        pass
