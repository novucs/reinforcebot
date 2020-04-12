import torch
import torch.nn.functional as F
from torch import nn


class Predictor(nn.Module):
    def __init__(self, in_dim):
        super(Predictor, self).__init__()
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
        self.fc1 = nn.Linear(linear_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0), -1)).squeeze()


class Ensemble:
    def __init__(self, observation_space, size):
        self.predictors = [Predictor(observation_space) for _ in range(size)]

    def predict(self, observation):
        observation = torch.from_numpy(observation).float()
        total_reward = 0
        for predictor in self.predictors:
            total_reward += predictor(observation)
        mean_reward = total_reward / len(self.predictors)
        return mean_reward
