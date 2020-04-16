import json
import os

import numpy as np


class RewardReplayBuffer:
    def __init__(self, observation_space, segment_size, max_size):
        self.o1 = np.empty((max_size, segment_size, *observation_space), dtype=np.float32)
        self.a1 = np.empty((max_size, segment_size), dtype=np.int8)
        self.o2 = np.empty((max_size, segment_size, *observation_space), dtype=np.float32)
        self.a2 = np.empty((max_size, segment_size), dtype=np.int8)
        self.p = np.empty((max_size,), dtype=np.float32)
        self.index = 0
        self.size = 0
        self.max_size = max_size
        self.observation_space = observation_space
        self.segment_size = segment_size

    def write(self, s1, s2, p):
        o1, a1 = s1
        o2, a2 = s2
        self.o1[self.index] = o1
        self.a1[self.index] = a1
        self.o2[self.index] = o2
        self.a2[self.index] = a2
        self.p[self.index] = p
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            (self.o1[indices], self.a1[indices]),
            (self.o2[indices], self.a2[indices]),
            self.p[indices],
        )

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'settings.json'), 'w') as settings_file:
            json.dump({
                'index': self.index,
                'size': self.size,
                'max_size': self.max_size,
                'observation_space': self.observation_space,
                'segment_size': self.segment_size,
            }, settings_file, indent=2)

        np.save(os.path.join(path, 'segment1_observations.npy'), self.o1)
        np.save(os.path.join(path, 'segment1_actions.npy'), self.a1)
        np.save(os.path.join(path, 'segment2_observations.npy'), self.o2)
        np.save(os.path.join(path, 'segment2_actions.npy'), self.a2)
        np.save(os.path.join(path, 'preferences.npy'), self.p)

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'settings.json'), 'r') as settings_file:
            settings = json.load(settings_file)

        buffer = RewardReplayBuffer(settings['observation_space'], settings['segment_size'], settings['max_size'])
        buffer.index = settings['index']
        buffer.size = settings['size']
        buffer.o1 = np.load(os.path.join(path, 'segment1_observations.npy'))
        buffer.a1 = np.load(os.path.join(path, 'segment1_actions.npy'))
        buffer.o2 = np.load(os.path.join(path, 'segment2_observations.npy'))
        buffer.a2 = np.load(os.path.join(path, 'segment2_actions.npy'))
        buffer.p = np.load(os.path.join(path, 'preferences.npy'))
        return buffer


class ExperienceReplayBuffer:
    def __init__(self, observation_space, max_size):
        self.o = np.empty((max_size, *observation_space), dtype=np.float32)
        self.a = np.empty((max_size,), dtype=np.int8)
        self.n = np.empty((max_size, *observation_space), dtype=np.float32)
        self.index = 0
        self.size = 0
        self.max_size = max_size
        self.observation_space = observation_space

    def write(self, o, a, n):
        self.o[self.index] = o
        self.a[self.index] = a
        self.n[self.index] = n
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.o[indices],
            self.a[indices],
            self.n[indices],
        )

    def sample_segment(self, size):
        start = np.random.randint(0, self.size - size)
        stop = start + size
        o = self.o[start:stop]
        a = self.a[start:stop]
        return o, a

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'settings.json'), 'w') as settings_file:
            json.dump({
                'index': self.index,
                'size': self.size,
                'max_size': self.max_size,
                'observation_space': self.observation_space,
            }, settings_file, indent=2)

        np.save(os.path.join(path, 'observations.npy'), self.o)
        np.save(os.path.join(path, 'actions.npy'), self.a)
        np.save(os.path.join(path, 'next_observations.npy'), self.n)

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'settings.json'), 'r') as settings_file:
            settings = json.load(settings_file)

        buffer = ExperienceReplayBuffer(settings['observation_space'], settings['max_size'])
        buffer.index = settings['index']
        buffer.size = settings['size']
        buffer.o = np.load(os.path.join(path, 'observations.npy'))
        buffer.a = np.load(os.path.join(path, 'actions.npy'))
        buffer.n = np.load(os.path.join(path, 'next_observations.npy'))
        return buffer


class DynamicExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, observation_space, max_size):
        super(DynamicExperienceReplayBuffer, self).__init__(observation_space, max_size)
        self.a = [''] * max_size
        self.action_space = {'', }

    def write(self, o, a, n):
        a = ','.join(map(str, a))
        super(DynamicExperienceReplayBuffer, self).write(o, a, n)
        self.action_space.add(a)

    def build(self):
        conversion = {action: idx for idx, action in enumerate(self.action_space)}
        buffer = ExperienceReplayBuffer(self.observation_space, self.max_size)
        buffer.o = self.o
        buffer.n = self.n
        buffer.a[:self.size] = np.array([conversion[a] for a in self.a[:self.size]], dtype=np.int8)
        action_mapping = {
            idx: set(map(int, action.split(','))) if action else set()
            for idx, action in enumerate(self.action_space)
        }
        return action_mapping, buffer
