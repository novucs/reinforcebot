import numpy as np

from reinforcebot.config import SEGMENT_SIZE


class RewardReplayBuffer:
    def __init__(self, observation_space, segment_size=SEGMENT_SIZE, max_size=16):
        self.o1 = np.empty((max_size, segment_size, *observation_space), dtype=np.float32)
        self.a1 = np.empty((max_size, segment_size), dtype=np.int8)
        self.o2 = np.empty((max_size, segment_size, *observation_space), dtype=np.float32)
        self.a2 = np.empty((max_size, segment_size), dtype=np.int8)
        self.p = np.empty((max_size, segment_size), dtype=np.float32)
        self.index = 0
        self.size = 0
        self.max_size = max_size

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

    def read(self, batch_size=8):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            (self.o1[indices], self.a1[indices]),
            (self.o2[indices], self.a2[indices]),
            self.p[indices],
        )


class ExperienceReplayBuffer:
    def __init__(self, observation_space, max_size=int(2.5e5)):
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

    def read(self, batch_size=128):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.o[indices],
            self.a[indices],
            self.n[indices],
        )

    def sample_segment(self, size=SEGMENT_SIZE):
        start = np.random.randint(0, self.size - size)
        stop = start + size
        o = self.o[start:stop]
        a = self.a[start:stop]
        return o, a


class DynamicExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, observation_space, max_size=int(2.5e5)):
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
