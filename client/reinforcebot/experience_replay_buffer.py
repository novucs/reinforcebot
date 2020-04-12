import numpy as np


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


class DynamicExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, observation_space, max_size=int(2.5e5)):
        super(DynamicExperienceReplayBuffer, self).__init__(observation_space, max_size)
        self.a = [''] * max_size
        self.action_space = set()

    def write(self, o, a, n):
        a = ','.join(map(str, a))
        super(DynamicExperienceReplayBuffer, self).write(o, a, n)
        self.action_space.add(a)

    def build(self):
        action_mapping = {idx: action for idx, action in enumerate(self.action_space)}
        inverse_action_mapping = {action: idx for idx, action in action_mapping.items()}
        buffer = ExperienceReplayBuffer(self.observation_space, self.max_size)
        buffer.o = self.o
        buffer.n = self.n
        buffer.a = np.array([inverse_action_mapping[a] for a in self.a[:self.size]], dtype=np.int8)
        return action_mapping, buffer
