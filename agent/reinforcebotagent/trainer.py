import time
from threading import Thread

import numpy as np
import torch


class LocalTrainer:
    def __init__(self, agent_profile):
        self.agent_profile = agent_profile
        self.thread = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        starting = True
        experience_batch_size = self.agent_profile.app_config['EXPERIENCE_BATCH_SIZE']
        reward_batch_size = self.agent_profile.app_config['REWARD_BATCH_SIZE']

        def train(experience_buffer):
            o, a, n = experience_buffer.read(experience_batch_size)
            with torch.no_grad():
                r = self.agent_profile.reward_ensemble.predict(o, a).numpy()
            d = np.zeros(a.shape, dtype=np.float32)
            self.agent_profile.agent.train((o, a, r, n, d))

        while self.running:
            if self.agent_profile.agent_experience.size > 0:
                starting = False
                train(self.agent_profile.agent_experience)

            if self.agent_profile.user_experience.size > 0:
                starting = False
                train(self.agent_profile.user_experience)

            if self.agent_profile.reward_buffer.size > 0:
                starting = False
                s1, s2, p = self.agent_profile.reward_buffer.read(reward_batch_size)
                self.agent_profile.reward_ensemble.train(s1, s2, p)

            if starting:
                time.sleep(1)

    def experience(self, experience):
        if 'agent_transition' in experience:
            observation, action, next_observation = experience['agent_transition']
            self.agent_profile.agent_experience.write(observation, action, next_observation)
        if 'user_transition' in experience:
            observation, action, next_observation = experience['user_transition']
            self.agent_profile.user_experience.write(observation, action, next_observation)
        if 'rewards' in experience:
            segment1, segment2, preference = experience['rewards']
            self.agent_profile.reward_buffer.write(segment1, segment2, preference)

    def sample_segments(self, segment_size):
        segment1 = self.agent_profile.agent_experience.sample_segment(segment_size)
        segment2 = self.agent_profile.agent_experience.sample_segment(segment_size)
        return segment1, segment2

    def stop(self):
        self.running = False
        self.thread.join()
