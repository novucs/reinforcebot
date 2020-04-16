import time
from threading import Thread

import numpy as np
import torch


class CloudComputeTrainer:
    def __init__(self, app, agent_profile):
        self.app = app
        self.agent_profile = agent_profile
        self.token = None
        self.running = False
        self.thread = None

    def start(self):
        self.token = self.app.start_runner(self.agent_profile.agent_id, self.agent_profile.agent.dump_parameters())
        if not self.token:
            return False

        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()
        return True

    def run(self):
        while self.running:
            time.sleep(1)
            parameters = self.app.fetch_runner_parameters(self.token)
            if not parameters:
                self.running = False
                break

            self.agent_profile.agent.load_parameters(parameters)

    def experience(self, experience):
        payload = {}

        if 'agent_transition' in experience:
            observation, action, next_observation = experience['agent_transition']
            self.agent_profile.agent_experience.write(observation, action, next_observation)
            payload['agent_transition'] = {
                'observation': observation.tolist(),
                'action': observation.tolist(),
                'next_observation': next_observation.tolist(),
            }

        if 'user_transition' in experience:
            observation, action, next_observation = experience['user_transition']
            self.agent_profile.user_experience.write(observation, action, next_observation)
            payload['user_transition'] = {
                'observation': observation.tolist(),
                'action': observation.tolist(),
                'next_observation': next_observation.tolist(),
            }

        if 'rewards' in experience:
            segment1, segment2, preference = experience['rewards']
            self.agent_profile.reward_buffer.write(segment1, segment2, preference)
            payload['user_transition'] = {
                'segment1': segment1.tolist(),
                'segment2': segment2.tolist(),
                'preference': preference,
            }

        self.app.add_runner_experience(self.token, payload)

    def sample_segments(self, segment_size):
        segment1 = self.agent_profile.agent_experience.sample_segment(segment_size)
        segment2 = self.agent_profile.agent_experience.sample_segment(segment_size)
        return segment1, segment2

    def stop(self):
        self.running = False
        self.thread.join()
        self.app.stop_runner(self.token)


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
        experience_batch_size = self.agent_profile.config['EXPERIENCE_BATCH_SIZE']
        reward_batch_size = self.agent_profile.config['REWARD_BATCH_SIZE']

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
