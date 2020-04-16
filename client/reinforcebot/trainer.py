import time
from threading import Thread


class CloudComputeTrainer:
    def __init__(self, app, agent_profile):
        self.app = app
        self.agent_profile = agent_profile
        self.token = None
        self.running = False
        self.thread = None

    def start(self):
        self.token = self.app.start_runner(self.agent_profile)
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
