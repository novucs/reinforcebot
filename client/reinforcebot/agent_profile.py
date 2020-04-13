from threading import Lock

from reinforcebot import reward
from reinforcebot.agent import Agent
from reinforcebot.config import ENSEMBLE_SIZE, OBSERVATION_SPACE
from reinforcebot.replay_buffer import ExperienceReplayBuffer, RewardReplayBuffer


class AgentProfile:
    def __init__(self):
        self.initialised = False
        self.loading_lock = Lock()
        self.agent = None
        self.action_mapping = None
        self.user_experience = None
        self.agent_experience = None
        self.reward_ensemble = None
        self.reward_buffer = None

    def load_initial_user_experience(self, action_mapping, user_experience):
        if self.initialised:
            raise ValueError('Agents cannot redefine their action space, a new agent must be created instead')

        self.loading_lock.acquire()
        self.initialised = True
        self.action_mapping = action_mapping
        self.user_experience = user_experience
        self.agent_experience = ExperienceReplayBuffer(OBSERVATION_SPACE)
        self.reward_ensemble = reward.Ensemble(OBSERVATION_SPACE, len(self.action_mapping), ENSEMBLE_SIZE)
        self.reward_buffer = RewardReplayBuffer(OBSERVATION_SPACE)
        self.agent = Agent(OBSERVATION_SPACE, len(action_mapping))
        self.loading_lock.release()
