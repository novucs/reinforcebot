import json
import os
import subprocess
from datetime import datetime
from threading import Lock

import requests

from reinforcebot import reward
from reinforcebot.agent import Agent
from reinforcebot.config import AGENTS_PATH, ENSEMBLE_SIZE, OBSERVATION_SPACE, API_URL, CACHE_PATH
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
        self.author = None
        self.agent_id = None
        self.name = None
        self.description = None

    def create_buffers(self):
        if self.user_experience is None:
            self.user_experience = ExperienceReplayBuffer(OBSERVATION_SPACE)
        self.agent_experience = ExperienceReplayBuffer(OBSERVATION_SPACE)
        self.reward_ensemble = reward.Ensemble(OBSERVATION_SPACE, len(self.action_mapping), ENSEMBLE_SIZE)
        self.reward_buffer = RewardReplayBuffer(OBSERVATION_SPACE)

    def load_initial_user_experience(self, action_mapping, user_experience):
        if self.initialised:
            raise ValueError('Agents cannot redefine their action space, a new agent must be created instead')

        self.loading_lock.acquire()
        self.initialised = True
        self.action_mapping = action_mapping
        self.user_experience = user_experience
        self.create_buffers()
        self.agent = Agent(OBSERVATION_SPACE, len(action_mapping))
        self.loading_lock.release()

    def agent_path(self):
        author_path = os.path.join(AGENTS_PATH, '_' if self.author is None else self.author['username'])
        path = os.path.join(author_path, self.name)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, buffers=False):
        path = self.agent_path()

        if self.author is not None:
            with open(os.path.join(path, '..', 'settings.json'), 'w') as settings_file:
                json.dump({'id': self.author['id']}, settings_file, indent=2)

        with open(os.path.join(path, 'settings.json'), 'w') as settings_file:
            json.dump({
                'initialised': self.initialised,
                'action_mapping': {k: list(v) for k, v in self.action_mapping.items()},
                'author': self.author,
                'agent_id': self.agent_id,
                'name': self.name,
                'description': self.description,
            }, settings_file, indent=2)

        if self.initialised:
            self.agent.save(os.path.join(path, 'parameters'))
            if buffers:
                buffers_path = os.path.join(path, 'buffers')
                os.makedirs(buffers_path, exist_ok=True)
                self.agent_experience.save(os.path.join(buffers_path, 'agent_experience'))
                self.user_experience.save(os.path.join(buffers_path, 'user_experience'))
                self.reward_buffer.save(os.path.join(buffers_path, 'rewards'))

    def backup(self):
        self.save(buffers=False)
        path = os.path.join(self.agent_path(), 'backups', f'{self.name}-{datetime.now().isoformat()}.tar.gz')
        subprocess.run(['tar', '-czf', str(path), 'parameters', 'settings.json'])

    @staticmethod
    def load(name, author=None):
        author_username = author['username'] if author else '_'
        agent_path = os.path.join(AGENTS_PATH, author_username, name)

        with open(os.path.join(agent_path, 'settings.json'), 'r') as settings_file:
            settings = json.load(settings_file)

        profile = AgentProfile()
        profile.initialised = settings['initialised']
        profile.author = settings['author']
        profile.agent_id = settings['agent_id']
        profile.name = settings['name']
        profile.description = settings['description']

        if profile.initialised:
            profile.action_mapping = {int(k): set(v) for k, v in settings['action_mapping'].items()}
            profile.agent = Agent.load(os.path.join(agent_path, 'parameters'))
            buffers_path = os.path.join(agent_path, 'buffers')
            if os.path.exists(buffers_path):
                profile.agent_experience = ExperienceReplayBuffer.load(os.path.join(buffers_path, 'agent_experience'))
                profile.user_experience = ExperienceReplayBuffer.load(os.path.join(buffers_path, 'user_experience'))
                profile.reward_buffer = RewardReplayBuffer.load(os.path.join(buffers_path, 'reward_buffer'))
            else:
                profile.create_buffers()
        return profile

    @staticmethod
    def download(agent_id):
        agent = requests.get(API_URL + f'agents/{agent_id}/').json()
        author = requests.get(API_URL + f'users/{agent["author"]}/').json()

        author_path = os.path.join(AGENTS_PATH, author['username'])
        agent_path = os.path.join(author_path, agent['name'])
        tarfile = os.path.join(CACHE_PATH, agent['name'] + '.tar.gz')

        os.makedirs(agent_path, exist_ok=True)
        with open(os.path.join(author_path, 'settings.json'), 'w') as settings_file:
            json.dump({'id': author['id']}, settings_file, indent=2)

        subprocess.run(['wget', agent['parameters'], '-O', tarfile])
        subprocess.run(['tar', '-xf', tarfile, '-C', str(agent_path)])

        with open(os.path.join(agent_path, 'settings.json'), 'r') as settings_file:
            agent_settings = json.load(settings_file)

        with open(os.path.join(agent_path, 'settings.json'), 'w') as settings_file:
            json.dump({
                **agent_settings,
                'author': author,
                'agent_id': agent_id,
                'name': agent['name'],
                'description': agent['description'],
            }, settings_file, indent=2)

        return AgentProfile.load(agent['name'], author)
