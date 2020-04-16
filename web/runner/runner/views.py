import json
import time
from threading import Thread

import numpy as np
import requests
from django.http import HttpResponse, JsonResponse
from reinforcebotagent.agent import Agent
from reinforcebotagent.agent_profile import AgentProfile
from reinforcebotagent.trainer import LocalTrainer

from runner.settings import API_URL, RUNNER_KEY, SESSION_LIMIT


class Session:
    def __init__(self, runner, session_id, data):
        self.runner = runner
        self.session_id = session_id
        self.running = True
        self.thread = None

        self.token = data['token']
        self.config = data['config']
        self.action_mapping = {k: set(v) for k, v in data['action_mapping'].items()}

        self.profile = AgentProfile(self.config)
        self.profile.initialised = True
        self.profile.agent = Agent(self.config['OBSERVATION_SPACE'], len(self.action_mapping))
        self.profile.action_mapping = self.action_mapping
        self.profile.agent.load_parameters(np.array(data['parameters']))
        self.profile.create_buffers()

        self.trainer = LocalTrainer(self.profile)

    def run(self):
        self.trainer.start()

        while self.running:
            start = time.time()
            time.sleep(1)
            time_elapsed = time.time() - start
            response = requests.put(API_URL + f'credits/{self.token}/', json={
                'runner_key': RUNNER_KEY,
                'used': (time_elapsed / 60) / 60,
            })

            if response.json().get('cancel', False):
                self.running = False

        self.runner.session_ended(self.session_id)
        self.trainer.stop()

    def load_experience(self, experience):
        parsed = {}

        if 'agent_transition' in experience:
            transition = experience['agent_transition']
            observation = np.array(transition['observation'])
            action = np.array(transition['action'])
            next_observation = np.array(transition['next_observation'])
            parsed['agent_transition'] = (observation, action, next_observation)

        if 'user_transition' in experience:
            transition = experience['user_transition']
            observation = np.array(transition['observation'])
            action = np.array(transition['action'])
            next_observation = np.array(transition['next_observation'])
            parsed['user_transition'] = (observation, action, next_observation)

        if 'rewards' in experience:
            rewards = experience['rewards']
            segment1 = np.array(rewards['segment1'])
            segment2 = np.array(rewards['segment2'])
            preference = rewards['preference']
            parsed['rewards'] = (segment1, segment2, preference)

        self.trainer.experience(parsed)

    def dump_parameters(self):
        return self.profile.agent.dump_parameters()


class Runner:
    def __init__(self):
        self.sessions = {}
        self.index = 0

    def is_available(self):
        return len(self.sessions) < SESSION_LIMIT

    def start(self, data):
        if not self.is_available():
            return None
        session_id = self.index = self.index + 1
        session = Session(self, session_id, data)
        self.sessions[session_id] = session
        session.thread = Thread(target=session.run)
        session.thread.start()
        return session_id

    def session_ended(self, session_id):
        del self.sessions[session_id]

    def stop(self, session_id):
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        session.running = False
        return session.dump_parameters()


_runner = Runner()


def handle_sessions(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = _runner.start(data)
        if session_id is None:
            return JsonResponse({'detail': 'Session limit reached'}, status=429)
        return JsonResponse({'session_id': session_id})

    return HttpResponse(status=400)


def handle_session(request, session_id):
    session = _runner.sessions.get(session_id)
    if session is None:
        return HttpResponse(status=404)

    if request.method == 'GET':
        return JsonResponse({'parameters': session.dump_parameters()})

    elif request.method == 'DELETE':
        _runner.stop(session_id)
        return JsonResponse({'parameters': session.dump_parameters()})

    return HttpResponse(status=400)


def handle_session_experience(request, session_id):
    session = _runner.sessions.get(session_id)
    if session is None:
        return HttpResponse(status=404)

    if request.method == 'POST':
        experience = json.loads(request.body)
        session.load_experience(experience)
        return JsonResponse({'parameters': session.dump_parameters()})

    return HttpResponse(status=400)
