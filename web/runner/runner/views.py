import json
import time
from threading import Thread

import numpy as np
import requests
from django.http import HttpResponse, JsonResponse

from runner.settings import SESSION_LIMIT, API_URL, RUNNER_KEY


class Session:
    def __init__(self, runner, session_id, token):
        self.runner = runner
        self.session_id = session_id
        self.token = token
        self.running = True
        self.thread = None
        self.parameters = np.zeros((1, 2))

    def run(self):
        while self.running:
            start = time.time()
            time.sleep(1)
            time_elapsed = time.time() - start
            response = requests.put(API_URL + f'credits/{self.token}/', json={
                'runner_key': RUNNER_KEY,
                'used': (time_elapsed / 60) / 60,
            })

            if response.json()['cancel']:
                self.running = False

        self.runner.session_ended(self.session_id)

    def load_experience(self, experience):
        pass


class Runner:
    def __init__(self):
        self.sessions = {}
        self.index = 0

    def is_available(self):
        return len(self.sessions) < SESSION_LIMIT

    def start(self, token):
        if not self.is_available():
            return None
        session_id = self.index = self.index + 1
        session = Session(self, session_id, token)
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
        session.thread.join()
        return session.parameters


_runner = Runner()


def handle_sessions(request):
    if request.method == 'POST':
        token = json.loads(request.body)['token']
        session_id = _runner.start(token)
        if session_id is None:
            return JsonResponse({'detail': 'Session limit reached'}, status=429)
        return JsonResponse({'session_id': session_id})

    return HttpResponse(status=400)


def handle_session(request, session_id):
    session = _runner.sessions.get(session_id)
    if session is None:
        return HttpResponse(status=404)

    if request.method == 'GET':
        return JsonResponse({'parameters': session.parameters.tolist()})

    elif request.method == 'DELETE':
        _runner.stop(session_id)
        return JsonResponse({'parameters': session.parameters.tolist()})

    return HttpResponse(status=400)


def handle_session_experience(request, session_id):
    session = _runner.sessions.get(session_id)
    if session is None:
        return HttpResponse(status=404)

    if request.method == 'POST':
        experience = json.loads(request.body)
        session.load_experience(experience)
        return JsonResponse({'parameters': session.parameters.tolist()})

    return HttpResponse(status=400)
