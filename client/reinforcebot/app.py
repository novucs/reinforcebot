import json
import os

import requests
from gi.repository import Gtk

from reinforcebot.config import API_URL, CONFIG, SESSION_FILE
from reinforcebot.keyboard_recorder import KeyboardRecorder
from reinforcebot.messaging import alert, notify
from reinforcebot.router import PageRouter


class App:
    def __init__(self, builder):
        self.builder = builder
        self.router = PageRouter(self)
        self.user = None
        self.signed_in = False
        self.jwt_access = None
        self.jwt_refresh = None
        self.tokens = set()
        self.keyboard_recorder = KeyboardRecorder()

    def authorised_fetch(self, callback):
        try:
            if not self.signed_in:
                return callback({})

            response = callback({'Authorization': f'JWT {self.jwt_access}'})

            if response.status_code == 401:
                response = requests.post(API_URL + 'auth/jwt/refresh/', json={'refresh': self.jwt_refresh})
                if response.status_code != 200:
                    return None
                self.jwt_access = response.json()['access']
                response = callback({'Authorization': f'JWT {self.jwt_access}'})
        except:
            notify('Unable to connect to online services')
            return None
        return response

    def sign_in(self):
        if not os.path.exists(SESSION_FILE):
            return False

        with open(SESSION_FILE, 'r') as session:
            jwt = json.load(session)

        if 'refresh' not in jwt:
            return False

        self.signed_in = True
        self.jwt_access = jwt['access']
        self.jwt_refresh = jwt['refresh']
        response = self.authorised_fetch(lambda h: requests.get(API_URL + 'auth/users/me/', headers=h))

        if response is None:
            self.signed_in = False
            return False

        self.user = response.json()
        return True

    def start(self):
        self.keyboard_recorder.start()
        self.router.setup()
        self.router.route('agent_list' if self.sign_in() else 'sign_in')

    def stop(self):
        for token in list(*self.tokens):
            self.stop_runner(token)
        Gtk.main_quit()
        self.keyboard_recorder.stop()

    def start_runner(self, agent_profile):
        response = self.authorised_fetch(
            lambda headers: requests.post(
                API_URL + 'runners/',
                json={
                    'agent_id': agent_profile.agent_id,
                    'parameters': agent_profile.agent.dump_parameters(),
                    'config': CONFIG,
                    'action_mapping': {k: list(v) for k, v in agent_profile.action_mapping.items()},
                },
                headers=headers,
            ))

        if response.status_code != 200:
            alert(self.router.current_page.window, response.json()['detail'])
            return None
        token = response.json()['token']
        self.tokens.add(token)
        return token

    def fetch_runner_parameters(self, token):
        response = self.authorised_fetch(lambda headers: requests.get(API_URL + f'runners/{token}/', headers=headers))
        if response.status_code != 200:
            return None
        return response.json()['parameters']

    def add_runner_experience(self, token, experience):
        response = self.authorised_fetch(
            lambda headers: requests.post(API_URL + f'runners/{token}/experience/', json=experience, headers=headers))
        if response.status_code != 200:
            return None
        return response.json()['parameters']

    def stop_runner(self, token):
        self.tokens.remove(token)
        response = self.authorised_fetch(
            lambda headers: requests.delete(API_URL + f'runners/{token}/', headers=headers))
        if response.status_code != 200:
            return None
        if 400 <= response.status_code < 500:
            alert(self.router.current_page.window, response.json()['detail'])
            return None
        return response.json()['parameters']

    def upload_model(self, agent_id, path):
        response = self.authorised_fetch(lambda headers: requests.patch(
            API_URL + f'agents/{agent_id}/',
            files={'parameters': (os.path.basename(path), open(path, 'rb'))},
            data={'changeReason': 'Updated parameters'},
            headers=headers,
        ))

        if response.status_code != 200:
            notify('Agent parameters upload failed.')
        else:
            notify('Successfully uploaded agent parameters.')
