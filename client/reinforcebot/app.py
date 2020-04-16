import json
import os

import requests

from reinforcebot.config import SESSION_FILE, API_URL
from reinforcebot.messaging import notify
from reinforcebot.router import PageRouter


class App:
    def __init__(self, builder):
        self.builder = builder
        self.router = PageRouter(self)
        self.user = None
        self.signed_in = False
        self.jwt_access = None
        self.jwt_refresh = None

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
        self.router.setup()
        self.router.route('agent_list' if self.sign_in() else 'sign_in')

    def start_runner(self, agent_id):
        response = self.authorised_fetch(
            lambda headers: requests.post(API_URL + 'runners/', json={'agent_id': agent_id}, headers=headers))
        if response.status_code != 200:
            notify(response.json()['detail'])
            return None
        return response.json()['token']

    def fetch_runner_parameters(self, token):
        response = self.authorised_fetch(lambda headers: requests.get(API_URL + f'runners/{token}/', headers=headers))
        if response.status_code != 200:
            notify(response.json()['detail'])
            return None
        return response.json()['parameters']

    def add_runner_experience(self, token, experience):
        response = self.authorised_fetch(
            lambda headers: requests.post(API_URL + f'runners/{token}/experience/', json=experience, headers=headers))
        if response.status_code != 200:
            notify(response.json()['detail'])
            return None
        return response.json()['parameters']

    def stop_runner(self, token):
        response = self.authorised_fetch(
            lambda headers: requests.delete(API_URL + f'runners/{token}/', headers=headers))
        if response.status_code != 200:
            notify(response.json()['detail'])
            return None
        return response.json()['parameters']
