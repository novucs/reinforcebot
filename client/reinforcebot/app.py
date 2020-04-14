import json

from reinforcebot.config import SESSION_FILE
from reinforcebot.router import PageRouter


class App:
    def __init__(self, builder):
        self.builder = builder
        self.router = PageRouter(self)
        self.signed_in = False
        self.jwt_access = None
        self.jwt_refresh = None

    def sign_in(self):
        with open(SESSION_FILE, 'r') as session:
            jwt = json.load(session)

        if 'refresh' not in jwt:
            return False

        self.signed_in = True
        self.jwt_access = jwt['access']
        self.jwt_refresh = jwt['refresh']
        return True

    def start(self):
        self.router.setup()
        self.router.route('agent_list' if self.sign_in() else 'sign_in')
