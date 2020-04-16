import json

import requests
from gi.repository import Gtk

from reinforcebot.config import API_URL, SESSION_FILE
from reinforcebot.messaging import alert


class SignInPage:
    def __init__(self, app):
        self.app = app
        self.builder = app.builder
        self.builder.get_object('signin-button') \
            .connect("clicked", lambda *_: self.on_sign_in_clicked(), None)
        self.builder.get_object('offline-button') \
            .connect("clicked", lambda *_: self.on_continue_offline_clicked(), None)

        self.window = self.builder.get_object("signin")
        self.window.set_title("ReinforceBot - Sign In")
        self.window.connect("destroy", lambda *_: self.app.stop)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_sign_in_clicked(self):
        username = self.builder.get_object('username').get_text()
        password = self.builder.get_object('password').get_text()
        try:
            jwt = requests.post(API_URL + 'auth/jwt/create/',
                                json={'username': username, 'password': password}).json()
        except:
            alert(self.window, 'Unable to connect to online services')
            return

        if 'access' not in jwt or 'refresh' not in jwt:
            alert(self.window, 'No active account found with the given credentials')
            return

        with open(SESSION_FILE, 'w') as session:
            json.dump(jwt, session, indent=2)

        self.window.hide()
        self.app.sign_in()
        self.app.router.route('agent_list')

    def on_continue_offline_clicked(self):
        self.window.hide()
        self.app.router.route('agent_list')
