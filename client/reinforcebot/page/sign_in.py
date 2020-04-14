import json

import requests
from gi.repository import Gtk

from reinforcebot.config import SESSION_FILE
from reinforcebot.messaging import notify


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
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_sign_in_clicked(self):
        username = self.builder.get_object('username').get_text()
        password = self.builder.get_object('password').get_text()
        jwt = requests.post('https://reinforcebot.novucs.net/api/auth/jwt/create/',
                            json={'username': username, 'password': password}).json()
        if 'access' not in jwt or 'refresh' not in jwt:
            notify('No active account found with the given credentials')
            return

        with open(SESSION_FILE, 'w') as session:
            json.dump(jwt, session)

        self.app.sign_in()
        self.app.router.route('agent_list')

    def on_continue_offline_clicked(self):
        self.window.hide()
        self.app.router.route('agent_list')
