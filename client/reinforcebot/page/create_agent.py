import json

import requests
from gi.repository import Gtk

from reinforcebot.agent_profile import AgentProfile
from reinforcebot.config import SESSION_FILE
from reinforcebot.messaging import notify


class CreateAgentPage:
    def __init__(self, app):
        self.app = app
        self.builder = app.builder
        self.builder.get_object('create-agent-button') \
            .connect("clicked", lambda *_: self.on_create_agent_clicked(), None)
        self.builder.get_object('cancel-create-agent-button') \
            .connect("clicked", lambda *_: self.on_cancel_clicked(), None)

        self.window = self.builder.get_object("create-agent")
        self.window.set_title("ReinforceBot - Create Agent")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_create_agent_clicked(self):
        name = self.builder.get_object('agent-name-entry').get_text()
        description = self.builder.get_object('agent-description-entry').get_text()
        agent_profile = AgentProfile()
        response = requests.post(
            'https://reinforcebot.novucs.net/api/agents/',
            files={
                'parameters': (
                    'parameters.xls',
                    open('report.xls', 'rb'),
                    'application/vnd.ms-excel',
                    {'Expires': '0'},
                )
            }
        )

    def on_cancel_clicked(self):
        self.window.hide()
        self.app.router.route('agent_list')
