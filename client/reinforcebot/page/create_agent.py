import os

import requests
from gi.repository import Gtk

from reinforcebot.agent_profile import AgentProfile
from reinforcebot.config import API_URL
from reinforcebot.messaging import alert, ask


class CreateAgentPage:
    def __init__(self, app):
        self.app = app
        self.builder = app.builder
        self.builder.get_object('create-agent-button') \
            .connect("clicked", lambda *_: self.on_create_agent_clicked(), None)
        self.builder.get_object('cancel-create-agent-button') \
            .connect("clicked", lambda *_: self.cancel(), None)

        self.window = self.builder.get_object("create-agent")
        self.window.set_title("ReinforceBot - Create Agent")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_create_agent_clicked(self):
        name = self.builder.get_object('agent-name-entry').get_text()
        description = self.builder.get_object('agent-description-entry').get_text()
        if name.strip() == "" or description.strip() == "":
            alert(self.window, 'Agents must have a name and a description')
            return

        agent_profile = AgentProfile()
        agent_profile.name = name
        agent_profile.description = description
        agent_profile.author = self.app.user

        if os.path.exists(agent_profile.agent_path(create=False)) and \
                not ask(self.window, 'This will overwrite an existing agent, are you sure?'):
            self.cancel()
            return

        backup_path = agent_profile.backup()

        if self.app.signed_in:
            response = self.app.authorised_fetch(lambda headers: requests.post(
                API_URL + 'agents/',
                files={'parameters': (os.path.basename(backup_path), open(backup_path, 'rb'))},
                data={'name': name, 'description': description, 'changeReason': 'Initial creation'},
                headers=headers,
            ))

            if response.status_code == 400:
                alert(self.window, 'Your account already has an agent by that name')
                return

            agent_profile.agent_id = response.json()['id']

        self.window.hide()
        self.app.router.route('agent_detail', agent_profile=agent_profile)

    def cancel(self):
        self.window.hide()
        self.app.router.route('agent_list')
