from concurrent.futures.thread import ThreadPoolExecutor

import requests
from gi.repository import Gtk

from reinforcebot.config import API_URL


class AgentListPage:
    def __init__(self, router, builder):
        self.router = router
        self.builder = builder
        self.builder.get_object('create-button') \
            .connect("clicked", lambda *_: self.on_create_clicked(), None)

        for i in range(1, 6):
            def clicked_callback(idx):
                return lambda *_: self.on_agent_detail_clicked(idx)
            self.builder.get_object(f'result{i}-button') \
                .connect("clicked", clicked_callback(i), None)

        self.window = self.builder.get_object("agentlist")
        self.window.set_title("ReinforceBot - Agent List")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

        self.results = None
        self.query_pool = ThreadPoolExecutor(8)

    def present(self):
        self.results = requests.get(API_URL + 'agents/?page=1&page_size=5').json()
        authors = self.query_pool.map(
            lambda a: requests.get(API_URL + f'users/{a["author"]}/').json(), self.results['results'])

        for idx, (agent, author) in enumerate(zip(self.results['results'], authors)):
            idx += 1
            name = f'{author["username"]}/{agent["name"]}'
            description = agent["description"].strip().split('\n')[0]
            if len(description) > 64:
                description = description[:60] + '...'

            self.builder.get_object(f'result{idx}-name').set_text(name)
            self.builder.get_object(f'result{idx}-description').set_text(description)

        self.window.present()

    def on_create_clicked(self):
        print('Create')

    def on_agent_detail_clicked(self, idx):
        self.window.hide()
        self.router.route('agent_detail', agent=self.results['results'][idx - 1])
