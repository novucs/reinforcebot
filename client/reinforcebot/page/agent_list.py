import math
import re
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

        self.builder.get_object('page-start-button') \
            .connect("clicked", lambda *_: self.change_page(1), None)
        self.builder.get_object('page-left-button') \
            .connect("clicked", lambda *_: self.change_page(self.page - 1), None)
        self.builder.get_object('page-right-button') \
            .connect("clicked", lambda *_: self.change_page(self.page + 1), None)
        self.builder.get_object('page-end-button') \
            .connect("clicked", lambda *_: self.change_page(self.end_page), None)

        for i in range(1, 6):
            def clicked_callback(idx):
                return lambda *_: self.on_agent_detail_clicked(idx)

            self.builder.get_object(f'result{i}-button') \
                .connect("clicked", clicked_callback(i), None)

        self.window = self.builder.get_object("agentlist")
        self.window.set_title("ReinforceBot - Agent List")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

        self.page = 1
        self.end_page = 1
        self.page_size = 5
        self.search = ""
        self.results = None
        self.authors = None
        self.query_pool = ThreadPoolExecutor(8)

    def present(self):
        self.fetch()
        self.render()
        self.window.present()

    def fetch(self):
        query = f'agents/?page={self.page}&page_size={self.page_size}'
        search = re.sub(r'([^\s\w]|_)+', '', self.search)
        if search:
            query += f'search="{self.search}"'

        self.results = requests.get(API_URL + query).json()
        self.authors = self.query_pool.map(
            lambda a: requests.get(API_URL + f'users/{a["author"]}/').json(), self.results['results'])
        self.end_page = math.ceil(self.results['count'] / self.page_size)

    def render(self):
        for idx, (agent, author) in enumerate(zip(self.results['results'], self.authors)):
            idx += 1
            name = f'{author["username"]}/{agent["name"]}'
            description = agent["description"].strip().split('\n')[0]
            if len(description) > 64:
                description = description[:60] + '...'

            self.builder.get_object(f'result{idx}').show()
            self.builder.get_object(f'result{idx}-button').show()
            self.builder.get_object(f'result{idx}-name').set_text(name)
            self.builder.get_object(f'result{idx}-description').set_text(description)

        for idx in range(len(self.results['results']) + 1, self.page_size + 1):
            self.builder.get_object(f'result{idx}').hide()
            self.builder.get_object(f'result{idx}-button').hide()

        previous_page = '.' if self.results['previous'] is None else str(self.page - 1)
        next_page = '.' if self.results['next'] is None else str(self.page + 1)
        self.builder.get_object('page-left-button').set_label(previous_page)
        self.builder.get_object('page-middle-button').set_label(str(self.page))
        self.builder.get_object('page-right-button').set_label(next_page)

    def change_page(self, page):
        if page < 1 or self.end_page < page or self.page == page:
            return
        self.page = page
        self.fetch()
        self.render()

    def on_create_clicked(self):
        print('Create')

    def on_agent_detail_clicked(self, idx):
        self.window.hide()
        self.router.route('agent_detail', agent=self.results['results'][idx - 1])
