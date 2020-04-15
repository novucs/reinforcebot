import math
import re
import time
from concurrent.futures.thread import ThreadPoolExecutor

import requests
from gi.repository import GLib, Gtk

from reinforcebot.agent_profile import AgentProfile
from reinforcebot.config import API_URL
from reinforcebot.messaging import alert


class AgentListPage:
    def __init__(self, app):
        self.app = app
        self.builder = app.builder
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

        self.builder.get_object('search') \
            .connect("changed", lambda *_: self.perform_search(), None)

        for i in range(1, 6):
            def clicked_callback(idx):
                return lambda *_: self.on_agent_detail_clicked(idx)

            self.builder.get_object(f'result{i}-button') \
                .connect("clicked", clicked_callback(i), None)

        self.window = self.builder.get_object("agentlist")
        self.window.set_title("ReinforceBot - Agent List")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

        def pulse_progress_bar(_):
            self.builder.get_object('agent-list-progress-bar').pulse()
            return True

        self.timeout_id = GLib.timeout_add(50, pulse_progress_bar, None)

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
            query += f'&search="{self.search}"'

        results = self.app.authorised_fetch(lambda h: requests.get(API_URL + query, headers=h))
        if results is None:
            def sign_out():
                self.window.hide()
                self.app.router.route('sign_in')
            GLib.idle_add(sign_out)
            return

        self.results = results.json()
        self.authors = list(self.query_pool.map(
            lambda a: requests.get(API_URL + f'users/{a["author"]}/').json(), self.results['results']))
        self.end_page = math.ceil(self.results['count'] / self.page_size)

    def render(self):
        if self.results is None:
            return

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

        if len(self.results['results']) == 0:
            self.builder.get_object('no-results-found').show()
        else:
            self.builder.get_object('no-results-found').hide()

        previous_page = '.' if self.results['previous'] is None else str(self.page - 1)
        next_page = '.' if self.results['next'] is None else str(self.page + 1)
        self.builder.get_object('page-left-button').set_label(previous_page)
        self.builder.get_object('page-middle-button').set_label(str(self.page))
        self.builder.get_object('page-right-button').set_label(next_page)

    def change_page(self, page):
        if page < 1 or self.end_page < page or self.page == page:
            return
        self.show_progress_bar(True)
        self.page = page

        def change_inner():
            self.fetch()
            GLib.idle_add(self.render)
            self.show_progress_bar(False)

        self.query_pool.submit(change_inner)

    def perform_search(self):
        self.page = 1
        search = self.builder.get_object('search').get_text()
        self.search = search
        self.show_progress_bar(True)

        def debounce():
            time.sleep(0.2)
            if self.search == search:
                self.fetch()
                self.show_progress_bar(False)
                GLib.idle_add(self.render)

        self.query_pool.submit(debounce)

    def on_create_clicked(self):
        self.window.hide()
        self.app.router.route('create_agent')

    def on_agent_detail_clicked(self, idx):
        agent = self.results['results'][idx - 1]
        self.show_progress_bar(True)

        def download():
            def finished():
                self.show_progress_bar(False)
                self.window.hide()
                self.app.router.route('agent_detail', agent_profile=agent_profile)

            def failed():
                self.show_progress_bar(False)
                alert(self.window, f'Cannot open {agent["name"]}. It may be a corrupt save.')

            try:
                agent_profile = AgentProfile.download(self.app, agent['id'])
            except Exception as e:
                print('Failed to download: ', e)
                GLib.idle_add(failed)
                return
            GLib.idle_add(finished)

        self.query_pool.submit(download)

    def show_progress_bar(self, show=True):
        def perform():
            progress_bar = self.builder.get_object('agent-list-progress-bar')
            if show:
                progress_bar.show()
            else:
                progress_bar.hide()

        GLib.idle_add(perform)
