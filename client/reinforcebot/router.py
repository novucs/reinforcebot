from reinforcebot.page.agent_detail import AgentDetailPage
from reinforcebot.page.agent_list import AgentListPage
from reinforcebot.page.sign_in import SignInPage


class PageRouter:
    def __init__(self, app):
        self.app = app
        self.pages = {}
        self.current_page = None

    def setup(self):
        self.pages = {
            'agent_detail': AgentDetailPage(self.app),
            'agent_list': AgentListPage(self.app),
            'sigin_in': SignInPage(self.app),
        }

    def route(self, page_name, **kwargs):
        self.current_page = self.pages[page_name]
        self.current_page.present(**kwargs)
