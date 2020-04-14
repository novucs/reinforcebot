from reinforcebot.page.agent_detail import AgentDetailPage
from reinforcebot.page.agent_list import AgentListPage
from reinforcebot.page.sign_in import SignInPage


class PageRouter:
    def __init__(self, builder):
        self.builder = builder
        self.pages = {
            'agent_detail': AgentDetailPage(self, builder),
            'agent_list': AgentListPage(self, builder),
            'sigin_in': SignInPage(self, builder),
        }
        self.current_page = None

    def route(self, page_name, **kwargs):
        self.current_page = self.pages[page_name]
        self.current_page.present(**kwargs)
