from gi.repository import Gtk


class AgentListPage:
    def __init__(self, router, builder):
        self.router = router
        self.builder = builder
        self.builder.get_object('create-button') \
            .connect("clicked", lambda *_: self.on_create_clicked(), None)
        self.builder.get_object('agent-detail-button-1') \
            .connect("clicked", lambda *_: self.on_agent_detail_clicked(), None)

        self.window = self.builder.get_object("agentlist")
        self.window.set_title("ReinforceBot - Agent List")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_create_clicked(self):
        print('Create')

    def on_agent_detail_clicked(self):
        self.window.hide()
        self.router.route('agent_detail')
