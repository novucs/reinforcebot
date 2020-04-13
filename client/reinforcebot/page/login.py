from gi.repository import Gtk

from reinforcebot.page import AgentDetailPage


class LoginPage:
    def __init__(self, builder):
        self.builder = builder
        self.builder.get_object('signin-button') \
            .connect("clicked", lambda *_: self.on_signin_clicked(), None)
        self.builder.get_object('offline-button') \
            .connect("clicked", lambda *_: self.on_continue_offline_clicked(), None)

        self.window = self.builder.get_object("signin")
        self.window.set_title("ReinforceBot - Sign In")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.show_all()

    def on_signin_clicked(self):
        print('Login')

    def on_continue_offline_clicked(self):
        self.window.hide()
        AgentDetailPage(self.builder)
