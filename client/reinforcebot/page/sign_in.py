from gi.repository import Gtk


class SignInPage:
    def __init__(self, router, builder):
        self.router = router
        self.builder = builder
        self.builder.get_object('signin-button') \
            .connect("clicked", lambda *_: self.on_signin_clicked(), None)
        self.builder.get_object('offline-button') \
            .connect("clicked", lambda *_: self.on_continue_offline_clicked(), None)

        self.window = self.builder.get_object("signin")
        self.window.set_title("ReinforceBot - Sign In")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

    def present(self):
        self.window.present()

    def on_signin_clicked(self):
        print('sign in')

    def on_continue_offline_clicked(self):
        self.window.hide()
        self.router.route('agent_list')
