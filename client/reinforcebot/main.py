import gi


def main():
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gtk
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    from reinforcebot.app import App
    app = App(builder)
    app.start()
    Gtk.main()


if __name__ == '__main__':
    main()
