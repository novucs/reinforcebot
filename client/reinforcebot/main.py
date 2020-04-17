from reinforcebot.resources import glade


def main():
    # Torch monkeypatch - Fixes shipping TorchVision > 0.2.2 with PyInstaller
    def script_method(fn, _rcb=None):
        return fn

    def script(obj, optimize=True, _frames_up=0, _rcb=None):
        return obj

    import torch.jit

    torch.jit.script_method = script_method
    torch.jit.script = script

    # Setup GTK
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gtk
    builder = Gtk.Builder()
    builder.add_from_file(glade())

    # Start the app
    from reinforcebot.app import App
    app = App(builder)
    app.start()
    Gtk.main()


if __name__ == '__main__':
    main()
