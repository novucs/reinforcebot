import subprocess

from gi.repository import GLib, Gtk


class AlertDialog(Gtk.Dialog):
    def __init__(self, parent, message):
        Gtk.Dialog.__init__(self, "Alert", parent, 0, (Gtk.STOCK_OK, Gtk.ResponseType.OK))
        self.set_default_size(150, 100)
        label = Gtk.Label(message)
        label.set_margin_top(16)
        label.set_margin_left(16)
        label.set_margin_right(16)
        box = self.get_content_area()
        box.add(label)
        self.show_all()


def notify(message):
    subprocess.Popen(('notify-send', '--hint', 'int:transient:1', 'ReinforceBot', message))


def alert(parent_window, message):
    def inner_alert():
        dialog = AlertDialog(parent_window, message)
        dialog.run()
        dialog.destroy()

    GLib.idle_add(inner_alert)
