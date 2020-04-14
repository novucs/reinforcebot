import subprocess

from gi.repository import Gtk


class AlertDialog(Gtk.Dialog):
    def __init__(self, parent, message):
        Gtk.Dialog.__init__(self, "Alert", parent, 0, (Gtk.STOCK_OK, Gtk.ResponseType.OK))
        self.set_default_size(150, 100)
        label = Gtk.Label(message)
        box = self.get_content_area()
        box.add(label)
        self.show_all()


def notify(message):
    subprocess.Popen(('notify-send', '--hint', 'int:transient:1', 'ReinforceBot', message))


def alert(parent_window, message):
    dialog = AlertDialog(parent_window, message)
    response = dialog.run()
    dialog.destroy()
    return response
