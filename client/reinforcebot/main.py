import gi
from gi.repository import Gtk


# handover control F1
# toggle training F2
# reward shaping F3
# stop recording F4


def main():
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    from reinforcebot.page.agent_detail import AgentDetailPage
    AgentDetailPage(builder)
    Gtk.main()


if __name__ == '__main__':
    main()
