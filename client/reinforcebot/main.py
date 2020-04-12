from threading import Thread

import cairo
import gi
import gym as gym

from reinforcebot.messaging import notify

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk

from reinforcebot import screen
from reinforcebot.experience import record_user_experience, handover_control

state = None


class App:
    def __init__(self, builder, window):
        self.builder = builder
        self.window = window
        self.screen_recorder = screen.Recorder()
        self.action_mapping = None
        self.user_experience = None

    def on_select_window_clicked(self):
        self.window.hide()
        screen.select_window(lambda *coordinates: self.capture(*coordinates))

    def on_select_area_clicked(self):
        self.window.hide()
        screen.select_area(lambda *coordinates: self.capture(*coordinates))

    def capture(self, x, y, width, height):
        print('Captured screen coordinates:', x, y, width, height)
        self.screen_recorder.start(x, y, width, height, lambda image: self.set_preview(image))
        self.set_displayed_coordinates(x, y, width, height)
        self.window.present()

    def set_preview(self, image):
        image.thumbnail((256, 256))
        image.putalpha(256)
        data = memoryview(bytearray(image.tobytes('raw', 'BGRa')))
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, image.width, image.height)
        pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, image.width, image.height)
        self.builder.get_object('preview').set_from_pixbuf(pixbuf)

    def on_record_clicked(self):
        if not self.screen_recorder.running:
            notify('You select an area of your screen to record')
            return

        def record():
            action_mapping, user_experience = record_user_experience(self.screen_recorder)
            self.action_mapping = action_mapping
            self.user_experience = user_experience

        thread = Thread(target=record)
        thread.start()

    def on_handover_control_clicked(self):
        if not self.screen_recorder.running:
            notify('You select an area of your screen to record')
            return

        if not self.action_mapping:
            notify('You must record experience yourself to let the agent know what buttons to press')
            return

        thread = Thread(target=lambda: handover_control(self.screen_recorder, self.action_mapping))
        thread.start()

    def set_displayed_coordinates(self, x, y, width, height):
        components = [self.builder.get_object(v) for v in ('x', 'y', 'width', 'height')]
        for component, value in zip(components, (x, y, width, height)):
            component.set_text(str(value))


def numbify(widget):
    def filter_numbers(entry, *args):
        text = entry.get_text().strip()
        entry.set_text(''.join([i for i in text if i in '0123456789']))

    widget.connect('changed', filter_numbers)


def main():
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    window = builder.get_object("window1")
    window.set_title("reinforcebot")
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    app = App(builder, window)
    numbify(builder.get_object('x'))
    numbify(builder.get_object('y'))
    numbify(builder.get_object('width'))
    numbify(builder.get_object('height'))
    builder.get_object('select-area-button').connect("clicked", lambda *_: app.on_select_area_clicked(), None)
    builder.get_object('select-window-button').connect("clicked", lambda *_: app.on_select_window_clicked(), None)
    builder.get_object('record-button').connect("clicked", lambda *_: app.on_record_clicked(), None)
    builder.get_object('handover-control-button').connect("clicked", lambda *_: app.on_handover_control_clicked(), None)
    Gtk.main()


class ReinforceBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ...

    def step(self, action):
        print('stepping')
        global state
        # self.step_callback(action)
        return state

    def reset(self):
        global state
        print('resetting')
        return state

    def render(self, mode='human'):
        print('rendering')

    def close(self):
        exit(0)


if __name__ == '__main__':
    gym.envs.register(
        id='reinforcebot-v0',
        entry_point='reinforcebot.main:ReinforceBotEnv',
    )
    main()
