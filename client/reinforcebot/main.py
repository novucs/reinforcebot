import cairo
import gi
import gym as gym
from pynput import keyboard

from reinforcebot import agent, screen

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk

state = None


class KeyboardRecorder:
    def __init__(self):
        self.keyboard_listener = keyboard.Listener(
            on_press=lambda key: self.on_press(key),
            on_release=lambda key: self.on_release(key),
        )
        self.pressed_keys = set()
        self.released_keys = set()

    def on_press(self, key):
        pass

    def on_release(self, key):
        pass

    def stop(self):
        self.keyboard_listener.stop()

    def start(self):
        self.keyboard_listener.start()

    def read(self):
        pass


class App:
    def __init__(self, builder, window):
        self.builder = builder
        self.window = window
        self.screen_recorder = screen.Recorder()
        self.keyboard_recorder = KeyboardRecorder()

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
        data = memoryview(bytearray(image.tobytes('raw', 'BGRa')))
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, image.width, image.height)
        pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, image.width, image.height)
        self.builder.get_object('preview').set_from_pixbuf(pixbuf)

    def on_record_clicked(self):
        self.keyboard_recorder.start()

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
