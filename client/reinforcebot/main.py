import subprocess
from datetime import datetime

import cairo
import gi
import gym as gym
import mss
from PIL import Image
from pynput import keyboard, mouse

from reinforcebot import agent

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

state = None


def limit_bounds(ox, oy, ow, oh):
    # handle out of bounds negative x/y
    width = min(ow + ox, ow)
    height = min(oh + oy, oh)
    x = max(0, ox)
    y = max(0, oy)

    # handle out of bounds positive x/y
    geometry = Gdk.get_default_root_window().get_geometry()
    width = min(geometry.width - x, width)
    height = min(geometry.height - y, height)

    if ox != x or oy != y or ow != width or oh != height:
        print('Warning: The window captured is not entirely visible')

    return x, y, width, height


def select_window(callback):
    def on_click(mouse_x, mouse_y, button, pressed):
        if pressed:
            return

        subprocess.Popen(('gnome-screenshot', '-c', '-w'))
        geometry = subprocess.check_output(
            f"wnckprop --xid=$(xdotool getmouselocation --shell | sed -n 's/WINDOW=//p') | sed -n 's/Geometry (x, y, width, height): //p'",
            shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        x, y, width, height = map(int, geometry.split(', '))
        x, y, width, height = limit_bounds(x, y, width, height)
        callback(x, y, width, height)
        mouse_listener.stop()

    mouse_listener = mouse.Listener(on_click=lambda *kwargs: on_click(*kwargs))
    mouse_listener.start()


def select_area(callback):
    subprocess.Popen(('gnome-screenshot', '-c', '-a'))

    def on_click(mouse_x, mouse_y, button, pressed):
        if pressed:
            origin[0] = mouse_x
            origin[1] = mouse_y
            return

        x = min(origin[0], mouse_x)
        y = min(origin[1], mouse_y)
        width = max(origin[0], mouse_x) - x
        height = max(origin[1], mouse_y) - y

        if width == 0 or height == 0:
            return

        x, y, width, height = limit_bounds(x, y, width, height)
        callback(x, y, width, height)
        mouse_listener.stop()

    origin = [0, 0]  # the position of the cursor when the left mouse button is pressed
    mouse_listener = mouse.Listener(on_click=lambda *kwargs: on_click(*kwargs))
    mouse_listener.start()


class App:
    def __init__(self, builder, window):
        self.builder = builder
        self.window = window

        self.image_on_canvas = None
        self.keyboard_listener = keyboard.Listener(
            on_press=lambda *kwargs: self.on_press(*kwargs),
            on_release=lambda *kwargs: self.on_release(*kwargs),
        )
        self.keyboard_listener.start()
        self.current_coordinates = 0, 0, 0, 0
        self.last_window_capture = 0

    def on_press(self, key):
        pass

    def on_release(self, key):
        pass

    def on_select_window_clicked(self):
        self.window.iconify()
        select_window(lambda x, y, width, height: self.capture(x, y, width, height))

    def on_select_area_clicked(self):
        self.window.iconify()
        select_area(lambda x, y, width, height: self.capture(x, y, width, height))

    def on_record_clicked(self):
        agent.main('reinforcebot-v0')

    def set_displayed_coordinates(self, x, y, width, height):
        components = self.coordinate_variables()
        for component, value in zip(components, (x, y, width, height)):
            component.set_text(str(value))

    def coordinate_variables(self):
        return [
            self.builder.get_object(v)
            for v in ('x', 'y', 'width', 'height')
        ]

    def set_image(self, x, y, width, height):
        pixbuf = Gdk.pixbuf_get_from_surface(self.image, x, y, width, height)
        self.builder.get_object('preview').set_from_pixbuf(pixbuf)

    def capture_image(self, x, y, width, height, captured_at):
        if self.last_window_capture > captured_at:
            return

        with mss.mss() as sct:
            try:
                sct_img = sct.grab({'top': y, 'left': x, 'width': width, 'height': height})
            except mss.exception.ScreenShotError:
                print('Window is off screen')
                return

        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img.thumbnail((256, 256))
        img.putalpha(256)
        global state
        state = img
        surface = cairo.ImageSurface.create_for_data(
            bytearray(img.tobytes('raw', 'BGRa')), cairo.FORMAT_RGB24, img.width, img.height)
        self.image = surface
        self.set_image(0, 0, img.width, img.height)
        GLib.idle_add(self.capture_image, x, y, width, height, captured_at)

    def capture(self, x, y, width, height):
        self.window.present()
        print('Captured screen coordinates:', x, y, width, height)
        self.last_window_capture = datetime.now()
        self.current_coordinates = x, y, width, height
        self.capture_image(x, y, width, height, self.last_window_capture)
        self.set_displayed_coordinates(x, y, width, height)
        self.select_window_enabled = False
        self.window.show()


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
