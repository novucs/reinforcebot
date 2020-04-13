import subprocess

import mss
from gi.repository import Gdk, GLib
from PIL import Image
from pynput import mouse

from reinforcebot.messaging import notify


class Recorder:
    def __init__(self):
        self.coordinates = {}
        self.callback = None
        self.running = False
        self.cache = None

    def stop(self):
        self.running = False

    def start(self, x, y, width, height, callback):
        self.callback = callback
        self.coordinates = {
            'left': x,
            'top': y,
            'width': width,
            'height': height,
        }

        if not self.running:
            self.running = True
            self._record()

    def _record(self):
        if not self.running:
            return False

        self.callback(self.screenshot())
        GLib.idle_add(self._record)
        return False

    def screenshot(self):
        with mss.mss() as sct:
            try:
                screenshot = sct.grab(self.coordinates)
            except mss.exception.ScreenShotError:
                raise ValueError(f'Window is off screen. Coordinates: {self.coordinates}')
        self.cache = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        return self.cache


def _limit_bounds(ox, oy, ow, oh):
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
        notify('Warning: The window captured is not entirely visible')

    return x, y, width, height


def select_window(callback):
    notify('Click a window you wish to record')

    def on_click(mouse_x, mouse_y, button, pressed):
        if pressed:
            return

        subprocess.Popen(('gnome-screenshot', '-c', '-w'))
        geometry = subprocess.check_output(
            f"wnckprop --xid=$(xdotool getmouselocation --shell | sed -n 's/WINDOW=//p') | sed -n 's/Geometry (x, y, width, height): //p'",
            shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        x, y, width, height = map(int, geometry.split(', '))
        x, y, width, height = _limit_bounds(x, y, width, height)
        callback(x, y, width, height)
        mouse_listener.stop()

    mouse_listener = mouse.Listener(on_click=lambda *kwargs: on_click(*kwargs))
    mouse_listener.start()


def select_area(callback):
    notify('Drag an area on your screen')
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

        x, y, width, height = _limit_bounds(x, y, width, height)
        callback(x, y, width, height)
        mouse_listener.stop()

    origin = [0, 0]  # the position of the cursor when the left mouse button is pressed
    mouse_listener = mouse.Listener(on_click=lambda *kwargs: on_click(*kwargs))
    mouse_listener.start()
