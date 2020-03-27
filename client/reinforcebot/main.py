import array
import subprocess
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cairo
import cv2
import gi
import mss
import numpy as np
import pytesseract
from PIL import (
    Image,
    # ImageTk,
    ImageChops,
)
from pynput import (
    keyboard,
    mouse,
)

gi.require_version("Gtk", "3.0")
from gi.repository import (
    Gtk,
    Gdk,
    GLib,
)


def on_scroll(x, y, dx, dy):
    pass


def on_move(x, y):
    pass


class App:
    def __init__(self, builder):
        self.builder = builder

        self.image_on_canvas = None
        self.select_area_enabled = False
        self.select_window_enabled = False
        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=lambda *kwargs: self.on_click(*kwargs),
            on_scroll=on_scroll,
        )
        self.mouse_listener.start()

        self.keyboard_listener = keyboard.Listener(
            on_press=lambda *kwargs: self.on_press(*kwargs),
            on_release=lambda *kwargs: self.on_release(*kwargs),
        )
        self.keyboard_listener.start()
        self.select_window_enabled = False
        self.select_area_enabled = False
        self.current_coordinates = 0, 0, 0, 0
        self.area_select_start = 0, 0
        self.last_window_capture = 0

    def on_press(self, key):
        pass

    def on_release(self, key):
        pass

    def on_mfile_item_clicked(self, itemid):
        if itemid == 'mfile_open':
            messagebox.showinfo('File', 'You clicked Open menuitem')

        if itemid == 'mfile_quit':
            self.quit()

    def on_about_clicked(self):
        messagebox.showinfo('About', 'You clicked About menuitem')

    def on_select_window_clicked(self):
        self.select_window_enabled = True

    def on_select_area_clicked(self):
        self.select_area_enabled = True
        # self.capture(0, 0, 200, 200)

    def on_coordinates_change(self):
        ox, oy, ow, oh = self.current_coordinates
        x, y, w, h = map(lambda c: int(c.get_text()), self.coordinate_variables())
        w = max(0, w + ox - x)
        h = max(0, h + oy - y)
        self.set_displayed_coordinates(x, y, w, h)
        self.capture(x, y, w, h)

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
        def redraw(widget, context):
            context.set_source_surface(self.image, 0, 0)
            context.paint()
            return False

        # self.builder.get_object('preview').connect('draw', redraw)

        pixbuf = Gdk.pixbuf_get_from_surface(self.image, x, y, width, height)
        self.builder.get_object('preview').set_from_pixbuf(pixbuf)

        # if self.image_on_canvas is None:
        #     self.canvas.grid(row=0, column=3, rowspan=3)
        #     self.image_on_canvas = self.canvas.create_image(
        #         0, 0, anchor=tk.NW, image=self.image)
        # else:
        #     self.canvas.itemconfig(self.image_on_canvas, image=self.image)

    def capture_image(self, x, y, width, height, captured_at):
        if self.last_window_capture > captured_at:
            return

        with mss.mss() as sct:
            try:
                i = sct.grab({'top': y, 'left': x, 'width': width, 'height': height})
            except mss.exception.ScreenShotError:
                print('Window is off screen')
                return

        np_image = np.array(i, dtype=np.uint8)

        # self.canvas.config(width=width, height=height)  # todo: idk if needed in update

        # size = tuple(x // 2 for x in i.size)  # half the size of the displayed image
        size = tuple(x for x in i.size)  # half the size of the displayed image
        res = cv2.resize(np_image, dsize=size)
        # inner_image = Image.frombytes("RGB", size, res, "raw", "BGRX")
        inner_image = Image.frombytes("RGBX", size, res, "raw", "RGBX")
        # inner_image = Image.frombytes("RGBA", size, res)
        inner_image.thumbnail((256, 256))
        byte_array = array.array('B', inner_image.tobytes())
        smaller_width, smaller_height = inner_image.size

        # image_size = inner_image.size
        thumb = inner_image.crop((0, 0, smaller_width, smaller_height))
        offset_x = int(max((width - smaller_width) / 2, 0))
        offset_y = int(max((height - smaller_height) / 2, 0))
        thumb = ImageChops.offset(thumb, offset_x, offset_y)
        byte_array = array.array('B', thumb.tobytes())

        # print(len(byte_array), inner_image.size)
        # cairo_surface = cairo.ImageSurface.create_for_data(byte_array, cairo.FORMAT_ARGB32, width, height, width * 4)
        cairo_surface = cairo.ImageSurface.create_for_data(byte_array, cairo.FORMAT_RGB24, smaller_width,
                                                           smaller_height)
        self.image = cairo_surface
        # self.image = ImageTk.PhotoImage(image=inner_image)

        # print("text found:", pytesseract.image_to_string(
        #     Image.frombytes("RGB", i.size, i.bgra, "raw", "BGRX"),
        #     lang='eng',
        #     config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
        # ))
        self.set_image(0, 0, smaller_width, smaller_height)

        # todo: schedule via gtk
        # self.master.after(1, lambda: self.capture_image(x, y, width, height, captured_at))
        GLib.idle_add(self.capture_image, x, y, width, height, captured_at)

    def on_click(self, mouse_x, mouse_y, button, pressed):
        if self.select_window_enabled:
            if pressed:
                return

            try:
                ox, oy, ow, oh = self.parse_window_capture()
            except ValueError:
                return
        elif self.select_area_enabled:
            if pressed:
                # todo: disable mouse interactions here, and hide main window.
                self.area_select_start = (mouse_x, mouse_y)
                return

            ox = min(self.area_select_start[0], mouse_x)
            oy = min(self.area_select_start[1], mouse_y)
            ow = max(self.area_select_start[0], mouse_x) - ox
            oh = max(self.area_select_start[1], mouse_y) - oy

            if ow == 0 or oh == 0:
                return

            self.select_area_enabled = False
        else:
            return

        # handle out of bounds negative x/y
        width = min(ow + ox, ow)
        height = min(oh + oy, oh)
        x = max(0, ox)
        y = max(0, oy)

        # handle out of bounds positive x/y
        width = min(self.builder.get_object('window1').get_screen().get_width() - x, width)
        height = min(self.builder.get_object('window1').get_screen().get_height() - y, height)

        if ox != x or oy != y or ow != width or oh != height:
            print('Warning: captured screen is not entirely visible')

        self.capture(x, y, width, height)

    def parse_window_capture(self):
        cmd = f"wnckprop --xid=$(xdotool getmouselocation --shell | grep WINDOW | sed 's/WINDOW=//') | grep 'Geometry (x, y, width, height):'"
        ps = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        geometry = output.decode('utf-8').split(':')[1].strip().split(', ')[-4:]
        ox, oy, ow, oh = map(int, geometry)
        return ox, oy, ow, oh

    def capture(self, x, y, width, height):
        print('Captured screen coordinates:', x, y, width, height)
        self.last_window_capture = datetime.now()
        self.current_coordinates = x, y, width, height

        # todo: maybe schedule this after time again?
        # self.master.after(0, lambda: self.capture_image(
        #     x, y, width, height, self.last_window_capture))
        self.capture_image(
            x, y, width, height, self.last_window_capture)

        self.set_displayed_coordinates(x, y, width, height)
        self.select_window_enabled = False


def numbify(widget):
    def filter_numbers(entry, *args):
        text = entry.get_text().strip()
        entry.set_text(''.join([i for i in text if i in '0123456789']))

    widget.connect('changed', filter_numbers)


if __name__ == '__main__':
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    window = builder.get_object("window1")
    window.set_title("reinforcebot")
    window.connect("destroy", Gtk.main_quit)
    window.show_all()

    app = App(builder)

    numbify(builder.get_object('x'))
    numbify(builder.get_object('y'))
    numbify(builder.get_object('width'))
    numbify(builder.get_object('height'))

    builder.get_object('select-area-button') \
        .connect("clicked", lambda widget, event: app.on_select_area_clicked(), None)
    builder.get_object('select-window-button') \
        .connect("clicked", lambda widget, event: app.on_select_window_clicked(), None)


    def record_button_callback(widget, event):
        print('print called')


    # button.override_color(Gtk.StateType.NORMAL, Gdk.color_parse('dark_red'))
    Gtk.main()
