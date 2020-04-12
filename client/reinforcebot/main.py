from threading import Thread

import cairo
import gi
from torchvision.transforms.functional import resize

from reinforcebot.config import FRAME_DISPLAY_SIZE, FRAME_SIZE, OBSERVATION_SPACE
from reinforcebot.experience_replay_buffer import ExperienceReplayBuffer
from reinforcebot.messaging import notify

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk

from reinforcebot import screen
from reinforcebot.experience import record_new_user_experience, handover_control, record_user_experience


# handover control F1
# toggle training F2
# reward shaping F3
# stop recording F4

class App:
    def __init__(self, builder, window):
        self.builder = builder
        self.window = window
        self.screen_recorder = screen.Recorder()
        self.action_mapping = None
        self.user_experience = None
        self.agent_experience = None

    def on_select_window_clicked(self):
        self.window.hide()
        screen.select_window(lambda *coordinates: self.capture(*coordinates))

    def on_select_area_clicked(self):
        self.window.hide()
        screen.select_area(lambda *coordinates: self.capture(*coordinates))

    def capture(self, x, y, width, height):
        print('Captured screen coordinates:', x, y, width, height)
        self.screen_recorder.start(x, y, width, height, lambda image: self.set_preview(image))
        self.window.present()

    def set_preview(self, image):
        image = image.convert('L').convert('RGBA')
        image = resize(image, FRAME_SIZE)
        image = resize(image, FRAME_DISPLAY_SIZE)
        data = memoryview(bytearray(image.tobytes('raw', 'BGRa')))
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, image.width, image.height)
        pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, image.width, image.height)
        self.builder.get_object('preview').set_from_pixbuf(pixbuf)

    def on_record_clicked(self):
        if not self.screen_recorder.running:
            notify('You must select an area of your screen to record')
            return

        def record():
            if self.action_mapping is None:
                action_mapping, user_experience = record_new_user_experience(self.screen_recorder)
                self.action_mapping = action_mapping
                self.user_experience = user_experience
            else:
                record_user_experience(self.screen_recorder, self.action_mapping, self.user_experience)

        thread = Thread(target=record)
        thread.start()

    def on_handover_control_clicked(self):
        if not self.screen_recorder.running:
            notify('You  must select an area of your screen to record')
            return

        if not self.action_mapping:
            notify('You must record experience yourself to let the agent know what buttons to press')
            return

        def control():
            if self.agent_experience is None:
                self.agent_experience = ExperienceReplayBuffer(OBSERVATION_SPACE)

            handover_control(self.screen_recorder, self.action_mapping, self.agent_experience)

        thread = Thread(target=control)
        thread.start()


def main():
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    window = builder.get_object("window1")
    window.set_title("reinforcebot")
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    app = App(builder, window)
    builder.get_object('select-area-button').connect("clicked", lambda *_: app.on_select_area_clicked(), None)
    builder.get_object('select-window-button').connect("clicked", lambda *_: app.on_select_window_clicked(), None)
    builder.get_object('record-button').connect("clicked", lambda *_: app.on_record_clicked(), None)
    builder.get_object('handover-control-button').connect("clicked", lambda *_: app.on_handover_control_clicked(), None)
    Gtk.main()


if __name__ == '__main__':
    main()
