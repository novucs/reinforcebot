import cairo
from gi.repository import Gdk, GLib, Gtk
from PIL import Image
from torchvision.transforms.functional import resize

from reinforcebot.config import FRAME_DISPLAY_SIZE, SEGMENT_SIZE


class HumanPreferenceChooser:
    def __init__(self, builder, agent_config, done_lock):
        self.done_lock = done_lock
        self.done_lock.acquire()
        self.running = True
        self.frame_id = 0
        self.agent_config = agent_config
        self.segment1 = self.agent_config.agent_experience.sample_segment()
        self.segment2 = self.agent_config.agent_experience.sample_segment()

        self.builder = builder
        self.builder.get_object('button-video1') \
            .connect("clicked", lambda *_: self.on_chosen_preference(1.0), None)
        self.builder.get_object('button-same') \
            .connect("clicked", lambda *_: self.on_chosen_preference(0.5), None)
        self.builder.get_object('button-video2') \
            .connect("clicked", lambda *_: self.on_chosen_preference(0.0), None)

        self.window = self.builder.get_object("preference")
        self.window.set_title("reinforcebot")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.present()

        self.render_segments()

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.done_lock.release()

    def on_chosen_preference(self, preference):
        self.agent_config.reward_buffer.write(self.segment1, self.segment2, preference)
        self.stop()
        self.window.hide()

    def render_segments(self):
        if not self.running:
            return False

        self.set_preview(1, Image.fromarray(self.segment1[0][self.frame_id][0]))
        self.set_preview(2, Image.fromarray(self.segment2[0][self.frame_id][0]))
        self.frame_id = (self.frame_id + 1) % SEGMENT_SIZE
        GLib.idle_add(self.render_segments)
        return False

    def set_preview(self, video, image):
        image = resize(image.convert('RGBA'), FRAME_DISPLAY_SIZE)
        data = memoryview(bytearray(image.tobytes('raw', 'BGRa')))
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, image.width, image.height)
        pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, image.width, image.height)
        self.builder.get_object(f'preview-video{video}').set_from_pixbuf(pixbuf)