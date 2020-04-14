from threading import Lock, Thread

import cairo
from gi.repository import Gdk, GLib, Gtk
from torchvision.transforms.functional import resize

from reinforcebot import screen
from reinforcebot.agent_profile import AgentProfile
from reinforcebot.config import FRAME_DISPLAY_SIZE, FRAME_SIZE
from reinforcebot.experience import handover_control, record_new_user_experience, record_user_experience
from reinforcebot.human_preference_chooser import HumanPreferenceChooser
from reinforcebot.messaging import notify
# handover control F1
# toggle training F2
# reward shaping F3
# stop recording F4
from reinforcebot.page import agent_list


class AgentDetailPage:
    def __init__(self, router, builder):
        self.router = router
        self.builder = builder
        self.builder.get_object('back-to-agent-listing-button') \
            .connect("clicked", lambda *_: self.on_agent_list_clicked(), None)
        self.builder.get_object('select-area-button') \
            .connect("clicked", lambda *_: self.on_select_area_clicked(), None)
        self.builder.get_object('select-window-button') \
            .connect("clicked", lambda *_: self.on_select_window_clicked(), None)
        self.builder.get_object('record-button') \
            .connect("clicked", lambda *_: self.on_record_clicked(), None)
        self.builder.get_object('handover-control-button') \
            .connect("clicked", lambda *_: self.on_handover_control_clicked(), None)

        self.window = self.builder.get_object("detail")
        self.window.set_title("ReinforceBot - Agent Detail")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

        self.screen_recorder = screen.Recorder()
        self.agent_profile = AgentProfile()

    def present(self):
        self.window.present()

    def on_agent_list_clicked(self):
        self.window.hide()
        self.router.route('agent_list')

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
            notify('Recording has begun. Press ESC to stop.')
            if not self.agent_profile.initialised:
                record_new_user_experience(self.screen_recorder, self.agent_profile)
                notify('Successfully saved user experience with new action set')
            else:
                record_user_experience(self.screen_recorder, self.agent_profile)
                notify('Successfully saved user experience')

        thread = Thread(target=record)
        thread.start()

    def on_handover_control_clicked(self):
        if not self.screen_recorder.running:
            notify('You  must select an area of your screen to record')
            return

        if not self.agent_profile.initialised:
            notify('You must record experience yourself to let the agent know what buttons to press')
            return

        def control():
            notify('Your agent is now controlling the keyboard. Press ESC to stop. Press F3 to manage rewards.')
            self.agent_profile.loading_lock.acquire()  # wait until agent config has loaded
            self.agent_profile.loading_lock.release()
            handover_control(self.screen_recorder, self.agent_profile, self.open_preference_chooser)
            notify('Agent control has been lifted')

        thread = Thread(target=control)
        thread.start()

    def open_preference_chooser(self):
        init_lock = Lock()
        done_lock = Lock()

        def open_chooser():
            HumanPreferenceChooser(self.builder, self.agent_profile, done_lock)
            init_lock.release()
            return False

        init_lock.acquire()
        GLib.idle_add(open_chooser)

        init_lock.acquire()  # wait until human preference chooser has initialised (acquires done_lock)
        init_lock.release()

        done_lock.acquire()  # wait until human has made a choice
        done_lock.release()
