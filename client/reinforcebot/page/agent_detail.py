import time
from threading import Lock, Thread

import cairo
from gi.repository import Gdk, GLib, Gtk
from torchvision.transforms.functional import resize

from reinforcebot import screen
from reinforcebot.config import BASE_URL, FRAME_DISPLAY_SIZE, FRAME_SIZE
from reinforcebot.experience import handover_control, record_new_user_experience, record_user_experience
from reinforcebot.human_preference_chooser import HumanPreferenceChooser
from reinforcebot.messaging import alert, notify


# handover control F1
# toggle training F2
# reward shaping F3
# stop recording F4


class AgentDetailPage:
    def __init__(self, app):
        print('Created agent detail page')
        self.app = app
        self.builder = app.builder
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
        self.builder.get_object('cloud-compute-button') \
            .connect("notify::active", lambda *_: self.on_use_cloud_compute_clicked(), None)

        self.window = self.builder.get_object("detail")
        self.window.set_title("ReinforceBot - Agent Detail")
        self.window.connect("destroy", Gtk.main_quit)
        self.window.set_position(Gtk.WindowPosition.CENTER)

        self.screen_recorder = screen.Recorder()
        self.agent_profile = None
        self.control_lock = Lock()
        self.recording = False
        self.using_cloud_compute = False

    def present(self, agent_profile):
        self.agent_profile = agent_profile
        description = '\n'.join(self.agent_profile.description.strip().split('\n')[:16])
        if len(description) > 256:
            description = description[:256].strip() + '...'

        self.builder.get_object('agent-name-label').set_text(self.agent_profile.name)
        self.builder.get_object('agent-description-label').set_text(description)

        self.builder.get_object('read-more-label').hide()
        self.builder.get_object('agent-link-label').hide()
        self.builder.get_object('cloud-compute-box').hide()
        self.builder.get_object('cloud-compute-button').set_active(self.using_cloud_compute)

        if self.agent_profile.agent_id:
            link = BASE_URL + f'agent/{self.agent_profile.agent_id}/'
            self.builder.get_object('agent-link-label').set_label(link)
            self.builder.get_object('agent-link-label').set_uri(link)
            self.builder.get_object('read-more-label').show()
            self.builder.get_object('agent-link-label').show()

            if self.app.signed_in:
                self.builder.get_object('cloud-compute-box').show()

        self.window.present()

    def on_agent_list_clicked(self):
        self.window.hide()
        self.app.router.route('agent_list')

    def on_select_window_clicked(self):
        if self.recording:
            alert(self.window, 'You cannot change the recorded area while recording')
            return

        self.window.hide()
        screen.select_window(lambda *coordinates: self.capture(*coordinates))

    def on_select_area_clicked(self):
        if self.recording:
            alert(self.window, 'You cannot change the recorded area while recording')
            return

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
        if self.recording:
            alert(self.window, 'Experience is already being recorded')
            return

        if not self.screen_recorder.running:
            alert(self.window, 'You must select an area of your screen to record')
            return

        def record():
            self.control_lock.acquire()
            self.recording = True
            notify('Recording has begun. Press ESC to stop.')
            if not self.agent_profile.initialised:
                record_new_user_experience(self.screen_recorder, self.agent_profile)
                GLib.idle_add(lambda: self.window.show())
                alert(self.window, 'Successfully saved user experience with new action set')
            else:
                record_user_experience(self.screen_recorder, self.agent_profile)
                GLib.idle_add(lambda: self.window.show())
                alert(self.window, 'Successfully saved user experience')
            self.recording = False
            self.agent_profile.save()
            self.control_lock.release()

        self.window.hide()
        thread = Thread(target=record)
        thread.start()

    def on_handover_control_clicked(self):
        if self.recording:
            alert(self.window, 'Experience is already being recorded')
            return

        if not self.screen_recorder.running:
            alert(self.window, 'You must select an area of your screen to record')
            return

        if not self.agent_profile.initialised:
            alert(self.window, 'You must record experience yourself to let the agent know what buttons to press')
            return

        def control():
            self.control_lock.acquire()
            self.recording = True
            time.sleep(1)
            notify('Your agent is now controlling the keyboard. Press ESC to stop. Press F3 to manage rewards.')
            self.agent_profile.loading_lock.acquire()  # wait until agent config has loaded
            self.agent_profile.loading_lock.release()
            handover_control(self.screen_recorder, self.agent_profile, self.open_preference_chooser)
            GLib.idle_add(lambda: self.window.show())
            alert(self.window, 'Agent control has been lifted')
            self.recording = False
            self.agent_profile.save()
            self.control_lock.release()

        self.window.hide()
        thread = Thread(target=control)
        thread.start()

    def on_use_cloud_compute_clicked(self):
        if self.recording:
            self.builder.get_object('cloud-compute-button').set_active(self.using_cloud_compute)
            alert(self.window, 'Cannot switch compute runner while recording')
            return
        self.using_cloud_compute = not self.using_cloud_compute

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
