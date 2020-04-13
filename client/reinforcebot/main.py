from threading import Thread

import cairo
import gi
from torchvision.transforms.functional import resize

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk

from reinforcebot import screen, reward
from reinforcebot.agent import Agent
from reinforcebot.config import ENSEMBLE_SIZE, FRAME_DISPLAY_SIZE, FRAME_SIZE, OBSERVATION_SPACE
from reinforcebot.experience import record_new_user_experience, handover_control, record_user_experience
from reinforcebot.messaging import notify
from reinforcebot.replay_buffer import ExperienceReplayBuffer, RewardReplayBuffer


# handover control F1
# toggle training F2
# reward shaping F3
# stop recording F4

class AgentConfig:
    def __init__(self):
        self.initialised = False
        self.agent = None
        self.action_mapping = None
        self.user_experience = None
        self.agent_experience = None
        self.reward_ensemble = None
        self.reward_buffer = None

    def load_initial_user_experience(self, action_mapping, user_experience):
        if self.initialised:
            raise ValueError('Agents cannot redefine their action space, a new agent must be created instead')

        self.action_mapping = action_mapping
        self.user_experience = user_experience
        self.agent_experience = ExperienceReplayBuffer(OBSERVATION_SPACE)
        self.reward_ensemble = reward.Ensemble(OBSERVATION_SPACE, len(self.action_mapping), ENSEMBLE_SIZE)
        self.reward_buffer = RewardReplayBuffer(OBSERVATION_SPACE)
        self.agent = Agent(OBSERVATION_SPACE, len(action_mapping))
        self.initialised = True


class App:
    def __init__(self, builder, window):
        self.builder = builder
        self.window = window
        self.screen_recorder = screen.Recorder()
        self.agent_config = AgentConfig()

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
            if not self.agent_config.initialised:
                record_new_user_experience(self.screen_recorder, self.agent_config)
                notify('Successfully saved user experience with new action set')
            else:
                record_user_experience(self.screen_recorder, self.agent_config)
                notify('Successfully saved user experience')

        thread = Thread(target=record)
        thread.start()

    def on_handover_control_clicked(self):
        if not self.screen_recorder.running:
            notify('You  must select an area of your screen to record')
            return

        if not self.agent_config.initialised:
            notify('You must record experience yourself to let the agent know what buttons to press')
            return

        def control():
            notify('Your agent is now controlling the keyboard. Press ESC to stop.')
            handover_control(self.screen_recorder, self.agent_config)
            notify('Agent control has been lifted')

        thread = Thread(target=control)
        thread.start()


def main():
    builder = Gtk.Builder()
    builder.add_from_file("main.glade")
    window = builder.get_object("detail")
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
