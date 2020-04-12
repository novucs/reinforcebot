import itertools
import time

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
from torchvision.transforms.functional import resize

from reinforcebot.experience_replay_buffer import DynamicExperienceReplayBuffer
from reinforcebot.messaging import notify

FRAME_SIZE = (80, 80)
OBSERVATION_SPACE = (2, *FRAME_SIZE)


class KeyboardBuffer:
    def __init__(self):
        self.keyboard_listener = keyboard.Listener(
            on_press=lambda key: self.on_press(key),
            on_release=lambda key: self.on_release(key),
        )
        self.pressed_keys = set()
        self.released_keys = set()

    def on_press(self, key):
        self.pressed_keys.add(key.vk if isinstance(key, keyboard.KeyCode) else key.value.vk)

    def on_release(self, key):
        self.released_keys.add(key.vk if isinstance(key, keyboard.KeyCode) else key.value.vk)

    def stop(self):
        self.keyboard_listener.stop()

    def start(self):
        self.keyboard_listener.start()

    def read(self):
        keys = self.pressed_keys.copy()
        self.pressed_keys -= self.released_keys
        return keys


def convert_frame(frame):
    frame = frame.convert('L')
    frame = resize(frame, FRAME_SIZE)
    return np.array(frame)


def record_new_user_experience(screen_recorder):
    notify('Recording has begun. Press ESC to stop.')

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    previous_frame = np.zeros(FRAME_SIZE)
    buffer = DynamicExperienceReplayBuffer(OBSERVATION_SPACE)

    while True:
        frame = convert_frame(screen_recorder.screenshot())
        observation = np.stack((previous_frame, frame))
        time.sleep(0.1)
        action = keyboard_recorder.read()

        if Key.esc.value.vk in action:
            break

        action -= {Key.esc.value.vk, *range(Key.f1.value.vk, Key.f20.value.vk)}
        next_frame = convert_frame(screen_recorder.screenshot())
        next_observation = np.stack((frame, next_frame))
        buffer.write(observation, action, next_observation)
        previous_frame = frame

    keyboard_recorder.stop()
    action_mapping, buffer = buffer.build()
    notify('Successfully saved user experience with new action set')
    return action_mapping, buffer


def record_user_experience(screen_recorder, action_mapping, buffer):
    notify('Recording has begun. Press ESC to stop.')

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    previous_frame = np.zeros(FRAME_SIZE)
    allowed_keys = set(itertools.chain(*action_mapping.values()))

    while True:
        frame = convert_frame(screen_recorder.screenshot())
        observation = np.stack((previous_frame, frame))
        time.sleep(0.1)
        keys = keyboard_recorder.read()

        if Key.esc.value.vk in keys:
            break

        keys -= {Key.esc.value.vk, *range(Key.f1.value.vk, Key.f20.value.vk)}
        keys &= allowed_keys

        if keys not in action_mapping.values():
            keys = next(iter(keys))

        action = next((a for a, k in action_mapping.items() if keys == k), 0)

        next_frame = convert_frame(screen_recorder.screenshot())
        next_observation = np.stack((frame, next_frame))
        buffer.write(observation, action, next_observation)
        previous_frame = frame

    keyboard_recorder.stop()
    notify('Successfully saved user experience')


def handover_control(screen_recorder, action_mapping):
    pass
