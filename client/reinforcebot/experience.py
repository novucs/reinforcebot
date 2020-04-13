import itertools
import time

import numpy as np
import torch
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from torchvision.transforms.functional import resize

from reinforcebot import reward
from reinforcebot.agent import Agent
from reinforcebot.config import ENSEMBLE_SIZE, FRAME_SIZE, OBSERVATION_SPACE, SEGMENT_SIZE, STEP_SECONDS, \
    UPDATE_TARGET_PARAMETERS_STEPS
from reinforcebot.messaging import notify
from reinforcebot.replay_buffer import DynamicExperienceReplayBuffer, RewardReplayBuffer


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
    buffer = DynamicExperienceReplayBuffer(OBSERVATION_SPACE)
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.screenshot())

    while True:
        observation = np.stack((previous_frame, frame))
        time.sleep(STEP_SECONDS)
        keys = keyboard_recorder.read()

        if Key.esc.value.vk in keys:
            break

        keys -= {Key.esc.value.vk, *range(Key.f1.value.vk, Key.f20.value.vk)}

        next_frame = convert_frame(screen_recorder.screenshot())
        next_observation = np.stack((frame, next_frame))
        buffer.write(observation, keys, next_observation)

        previous_frame, frame = frame, next_frame

    keyboard_recorder.stop()
    action_mapping, buffer = buffer.build()
    notify('Successfully saved user experience with new action set')
    return action_mapping, buffer


def record_user_experience(screen_recorder, action_mapping, buffer):
    notify('Recording has begun. Press ESC to stop.')

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    allowed_keys = set(itertools.chain(*action_mapping.values()))
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.screenshot())

    while True:
        observation = np.stack((previous_frame, frame))
        time.sleep(STEP_SECONDS)
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

        previous_frame, frame = frame, next_frame

    keyboard_recorder.stop()
    notify('Successfully saved user experience')


def handover_control(screen_recorder, action_mapping, experience_buffer):
    notify('Your agent is now controlling the keyboard. Press ESC to stop.')

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    agent = Agent(OBSERVATION_SPACE, len(action_mapping))
    controller = keyboard.Controller()
    pressed_keys = set()
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.screenshot())
    ensemble = reward.Ensemble(OBSERVATION_SPACE, len(action_mapping), ENSEMBLE_SIZE)
    reward_buffer = RewardReplayBuffer(OBSERVATION_SPACE)
    step = 0

    while True:
        if Key.esc.value.vk in keyboard_recorder.read():
            break

        step += 1
        observation = np.stack((previous_frame, frame))
        action = agent.act(observation)

        released_keys = pressed_keys - action_mapping[action]
        pressed_keys = action_mapping[action]

        for key in released_keys:
            controller.release(KeyCode.from_vk(key))

        for key in pressed_keys:
            controller.press(KeyCode.from_vk(key))

        time.sleep(STEP_SECONDS)

        next_frame = convert_frame(screen_recorder.screenshot())
        next_observation = np.stack((frame, next_frame))
        experience_buffer.write(observation, action, next_observation)

        previous_frame, frame = frame, next_frame

        o, a, n = experience_buffer.read()
        with torch.no_grad():
            r = ensemble.predict(o, a).numpy()
        d = np.zeros(a.shape, dtype=np.float32)
        agent.train((o, a, r, n, d))

        if experience_buffer.size > SEGMENT_SIZE:
            # TODO: Switch out dummy reward buffer sampling with user controlled
            #       sampling
            s1 = experience_buffer.sample_segment()
            s2 = experience_buffer.sample_segment()

            reward_buffer.write(s1, s2, 1)

            s1, s2, p = reward_buffer.read()
            ensemble.train(s1, s2, p)

        if step % UPDATE_TARGET_PARAMETERS_STEPS == 0:
            agent.critic_target.load_state_dict(agent.critic.state_dict())

    for key in pressed_keys:
        controller.release(KeyCode.from_vk(key))

    keyboard_recorder.stop()
    notify('Agent control has been lifted')
