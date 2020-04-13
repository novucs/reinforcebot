import itertools
import time
from threading import Thread

import numpy as np
import torch
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from torchvision.transforms.functional import resize

from reinforcebot.config import FRAME_SIZE, OBSERVATION_SPACE, SEGMENT_SIZE, STEP_SECONDS, \
    UPDATE_TARGET_PARAMETERS_STEPS
from reinforcebot.replay_buffer import DynamicExperienceReplayBuffer


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


def record_new_user_experience(screen_recorder, agent_profile):
    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    buffer = DynamicExperienceReplayBuffer(OBSERVATION_SPACE)
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.cache)

    while True:
        observation = np.stack((previous_frame, frame))
        time.sleep(STEP_SECONDS)
        keys = keyboard_recorder.read()

        if Key.esc.value.vk in keys:
            break

        keys -= {Key.esc.value.vk, *range(Key.f1.value.vk, Key.f20.value.vk)}

        next_frame = convert_frame(screen_recorder.cache)
        next_observation = np.stack((frame, next_frame))
        buffer.write(observation, keys, next_observation)

        previous_frame, frame = frame, next_frame

    keyboard_recorder.stop()
    agent_profile.load_initial_user_experience(*buffer.build())


def record_user_experience(screen_recorder, agent_profile):
    action_mapping, buffer = agent_profile.action_mapping, agent_profile.user_experience

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    allowed_keys = set(itertools.chain(*action_mapping.values()))
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.cache)

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

        next_frame = convert_frame(screen_recorder.cache)
        next_observation = np.stack((frame, next_frame))
        buffer.write(observation, action, next_observation)

        previous_frame, frame = frame, next_frame

    keyboard_recorder.stop()


def handover_control(screen_recorder, agent_profile, choose_preference):
    agent, action_mapping, experience_buffer, reward_ensemble, reward_buffer = \
        agent_profile.agent, agent_profile.action_mapping, agent_profile.agent_experience, \
        agent_profile.reward_ensemble, agent_profile.reward_buffer

    keyboard_recorder = KeyboardBuffer()
    keyboard_recorder.start()
    controller = keyboard.Controller()
    pressed_keys = set()
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.cache)
    step = 0
    running = True

    def train():
        while running:
            if experience_buffer.size < SEGMENT_SIZE:
                time.sleep(1)
                continue

            o, a, n = experience_buffer.read()

            with torch.no_grad():
                r = reward_ensemble.predict(o, a).numpy()

            d = np.zeros(a.shape, dtype=np.float32)
            agent.train((o, a, r, n, d))

            if reward_buffer.size > 0:
                s1, s2, p = reward_buffer.read()
                reward_ensemble.train(s1, s2, p)

    train_thread = Thread(target=train)
    train_thread.start()
    step_start = time.time()

    while True:
        user_pressed_keys = keyboard_recorder.read()
        if Key.esc.value.vk in user_pressed_keys:
            running = False
            break

        if Key.f3.value.vk in user_pressed_keys:
            choose_preference()
            keyboard_recorder.read()

        step += 1
        observation = np.stack((previous_frame, frame))
        action = agent.act(observation)

        released_keys = pressed_keys - action_mapping[action]
        pressed_keys = action_mapping[action]

        for key in released_keys:
            controller.release(KeyCode.from_vk(key))

        for key in pressed_keys:
            controller.press(KeyCode.from_vk(key))

        time.sleep(max(0, STEP_SECONDS - (time.time() - step_start)))
        step_start = time.time()

        next_frame = convert_frame(screen_recorder.cache)
        next_observation = np.stack((frame, next_frame))
        experience_buffer.write(observation, action, next_observation)
        previous_frame, frame = frame, next_frame

        if step % UPDATE_TARGET_PARAMETERS_STEPS == 0:
            agent.critic_target.load_state_dict(agent.critic.state_dict())

    for key in pressed_keys:
        controller.release(KeyCode.from_vk(key))

    keyboard_recorder.stop()
