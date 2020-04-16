import itertools
import time

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from reinforcebotagent.replay_buffer import DynamicExperienceReplayBuffer
from torchvision.transforms.functional import resize

from reinforcebot.config import EXPERIENCE_BUFFER_SIZE, FRAME_SIZE, OBSERVATION_SPACE, STEP_SECONDS, \
    UPDATE_TARGET_PARAMETERS_STEPS


def convert_frame(frame):
    frame = frame.convert('L')
    frame = resize(frame, FRAME_SIZE)
    return np.array(frame)


def record_new_user_experience(screen_recorder, keyboard_recorder, agent_profile):
    keyboard_recorder.read()
    buffer = DynamicExperienceReplayBuffer(OBSERVATION_SPACE, EXPERIENCE_BUFFER_SIZE)
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

    agent_profile.load_initial_user_experience(*buffer.build())


def record_user_experience(screen_recorder, keyboard_recorder, agent_profile):
    keyboard_recorder.read()
    action_mapping, buffer = agent_profile.action_mapping, agent_profile.user_experience

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


def handover_control(screen_recorder, keyboard_recorder, trainer, choose_preference):
    keyboard_recorder.read()
    controller = keyboard.Controller()
    pressed_keys = set()
    previous_frame = np.zeros(FRAME_SIZE)
    frame = convert_frame(screen_recorder.cache)
    step = 0
    step_start = time.time()

    while trainer.running:
        user_pressed_keys = keyboard_recorder.read()
        if Key.esc.value.vk in user_pressed_keys:
            break

        if Key.f1.value.vk in user_pressed_keys:
            choose_preference(trainer)
            keyboard_recorder.read()

        step += 1
        observation = np.stack((previous_frame, frame))
        action = trainer.agent_profile.agent.act(observation)

        released_keys = pressed_keys - trainer.agent_profile.action_mapping[action]
        pressed_keys = trainer.agent_profile.action_mapping[action]

        for key in released_keys:
            controller.release(KeyCode.from_vk(key))

        for key in pressed_keys:
            controller.press(KeyCode.from_vk(key))

        time.sleep(max(0, STEP_SECONDS - (time.time() - step_start)))
        step_start = time.time()

        next_frame = convert_frame(screen_recorder.cache)
        next_observation = np.stack((frame, next_frame))
        trainer.experience({'agent_transition': (observation, action, next_observation)})
        previous_frame, frame = frame, next_frame

        if step % UPDATE_TARGET_PARAMETERS_STEPS == 0:
            trainer.agent_profile.agent.update_targets()

    for key in pressed_keys:
        controller.release(KeyCode.from_vk(key))
