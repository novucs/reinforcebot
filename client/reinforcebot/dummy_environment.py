from time import sleep

import gym
from pynput import keyboard

action = 0
running = True


def on_press(key):
    global action
    if key == keyboard.Key.up:
        action = 2
    if key == keyboard.Key.down:
        action = 5


def on_release(key):
    global action
    action = 0


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
env = gym.make("Pong-v0")
observation = env.reset()

while running:
    env.render()
    sleep(.1)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
