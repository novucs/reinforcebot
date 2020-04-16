from pynput import keyboard


class KeyboardRecorder:
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
