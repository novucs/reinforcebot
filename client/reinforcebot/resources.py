import os
import sys


def resource_path(relative):
    path = os.path.join(getattr(sys, '_MEIPASS', '.'), 'resources', relative)
    print(path)
    return path


def glade():
    return resource_path('main.glade')


def xdotool():
    return resource_path('xdotool')


def wnckprop():
    return resource_path('wnckprop')


def gnome_screenshot():
    return resource_path('gnome-screenshot')
