import subprocess


def notify(message):
    subprocess.Popen(('notify-send', 'ReinforceBot', message))
