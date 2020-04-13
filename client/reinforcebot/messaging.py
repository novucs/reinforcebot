import subprocess


def notify(message):
    subprocess.Popen(('notify-send', '--hint', 'int:transient:1', 'ReinforceBot', message))
