# ReinforceBot Client
The ReinforceBot client, used for RL agent creation and training on the desktop.

**Note:** *This has only been tested on Manjaro GNOME, other distributions may differ.*

## Requirements
* Linux on Xorg
* Python 3.7
* Optionally CUDA (10.2)

## Installation
1.  Setup the Python virtual environment.
    ```bash
    python -m venv venv
    venv/bin/python -m pip install -r requirements.txt
    ```

1.  Build the binary.
    ```bash
    venv/bin/python -m pip install pyinstaller
    venv/bin/pyinstaller --noconfirm --name reinforcebot --add-data resources:resources reinforcebot/main.py
    ```

2.  Install the binary.
    ```bash
    sudo cp -r dist/reinforcebot /opt/reinforcebot
    ```

3.  Optionally add a desktop entry.

    Create a new file:
    `$HOME/.local/share/applications/reinforcebot.desktop`
    
    Add the following contents:
    ```
    [Desktop Entry]
    Version=1.0
    Type=Application
    Name=ReinforceBot
    Icon=/opt/reinforcebot/resources/icon.svg
    Exec="/opt/reinforcebot/reinforcebot" %f
    Comment=A Reinforcement Learning Toolkit for the Desktop
    Categories=Science;
    Terminal=false
    ```

4.  Start the application via clicking on the ReinforceBot desktop entry, or by executing `/opt/reinforcebot/reinforcebot`

## Uninstall
Remove all installed files:
```bash
sudo rm -rf /opt/reinforcebot $HOME/.local/share/applications/reinforcebot.desktop
```
