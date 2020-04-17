# ReinforceBot Client
The ReinforceBot client, used for RL agent creation and training on the desktop.

## Requirements
* Linux on Xorg
* Optionally CUDA (10.2)

## Installation
1.  Build the binary.
    ```bash
    ./package.sh
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
