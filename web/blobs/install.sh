set -e
mkdir -p /tmp
cd /tmp
wget http://localhost:8080/blobs/reinforcebot-client.tar.gz -O reinforcebot-client.tar.gz
tar -xzvf reinforcebot-client.tar.gz -C /opt
rm -rf reinforcebot-client.tar.gz
mkdir -p /usr/local/share/applications
touch /usr/local/share/applications/reinforcebot.desktop
echo '[Desktop Entry]
Version=1.0
Type=Application
Name=ReinforceBot
Icon=/opt/reinforcebot/resources/icon.svg
Exec="/opt/reinforcebot/reinforcebot" %f
Comment=A Reinforcement Learning Toolkit for the Desktop
Categories=Science;
Terminal=false' >>/usr/local/share/applications/reinforcebot.desktop
printf "\n\033[32;1mReinforceBot successfully installed!\033[0m\n\n"
