venv/bin/python -m pip install pyinstaller
venv/bin/pyinstaller --noconfirm --name reinforcebot --add-data resources:resources reinforcebot/main.py
