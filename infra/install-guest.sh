#!/bin/sh
set -eu
cd "$HOME/studyflow"
if ! id studyflow >/dev/null 2>&1; then
  sudo useradd --system --create-home --home-dir /var/lib/studyflow --shell /usr/sbin/nologin studyflow
fi
sudo install -d -m 755 /opt/studyflow
sudo cp -R app.py studyflow resources examples requirements.lock /opt/studyflow/
sudo python3.12 -m venv /opt/studyflow/.venv
sudo /opt/studyflow/.venv/bin/pip install --disable-pip-version-check --no-index --find-links "$PWD/wheelhouse" -r requirements.lock
sudo install -d -o studyflow -g studyflow -m 700 /var/lib/studyflow
sudo tee /etc/systemd/system/studyflow.service >/dev/null <<'UNIT'
[Unit]
Description=StudyFlow isolated application
After=network.target

[Service]
User=studyflow
Group=studyflow
WorkingDirectory=/opt/studyflow
Environment=HOME=/var/lib/studyflow
Environment=DATA_DIR=/var/lib/studyflow
Environment=OLLAMA_URL=http://192.168.5.2:11435
Environment=METRICS_BIND=0.0.0.0
ExecStart=/opt/studyflow/.venv/bin/python -m studyflow.serve --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
RestrictSUIDSGID=true
ReadWritePaths=/var/lib/studyflow

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable studyflow.service
sudo systemctl restart studyflow.service
sudo systemctl is-active studyflow.service
