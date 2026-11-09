#!/bin/sh
set -eu
[ "$(uname -s)" = Linux ] || { echo 'Run inside the Linux guest' >&2; exit 1; }
sudo iptables -C OUTPUT -j REJECT
sudo ip6tables -C OUTPUT -j REJECT
sudo iptables -C OUTPUT -d 192.168.5.2 -p tcp --dport 11435 -j ACCEPT
echo 'Firewall rules: IPv4 and IPv6 egress restricted; only the host model gateway is allowlisted.'
test "$(systemctl show studyflow -p User --value)" = studyflow
test "$(systemctl show studyflow -p NoNewPrivileges --value)" = yes
sudo systemctl is-active studyflow.service
echo 'Service hardening: User=studyflow; NoNewPrivileges=yes.'
if sudo -u studyflow sudo -n true 2>/dev/null; then
  echo 'Application account has sudo access' >&2
  exit 1
fi
sudo -u studyflow /opt/studyflow/.venv/bin/python - <<'PY'
import os
from pathlib import Path
import httpx

mounts = Path('/proc/mounts').read_text().splitlines()
assert not any(line.split()[2] in {'9p', 'virtiofs'} for line in mounts), 'Host filesystem is mounted'
print('Host filesystem mounts: none (no 9p/virtiofs).')
assert not Path('/var/run/docker.sock').exists(), 'Docker socket is available'
assert not Path('/run/docker.sock').exists(), 'Docker socket is available'
print('Docker socket: absent.')
for url in ['http://1.1.1.1', 'http://[2606:4700:4700::1111]']:
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            client.get(url)
    except (httpx.ConnectError, httpx.TimeoutException):
        print('Blocked egress:', url)
    else:
        raise AssertionError('Unexpected external access: ' + url)
with httpx.Client(base_url=os.getenv('OLLAMA_URL', 'http://192.168.5.2:11435'), timeout=5, trust_env=False) as client:
    response = client.post('/api/delete', json={})
    assert response.status_code == 403, response.status_code
    print(f'Model gateway rejects disallowed /api/delete: HTTP {response.status_code}.')
with httpx.Client(timeout=5, trust_env=False) as client:
    app_status = client.get('http://127.0.0.1:8501/_stcore/health').status_code
    assert app_status == 200, app_status
    metrics_status = client.get('http://127.0.0.1:9100/metrics').status_code
    assert metrics_status == 200, metrics_status
    print(f'Application health: HTTP {app_status}.')
    print(f'Metrics endpoint: HTTP {metrics_status}.')
print('Isolation and health checks passed.')
PY
