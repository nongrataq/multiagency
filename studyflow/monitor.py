import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from prometheus_client import start_http_server
from .telemetry import AVAILABLE, EVAL, EVAL_VALID, configure, event

class Alerts(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/alerts':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < size <= 100_000:
            self.send_error(413)
            return
        try:
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('alert must be an object')
        except ValueError:
            self.send_error(400)
            return
        directory = Path(os.getenv('DATA_DIR', 'data'))
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'alerts.jsonl').open('a') as f:
            f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'notification': body}, ensure_ascii=False) + '\n')
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def refresh_evaluation(path):
    try:
        result = json.loads(path.read_text())
        ratio = result['pass_ratio']
        if type(ratio) not in {int, float} or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('invalid pass ratio')
    except FileNotFoundError:
        EVAL.set(float('nan'))
        EVAL_VALID.set(float('nan'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        EVAL.set(float('nan'))
        EVAL_VALID.set(0)
        logging.getLogger('studyflow').warning(json.dumps({'event': 'eval_read_error', 'error': type(exc).__name__}))
    else:
        EVAL.set(ratio)
        EVAL_VALID.set(1)


def main():
    configure(os.getenv('DATA_DIR', 'data'))
    bind = os.getenv('METRICS_BIND', '127.0.0.1')
    start_http_server(9101, addr=bind)
    threading.Thread(target=ThreadingHTTPServer((bind, 9102), Alerts).serve_forever, daemon=True).start()
    with httpx.Client(timeout=5, trust_env=False) as client:
        while True:
            result = Path(os.getenv('DATA_DIR', 'data')) / 'eval-results.json'
            refresh_evaluation(result)
            with event('health_probe'):
                try:
                    response = client.get(os.getenv('OLLAMA_URL', 'http://localhost:11434') + '/api/tags')
                    AVAILABLE.set(int(response.status_code == 200))
                except httpx.HTTPError:
                    AVAILABLE.set(0)
            time.sleep(15)

if __name__ == '__main__':
    main()
