import json
import http.client
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from studyflow.gateway import Gateway


@contextmanager
def server(handler):
    instance = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=instance.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{instance.server_port}'
    finally:
        instance.shutdown()
        instance.server_close()


@pytest.mark.parametrize('path', ['/api/chat', '/v1/systemone'])
def test_gateway_only_forwards_allowed_calls(monkeypatch, path):
    received = []
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        def log_message(self, *args):
            pass
    with server(Upstream) as upstream, server(Gateway) as proxy, httpx.Client(timeout=2, trust_env=False) as client:
        monkeypatch.setenv('OLLAMA_URL', upstream)
        payload = {'model': 'test'}
        response = client.post(proxy + path, json=payload)
        assert response.status_code == 200 and response.json() == {'ok': True}
        assert received == [(path, payload)]
        assert client.post(proxy + '/api/delete', json={}).status_code == 403
        assert client.post(proxy + '/api/pull', json={}).status_code == 403
        assert client.post(proxy + '/api/chat', content=b'x' * 65537).status_code == 413
        assert len(received) == 1


@pytest.mark.parametrize('length,status', [('invalid', 400), ('-1', 413)])
def test_gateway_rejects_invalid_content_length(length, status):
    with server(Gateway) as proxy:
        client = http.client.HTTPConnection(proxy.removeprefix('http://'), timeout=1)
        try:
            client.request('POST', '/api/chat', body=b'', headers={'Content-Length': length})
            assert client.getresponse().status == status
        finally:
            client.close()
