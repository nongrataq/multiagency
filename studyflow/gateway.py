import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx

class Gateway(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in {'/api/chat', '/v1/systemone'}:
            self.send_error(403)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < size <= 65536:
            self.send_error(413)
            return
        try:
            with httpx.Client(timeout=180, trust_env=False) as client:
                response = client.post(os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434') + self.path, content=self.rfile.read(size), headers={'Content-Type': 'application/json'})
            self.send_response(response.status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.content)
        except httpx.HTTPError:
            self.send_error(502)
    def log_message(self, *args):
        pass

if __name__ == '__main__':

    ThreadingHTTPServer((os.getenv('GATEWAY_BIND', '127.0.0.1'), 11435), Gateway).serve_forever()
