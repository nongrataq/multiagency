import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from streamlit.testing.v1 import AppTest

from studyflow import evaluate, monitor, telemetry
from studyflow.contracts import Verdict
from studyflow.flow import build_graph, quote_matches, run
from studyflow.memory import Memory
from studyflow.models import ModelError, Ollama
from test_studyflow import Models
from ui_helpers import quiz_radios, select_page


def test_quote_matching_ignores_markdown_table_and_code_decoration():
    body = '| `GET` | Получение ресурса |\n|---|---|\n| `POST` | Создание ресурса |'
    assert quote_matches('GET Получение ресурса', body)
    assert quote_matches('POST — Создание ресурса', body)
    assert not quote_matches('PATCH Удаление ресурса', body)

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def memory(tmp_path):
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# ACID\nCOMMIT фиксирует транзакцию.')
    yield m
    m.close()


def test_rejected_history_keeps_reasons_without_draft(memory):
    result = run(build_graph(memory, Models(reject=9)), 'Проверь ACID', 'student')
    memory.record_interaction(result, 'Проверь ACID')
    response = json.loads(memory.history()[0]['response'])
    assert 'text' not in response
    assert 'questions' not in response
    assert response['issues'] == ['Исправить ответ']


def test_legacy_rejected_history_does_not_render_draft(memory, monkeypatch, tmp_path):
    result = run(build_graph(memory, Models(reject=9)), 'Проверь ACID', 'student')
    memory.record_interaction(result, 'Проверь ACID')
    with memory.db:
        memory.db.execute('UPDATE interactions SET response=?', (json.dumps(result['draft']),))
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr('prometheus_client.start_http_server', lambda *a, **kw: None)
    app = AppTest.from_file(ROOT / 'app.py').run()
    select_page(app, 'История')
    assert not app.exception
    assert not any(m.value == result['draft']['text'] for m in app.markdown)


def test_saved_quiz_can_be_opened_and_graded_in_new_session(memory, monkeypatch, tmp_path):
    result = run(build_graph(memory, Models()), 'Проверь ACID', 'student')
    memory.record_interaction(result, 'Проверь ACID')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr('prometheus_client.start_http_server', lambda *a, **kw: None)
    app = AppTest.from_file(ROOT / 'app.py').run()
    select_page(app, 'История')
    assert not app.exception
    open_button = next(b for b in app.button if b.label == 'Открыть тест')
    open_button.click().run()
    assert not app.exception
    assert next(r for r in app.radio if r.key == 'page').value == 'Занятие'
    assert quiz_radios(app)[0].label == 'Что фиксирует транзакцию?'
    assert any('COMMIT фиксирует' in m.value for m in app.markdown)
    assert len(quiz_radios(app)) == 3
    for radio in quiz_radios(app):
        radio.set_value(0)
    next(b for b in app.button if b.label == 'Проверить ответы').click().run()
    assert not app.exception
    assert any('Верно: 3 из 3' in m.value for m in app.success)
    assert memory.progress()[0]['attempts'] == 3
    memory.ingest('course.md', '# Новое\nДругой материал.')
    app.run()
    assert not quiz_radios(app)
    select_page(app, 'История')
    assert not any(b.label == 'Открыть тест' for b in app.button)


@pytest.mark.parametrize('endpoint,reply', [('/v1/systemone', {}), ('/api/chat', {'message': {'content': '{"approved":"PRIVATE_NOTE_CANARY","issues":[]}'}})])
def test_contract_errors_count_as_failures_without_leaking(endpoint, reply, monkeypatch):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry.trace, 'get_tracer', provider.get_tracer)
    model = Ollama(f'http://127.0.0.1:{server.server_port}')
    name = model.router if endpoint == '/v1/systemone' else model.validator
    before = {s: telemetry.CALLS.labels(name, s)._value.get() for s in ['ok', 'error']}
    try:
        with pytest.raises(ModelError):
            with telemetry.event('workflow'):
                if endpoint == '/v1/systemone':
                    model.route('ACID')
                else:
                    model.chat(name, 'system', {}, Verdict)
    finally:
        model.close()
        server.shutdown()
        server.server_close()
    assert received[0][0] == endpoint
    assert telemetry.CALLS.labels(name, 'ok')._value.get() == before['ok']
    expected_errors = 2 if endpoint == '/api/chat' else 1
    assert telemetry.CALLS.labels(name, 'error')._value.get() == before['error'] + expected_errors
    spans = exporter.get_finished_spans()
    assert spans
    assert all(s.status.status_code.name == 'ERROR' for s in spans)
    assert 'PRIVATE_NOTE_CANARY' not in str([(s.attributes, s.events, s.status.description) for s in spans])
    provider.shutdown()


def test_unprobed_ollama_is_unknown():

    assert math.isnan(telemetry.AVAILABLE._value.get())


def test_monitor_survives_corrupt_result_and_recovers(tmp_path):
    path = tmp_path / 'eval-results.json'
    path.write_text('')
    monitor.refresh_evaluation(path)
    assert math.isnan(telemetry.EVAL._value.get())
    assert telemetry.EVAL_VALID._value.get() == 0
    evaluate.atomic_json(path, {'pass_ratio': 0.9})
    monitor.refresh_evaluation(path)
    assert telemetry.EVAL._value.get() == 0.9
    assert telemetry.EVAL_VALID._value.get() == 1


def test_eval_retains_outputs_and_gates_each_candidate(tmp_path, monkeypatch):
    class Candidates(Models):
        router, context, threads = 'test-router', 4096, 6
        def route(self, request):
            self.reject = 9 if self.writer == 'qwen3.5:0.8b' and request.endswith('1') else 0
            return 'quiz'
        def metadata(self):
            return {'version': {'version': 'test'}, 'models': {'models': []}}
        def close(self):
            pass
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text('\n'.join(json.dumps(dict(id=f'case{i}', request=f'Проверь ACID {i}', mode='quiz', status='approved', rubric='Проверь по источникам')) for i in [1, 2]))
    monkeypatch.setattr(evaluate, 'Ollama', Candidates)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--compare', '--dataset', str(dataset)])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1
    result = json.loads((tmp_path / 'data/eval-results.json').read_text())
    assert result['per_model']['qwen3.5:0.8b']['pass_ratio'] == 0.5
    assert result['pass_ratio'] == 0.5
    assert result['overall_pass_ratio'] == pytest.approx(5/6)
    assert result['metadata']['runtime']['version']['version'] == 'test'
    assert result['complete']
    assert len(list((tmp_path / 'data/evals').glob('*/results.json'))) == 1
    for row in result['records']:
        assert row['result']['draft']['text']
        assert row['result']['sources']
        assert row['result']['verdict']
    assert any('judge' in row and row['judge']['approved'] for row in result['records'])


def test_metrics_ready_before_browser_session(tmp_path):
    def free_port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]
    metrics_port, app_port = free_port(), free_port()
    env = {**os.environ, 'DATA_DIR': str(tmp_path), 'METRICS_BIND': '127.0.0.1', 'METRICS_PORT': str(metrics_port)}
    process = subprocess.Popen([sys.executable, '-m', 'studyflow.serve', '--server.address=127.0.0.1', f'--server.port={app_port}', '--server.headless=true', '--browser.gatherUsageStats=false'], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(timeout=0.5, trust_env=False) as client:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f'Application exited: {process.returncode}')
                try:
                    metrics = client.get(f'http://127.0.0.1:{metrics_port}/metrics')
                    health = client.get(f'http://127.0.0.1:{app_port}/_stcore/health')
                    if metrics.status_code == health.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail('Metrics/health did not start without a browser session')
            assert 'studyflow_runs_total' in metrics.text or 'studyflow_runs' in metrics.text
            assert 'studyflow_ollama_available NaN' in metrics.text
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_atomic_eval_publication_preserves_last_result_on_failure(tmp_path):
    path = tmp_path / 'eval-results.json'
    evaluate.atomic_json(path, {'pass_ratio': 0.9})
    with pytest.raises(TypeError):
        evaluate.atomic_json(path, {'pass_ratio': object()})
    assert json.loads(path.read_text()) == {'pass_ratio': 0.9}
    assert list(tmp_path.iterdir()) == [path]


def test_monitor_reads_atomic_updates_while_writer_is_active(tmp_path):
    path = tmp_path / 'eval-results.json'
    evaluate.atomic_json(path, {'pass_ratio': 1})
    def publish():
        for i in range(20):
            evaluate.atomic_json(path, {'pass_ratio': i / 20})
    writer = threading.Thread(target=publish)
    writer.start()
    while writer.is_alive():
        monitor.refresh_evaluation(path)
        assert telemetry.EVAL_VALID._value.get() == 1
    writer.join()
    monitor.refresh_evaluation(path)
    assert telemetry.EVAL._value.get() == 0.95


def test_non_quiz_cannot_persist_quiz_fields(memory):
    class WrongMode(Models):
        def route(self, request):
            return 'explain'
    result = run(build_graph(memory, WrongMode()), 'Объясни ACID', 'student')
    assert result['status'] == 'rejected'
    assert result['attempts'] == 2
    assert memory.artifact(result['run']) is None
    assert 'questions' in result['verdict']['issues'][0]


def test_approval_cannot_hide_reported_problems():
    with pytest.raises(ValueError, match='approval cannot contain issues'):
        Verdict(checks=[], approved=True, issues=['Неподтверждённое утверждение'])


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'invented_quote', 'unsupported'])
def test_approval_requires_complete_real_evidence(memory, fault):
    class BadReview(Models):
        def chat(self, model, system, context, schema):
            answer = super().chat(model, system, context, schema)
            if schema == Verdict:
                if fault == 'missing':
                    answer = answer.model_copy(update={'checks': []})
                elif fault == 'duplicate':
                    answer = answer.model_copy(update={'checks': answer.checks * 2})
                else:
                    check = answer.checks[0].model_copy(update={'quote': 'Придуманная цитата'} if fault == 'invented_quote' else {'supported': False, 'reason': 'Нет подтверждения'})
                    answer = answer.model_copy(update={'checks': [check, *answer.checks[1:]]})
            return answer
    result = run(build_graph(memory, BadReview()), 'Проверь ACID', 'student')
    assert result['status'] == 'rejected'
    assert memory.artifact(result['run']) is None
