import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from studyflow.contracts import Question, QuizDraft, QuizProposal, Draft, Plan, ClaimCheck, Verdict
from studyflow.flow import build_graph, run, bounded_sources, quiz_evidence, review_context, QUIZ_INTRO
from studyflow.memory import Memory
from studyflow.models import ModelError, Ollama
from studyflow.monitor import Alerts
from test_studyflow import Models
from ui_helpers import quiz_radios, select_page, select_activity, click_button

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def memory(tmp_path):
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# ACID\nCOMMIT фиксирует транзакцию.')
    yield m
    m.close()


def test_selected_answers_survive_reopen_and_cannot_be_rewritten(memory, tmp_path):
    result = run(build_graph(memory, Models()), 'ACID', 'student')
    grades = memory.grade(result['run'], [1, 0, 1])
    assert [g['correct'] for g in grades] == [0, 1, 0]
    assert [g['selected'] for g in grades] == [1, 0, 1]
    reopened = Memory(tmp_path / 'memory.sqlite')
    try:
        assert [g['selected'] for g in reopened.grade(result['run'], [0, 0, 0])] == [1, 0, 1]
        assert reopened.progress()[0]['attempts'] == 3
    finally:
        reopened.close()


def test_legacy_progress_migration_keeps_results(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE progress(run TEXT,question INTEGER,topic TEXT,correct INTEGER,created TEXT,PRIMARY KEY(run,question)); INSERT INTO progress VALUES('old',0,'ACID',1,'2026-01-01');")
    m = Memory(path)
    try:
        assert m.grades('old')[0]['correct'] == 1
        assert m.grades('old')[0]['selected'] is None
    finally:
        m.close()


def test_explicit_quiz_skips_route_and_mode_does_not_leak(memory, tmp_path):
    class Counted(Models):
        routes = 0
        def route(self, request):
            self.routes += 1
            return super().route(request)
    model = Counted()
    with SqliteSaver.from_conn_string(str(tmp_path / 'checkpoint.sqlite')) as saver:
        graph = build_graph(memory, model, saver)
        quiz = run(graph, 'Создай тест по конспекту', 'student', mode='quiz')
        assert quiz['status'] == 'approved'
        assert model.routes == 0
        assert [name for name, _ in model.calls] == [model.writer, model.validator]
        model.mode = 'explain'
        explanation = run(graph, 'Объясни ACID', 'student')
        assert explanation['mode'] == 'explain'
        assert model.routes == 1


def test_compact_quiz_uses_source_text_for_explanation(memory):
    result = run(build_graph(memory, Models()), 'Создай тест по конспекту', 'student', mode='quiz')
    source = result['sources'][0]
    assert result['draft']['questions'][0]['explanation'] in source['body']
    assert result['draft']['citations'][0]['version'] == source['version']
    context = review_context('test', 'quiz', result['sources'], result['draft'])
    assert len(context['statements']) == 3
    custom = {**result['draft'], 'text': 'COMMIT удаляет базу данных.'}
    assert len(review_context('test', 'quiz', result['sources'], custom)['statements']) == 4


def test_unknown_quiz_evidence_is_not_persisted(memory):
    class Unknown(Models):
        def chat(self, model, system, context, schema):
            answer = super().chat(model, system, context, schema)
            if schema is QuizProposal:
                answer.questions[0].evidence_id = 999
            return answer
    with pytest.raises(ModelError, match='отсутствующий фрагмент'):
        run(build_graph(memory, Unknown()), 'Создай тест по конспекту', 'student', mode='quiz')
    assert memory.db.execute('SELECT count(*) FROM artifacts').fetchone()[0] == 0


def test_source_and_evidence_budgets_keep_exact_text():
    sources = [{'id': i, 'body': 'A' * 4000} for i in range(3)]
    selected = bounded_sources(sources)
    assert sum(len(s['body'].encode()) for s in selected) <= 6000
    assert all(s in sources for s in selected)
    evidence = quiz_evidence([{'id': 1, 'body': 'First fact. Second fact. Third fact.'}], byte_budget=24)
    assert len(evidence) == 2
    assert all(item['text'] in 'First fact. Second fact. Third fact.' for item in evidence)
    assert sum(len(item['text'].encode()) for item in evidence) <= 24


def test_quiz_model_schema_constrains_evidence_ids(monkeypatch):
    captured = []
    def response(self, path, payload, decode):
        captured.append(payload['format'])
        return decode({'message': {'content': json.dumps({'questions': [{'text': f'Question {i}?', 'options': ['A', 'B'], 'correct': 0, 'evidence_id': 7} for i in range(3)]})}})
    monkeypatch.setattr(Ollama, 'post', response)
    model = Ollama('http://localhost:1')
    try:
        model.chat(model.writer, 'system', {'evidence': [{'id': 7, 'text': 'Fact.'}]}, QuizProposal)
        assert captured[0]['$defs']['ProposedQuestion']['properties']['evidence_id']['enum'] == [7]
        assert set(captured[0]['$defs']['ProposedQuestion']['properties']) == {'text', 'options', 'correct', 'evidence_id'}
    finally:
        model.close()


def setup_app(memory, tmp_path, monkeypatch, model):
    import studyflow.models
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    model.close = lambda: None
    monkeypatch.setattr(studyflow.models, 'Ollama', lambda: model)
    return AppTest.from_file(ROOT / 'app.py', default_timeout=20).run()


def test_quiz_ui_incomplete_answers_grading_and_reopen(memory, tmp_path, monkeypatch):
    class ExplicitOnly(Models):
        def route(self, request):
            raise AssertionError('Explicit quiz must not use the router')
    model = ExplicitOnly()
    app = setup_app(memory, tmp_path, monkeypatch, model)
    select_activity(app, 'Пройти тест')
    click_button(app, 'Создать тест')
    assert not app.exception
    assert len(quiz_radios(app)) == 3
    assert len(model.calls) == 2
    next(b for b in app.button if b.label == 'Проверить ответы').click().run()
    assert any('Ответьте на каждый вопрос' in e.value for e in app.error)
    assert not memory.progress()
    for radio, choice in zip(quiz_radios(app), [1, 0, 1]):
        radio.set_value(choice)
    next(b for b in app.button if b.label == 'Проверить ответы').click().run()
    assert not app.exception
    assert any('Верно: 1 из 3' in s.value for s in app.success)
    assert not quiz_radios(app)
    assert not any(b.label == 'Проверить ответы' for b in app.button)
    app.run()
    assert any('Верно: 1 из 3' in s.value for s in app.success)
    new_session = AppTest.from_file(ROOT / 'app.py').run()
    select_page(new_session, 'История')
    next(b for b in new_session.button if b.label == 'Открыть тест').click().run()
    assert not new_session.exception
    assert any('Верно: 1 из 3' in s.value for s in new_session.success)
    assert not quiz_radios(new_session)
    assert len(model.calls) == 2, 'Grading/reopening must not call models'


def test_quiz_button_uses_entered_topic(memory, tmp_path, monkeypatch):
    model = Models()
    app = setup_app(memory, tmp_path, monkeypatch, model)
    select_activity(app, 'Пройти тест')
    app.text_area[0].set_value('COMMIT')
    click_button(app, 'Создать тест')
    assert not app.exception
    context = next(context for name, context in model.calls if name == model.writer)
    assert context['request'] == 'Создай тест по теме: COMMIT'
    assert context['mode'] == 'quiz'


def test_failed_request_remains_visible_and_is_recorded(memory, tmp_path, monkeypatch):
    class Failed(Models):
        def route(self, request):
            raise ModelError('Истекло время ожидания модели')
    app = setup_app(memory, tmp_path, monkeypatch, Failed())
    app.text_area[0].set_value('Объясни ACID')
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert memory.history()[0]['status'] == 'error'
    assert 'Истекло' in json.loads(memory.history()[0]['response'])['issues'][0]
    app.run()
    assert any('Истекло' in e.value for e in app.error)


def test_interrupted_request_is_recorded_and_lock_is_released(memory, tmp_path, monkeypatch):
    import streamlit as st
    class InterruptedOnce(Models):
        interrupted = False
        def route(self, request):
            if not self.interrupted:
                self.interrupted = True
                st.rerun()
            return super().route(request)
    app = setup_app(memory, tmp_path, monkeypatch, InterruptedOnce())
    app.text_area[0].set_value('ACID')
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert memory.history()[0]['status'] == 'error'
    assert 'прерван' in json.loads(memory.history()[0]['response'])['issues'][0]
    assert any('прерван' in e.value for e in app.error)
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert len(quiz_radios(app)) == 3


def test_whole_request_deadline_is_shared_across_http_calls():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            calls.append(self.path)
            time.sleep(0.03 if len(calls) == 1 else 0.8)
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(json.dumps({'answers': {'mode': {'choice': 'quiz'}}}).encode())
            except BrokenPipeError:
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = Ollama(f'http://127.0.0.1:{server.server_port}', timeout=2)
    model.request_timeout = 0.4
    model.begin_request()
    try:
        assert model.route('ACID') == 'quiz'
        with pytest.raises(ModelError, match='время ожидания'):
            model.route('ACID')
        assert len(calls) == 2
        with pytest.raises(ModelError, match='общее время'):
            model.route('ACID')
        assert len(calls) == 2
    finally:
        model.close()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('options', [['COMMIT', ' commit '], ['', 'ROLLBACK']])
def test_ambiguous_options_are_rejected(options):
    with pytest.raises(ValidationError):
        Question(text='Что фиксирует транзакцию?', options=options, correct=0, explanation='COMMIT.', topic='ACID')


def test_duplicate_quiz_questions_are_rejected():
    question = {'text': 'Вопрос?', 'options': ['Да', 'Нет'], 'correct': 0, 'explanation': 'Ответ.', 'topic': 'ACID'}
    with pytest.raises(ValidationError):
        QuizDraft(text='Тест', citations=[{'source_id': 1}], questions=[question] * 3)


@pytest.mark.parametrize('schema,payload', [
    (Plan, {'query': '   '}),
    (Draft, {'text': '\n ', 'citations': [{'source_id': 1}]}),
    (ClaimCheck, {'statement_id': 0, 'source_id': 1, 'quote': ' ', 'reason': 'Подтверждено', 'supported': True}),
    (Verdict, {'checks': [], 'issues': [' '], 'approved': False}),
])
def test_blank_answers_and_empty_evidence_are_not_valid(schema, payload):
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_heading_only_upload_does_not_destroy_existing_document(memory):
    before = memory.documents()[0]['version']
    with pytest.raises(ValueError, match='только заголовки'):
        memory.ingest('course.md', '# Только заголовок')
    assert memory.documents()[0]['version'] == before
    assert memory.search('COMMIT')


def test_corrupt_notification_does_not_break_quiz_page(memory, tmp_path, monkeypatch):
    (tmp_path / 'alerts.jsonl').write_text('{bad\n' + json.dumps({'notification': 'valid'}) + '\n')
    app = setup_app(memory, tmp_path, monkeypatch, Models())
    select_page(app, 'Состояние сервиса')
    assert not app.exception
    assert any('уведомление повреждено' in w.value for w in app.warning)


@pytest.mark.parametrize('length,body,expected', [('-1', b'', 413), ('bad', b'', 400), ('2', b'[]', 400)])
def test_alert_webhook_rejects_invalid_input(length, body, expected):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Alerts)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import http.client
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=1)
        client.request('POST', '/alerts', body=body, headers={'Content-Length': length})
        assert client.getresponse().status == expected
        client.close()
    finally:
        server.shutdown()
        server.server_close()
