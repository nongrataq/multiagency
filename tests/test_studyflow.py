import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from studyflow.contracts import Plan, Draft, QuizDraft, QuizProposal, Verdict
from studyflow.memory import Memory
from studyflow.models import Ollama, ModelError
from studyflow.flow import build_graph, run
from ui_helpers import quiz_radios, select_activity, click_button

@pytest.fixture
def memory(tmp_path):
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# Транзакции\nACID: атомарность и согласованность. COMMIT фиксирует транзакцию.')
    yield m
    m.close()

class Models:
    small, writer, validator = 'qwen-small', 'qwen-writer', 'gemma-validator'
    def __init__(self, reject=0, bad_citation=False, wrong_metadata=False, mode='quiz', query='ACID', question_count=3):
        self.calls, self.reject, self.bad_citation, self.wrong_metadata = [], reject, bad_citation, wrong_metadata
        self.mode, self.query, self.question_count = mode, query, question_count
    def route(self, request):
        return self.mode
    def begin_request(self):
        pass
    def chat(self, model, system, context, schema):
        self.calls.append((model, context))
        if schema == Plan:
            return Plan(query=self.query)
        if schema == Verdict:
            self.reject -= 1
            checks = [{'statement_id': s['id'], 'source_id': context['sources'][0]['id'], 'quote': context['sources'][0]['body'], 'reason': 'Подтверждено источником', 'supported': True} for s in context.get('statements', [])]
            return Verdict(checks=checks, approved=self.reject < 0, issues=['Исправить ответ'] if self.reject >= 0 else [])
        if schema == QuizProposal:
            return QuizProposal(questions=[{'text': text, 'options': ['COMMIT', 'ROLLBACK'], 'correct': 0, 'evidence_id': context['evidence'][0]['id']} for text in ['Что фиксирует транзакцию?', 'Какой оператор фиксирует транзакцию?', 'Как называется команда фиксации?']])
        source = context['sources'][0]
        texts = ['Что фиксирует транзакцию?', 'Какой оператор фиксирует транзакцию?', 'Что делает COMMIT?', 'Как называется команда фиксации?', 'Какой оператор завершает транзакцию?']
        questions = [
            {'text': text, 'options': ['COMMIT', 'ROLLBACK'], 'correct': 0, 'explanation': 'COMMIT фиксирует транзакцию.', 'topic': 'Транзакции'}
            for text in texts[:self.question_count]
        ] if self.mode == 'quiz' else []
        return Draft(text='Тест по транзакциям', citations=[{'source_id': 999 if self.bad_citation else source['id'], 'version': 'wrong' if self.wrong_metadata else source['version'], 'section': 'wrong' if self.wrong_metadata else source['heading']}], questions=questions)

def test_verified_flow_and_idempotent_grading(memory):
    model = Models()
    result = run(build_graph(memory, model), 'Проверь ACID', 'one')
    assert result['status'] == 'approved'
    assert model.calls[-1][0] == model.validator
    grades = memory.grade(result['run'], [0, 0, 0])
    assert sum(g['correct'] for g in grades) == 3
    memory.grade(result['run'], [1, 1, 1])
    assert memory.progress()[0]['attempts'] == 3
    assert memory.progress()[0]['mastery'] == 1

def test_reject_then_repair(memory):
    model = Models(reject=1, mode='explain')
    result = run(build_graph(memory, model), 'ACID', 'one')
    assert result['status'] == 'approved'
    assert result['attempts'] == 2
    generations = [context for name, context in model.calls if name == model.writer]
    assert 'previous_draft' not in generations[0]
    assert generations[1]['previous_draft']['text'] == result['draft']['text']
    assert generations[1]['feedback'] == {'issues': ['Исправить ответ']}

def test_search_planner_schema_and_stage_progress(memory):
    stages = []
    model = Models()
    result = run(build_graph(memory, model, on_stage=stages.append), 'ACID', 'one')
    assert result['status'] == 'approved'
    assert len(stages) == 4
    assert list(Plan.model_json_schema()['properties']) == ['query']

def test_rejection_never_persisted(memory):
    result = run(build_graph(memory, Models(reject=9)), 'ACID', 'one')
    assert result['status'] == 'rejected'
    assert result['attempts'] == 2
    assert memory.artifact(result['run']) is None

def test_short_quiz_is_rejected_and_never_persisted(memory):
    from pydantic import ValidationError
    class ShortQuiz(Models):
        def chat(self, model, system, context, schema):
            answer = super().chat(model, system, context, schema)
            if schema is QuizProposal:
                return QuizProposal.model_validate({'questions': [answer.questions[0].model_dump()]})
            return answer
    with pytest.raises(ValidationError):
        run(build_graph(memory, ShortQuiz()), 'Проверь ACID', 'one')
    assert memory.db.execute('SELECT count(*) FROM artifacts').fetchone()[0] == 0

def test_quiz_schema_requires_three_questions():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QuizDraft.model_validate({'text': 'Тест', 'citations': [{'source_id': 1}], 'questions': [
            {'text': 'Вопрос?', 'options': ['Да', 'Нет'], 'correct': 0, 'explanation': 'Ответ.', 'topic': 'Тема'}
        ]})

def test_quiz_schema_uses_exactly_three_questions():
    from pydantic import ValidationError
    question = {'text': 'Вопрос?', 'options': ['Да', 'Нет'], 'correct': 0, 'explanation': 'Ответ.', 'topic': 'Тема'}
    with pytest.raises(ValidationError):
        QuizDraft.model_validate({'text': 'Тест', 'citations': [{'source_id': 1}], 'questions': [question] * 4})

def test_validator_context_contains_only_cited_sources():
    from studyflow.flow import review_context
    sources = [{'id': 1, 'body': 'A'}, {'id': 2, 'body': 'B'}]
    draft = {'text': 'Ответ.', 'citations': [{'source_id': 2}], 'questions': []}
    context = review_context('запрос', 'explain', sources, draft)
    assert context['sources'] == [sources[1]]

def test_unknown_citation_is_rejected(memory):
    model = Models(bad_citation=True, mode='explain')
    result = run(build_graph(memory, model), 'ACID', 'one')
    assert result['status'] == 'rejected'
    assert not any(m == model.validator for m, _ in model.calls)

def test_citation_metadata_is_bound_to_retrieved_source(memory):
    result = run(build_graph(memory, Models(wrong_metadata=True, mode='explain')), 'Объясни ACID', 'one')
    assert result['status'] == 'approved'
    citation = result['draft']['citations'][0]
    source = result['sources'][0]
    assert (citation['source_id'], citation['version'], citation['section']) == (source['id'], source['version'], source['heading'])

def test_generic_study_plan_uses_uploaded_material(memory):
    model = Models(mode='plan', query='подготовка')
    result = run(build_graph(memory, model), 'Составь план подготовки', 'one')
    assert result['status'] == 'approved'
    assert result['search_query'] == 'все разделы загруженных материалов'
    assert result['sources']

def test_generic_quiz_uses_uploaded_material(memory):
    model = Models(mode='quiz', query='вопросы')
    result = run(build_graph(memory, model), 'Создай тест по конспекту', 'one')
    assert result['status'] == 'approved'
    assert result['search_query'] == 'все разделы загруженных материалов'
    assert result['sources']

def test_interaction_history_keeps_previous_requests_and_answers(memory):
    result = run(build_graph(memory, Models()), 'Проверь ACID', 'student')
    memory.record_interaction(result, 'Проверь ACID', 'student')
    entry = memory.history('student')[0]
    assert entry['request'] == 'Проверь ACID'
    assert entry['status'] == 'approved'
    assert json.loads(entry['response'])['text'] == result['draft']['text']

def test_missing_evidence_stops_before_generation(tmp_path):
    m = Memory(tmp_path / 'empty.sqlite')
    model = Models()
    result = run(build_graph(m, model), 'ACID', 'one')
    assert result['status'] == 'insufficient'
    assert len(model.calls) == 1
    m.close()

def test_source_change_invalidates_and_removes_search(memory):
    result = run(build_graph(memory, Models()), 'ACID', 'one')
    memory.ingest('course.md', '# Новая тема\nИндексы ускоряют поиск.')
    assert not memory.search('ACID')
    assert memory.artifact(result['run'])['stale']
    with pytest.raises(ValueError, match='устарел'):
        memory.grade(result['run'], [0])
    with pytest.raises(ValueError, match='изменился'):
        memory.save('new', 'quiz', Draft.model_validate(result['draft']))

def test_persistent_dialog_and_progress(memory, tmp_path):
    path = str(tmp_path / 'checkpoint.sqlite')
    with SqliteSaver.from_conn_string(path) as saver:
        first = run(build_graph(memory, Models(), saver), 'ACID', 'one')
    memory.grade(first['run'], [0, 0, 0])
    model = Models()
    with SqliteSaver.from_conn_string(path) as saver:
        run(build_graph(memory, model, saver), 'ACID ещё', 'one')
    assert 'ACID' in model.calls[0][1]['previous_context']
    teacher_context = next(context for name, context in model.calls if name == model.writer)
    assert teacher_context['progress'][0]['mastery'] == 1

@pytest.mark.parametrize('answers', [[None], [8], [], [True]])
def test_invalid_answers_do_not_write(memory, answers):
    result = run(build_graph(memory, Models()), 'ACID', 'one')
    with pytest.raises(ValueError):
        memory.grade(result['run'], answers)
    assert not memory.progress()

def test_real_http_contract_and_failure(memory):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, body))
            if self.path == '/v1/systemone':
                reply = {'answers': {'mode': {'choice': 'quiz'}}, 'usage': {'input_tokens': 4}}
            else:
                context = json.loads(body['messages'][1]['content'])
                schema = body['format']['title']
                fake = Models()
                cls = {'Plan': Plan, 'Draft': Draft, 'QuizDraft': QuizDraft, 'QuizProposal': QuizProposal, 'Verdict': Verdict}[schema]
                reply = {'message': {'content': fake.chat(body['model'], '', context, cls).model_dump_json()}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Ollama(f'http://127.0.0.1:{server.server_port}', timeout=0.5)
    try:
        result = run(build_graph(memory, model), 'ACID', 'http')
        assert result['status'] == 'approved'
        assert len(calls) == 4
        assert calls[0][0] == '/v1/systemone'
        assert all(body['stream'] is False and body['keep_alive'] == '5m' for path, body in calls if path == '/api/chat')
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(ModelError):
        model.route('ACID')
    model.close()

def test_chat_retries_once_after_schema_failure():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            content = '{' if len(calls) == 1 else Plan(query='ACID').model_dump_json()
            reply = {'message': {'content': content}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = Ollama(f'http://127.0.0.1:{server.server_port}')
    try:
        result = model.chat(model.small, 'system', {'request': 'ACID'}, Plan)
        assert result.query == 'ACID'
        assert len(calls) == 2
        assert 'не прошёл проверку структуры' in calls[1]['messages'][0]['content']
    finally:
        model.close()
        server.shutdown()
        server.server_close()

def test_chat_reports_repeated_schema_failure_without_model_output(monkeypatch):
    calls = []
    def invalid_response(path, payload, decode):
        calls.append(payload['keep_alive'])
        return decode({'message': {'content': 'raw invalid model output'}})
    monkeypatch.setattr(Ollama, 'post', lambda self, path, payload, decode: invalid_response(path, payload, decode))
    model = Ollama('http://localhost:1')
    with pytest.raises(ModelError) as error:
        model.chat(model.small, 'system', {'request': 'ACID'}, Plan)
    assert len(calls) == 2
    assert calls == ['5m', '5m']
    assert 'Локальная модель дважды вернула ответ неверного формата (Plan)' in str(error.value)
    assert 'raw invalid model output' not in str(error.value)
    assert 'Ответ не сохранён' in str(error.value)
    model.close()

@pytest.mark.parametrize('schema,limit', [(Plan, 96), (Draft, 600), (QuizDraft, 900), (Verdict, 700)])
def test_model_output_limit_is_bounded_by_schema(monkeypatch, schema, limit):
    captured = []
    def successful_response(self, path, payload, decode):
        captured.append(payload['options']['num_predict'])
        if schema is Plan:
            result = Plan(query='ACID')
        elif schema is Verdict:
            result = Verdict(checks=[], approved=True)
        elif schema is QuizDraft:
            question = {'text': 'Вопрос?', 'options': ['Да', 'Нет'], 'correct': 0, 'explanation': 'Ответ.', 'topic': 'Тема'}
            result = QuizDraft(text='Тест', citations=[{'source_id': 1}], questions=[{**question, 'text': f'Вопрос {i}?'} for i in range(3)])
        else:
            result = Draft(text='Ответ', citations=[{'source_id': 1}], questions=[])
        return decode({'message': {'content': result.model_dump_json()}})
    monkeypatch.setattr(Ollama, 'post', successful_response)
    model = Ollama('http://localhost:1')
    model.chat(model.small, 'system', {'request': 'ACID'}, schema)
    assert captured == [limit]
    model.close()

@pytest.mark.parametrize('count,limit', [(6, 1080), (30, 2000)])
def test_validator_budget_fits_statement_count(monkeypatch, count, limit):
    captured = []
    def response(self, path, payload, decode):
        captured.append(payload['options']['num_predict'])
        return Verdict(checks=[], approved=True)
    monkeypatch.setattr(Ollama, 'post', response)
    model = Ollama('http://localhost:1')
    try:
        model.chat(model.validator, 'system', {'statements': [{}] * count}, Verdict)
        assert captured == [limit]
    finally:
        model.close()

@pytest.mark.parametrize('status,body', [(500, '<html>Error</html>'), (200, '{'), (200, '{}')])
def test_upstream_errors_are_explicit(status, body):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body.encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = Ollama(f'http://127.0.0.1:{server.server_port}')
    try:
        with pytest.raises(ModelError):
            model.route('ACID')
    finally:
        model.close()
        server.shutdown()
        server.server_close()

def test_streamlit_user_can_complete_quiz(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import studyflow.models
    import prometheus_client
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr(prometheus_client, 'start_http_server', lambda *args, **kwargs: None)
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# ACID\nCOMMIT фиксирует транзакцию.')
    m.close()
    model = Models()
    model.close = lambda: None
    monkeypatch.setattr(studyflow.models, 'Ollama', lambda: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    app.text_area[0].set_value('Проверь ACID')
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert quiz_radios(app)[0].label == 'Что фиксирует транзакцию?'
    assert len(quiz_radios(app)) == 3
    for radio in quiz_radios(app):
        radio.set_value(0)
    click_button(app, 'Проверить ответы')
    assert not app.exception
    assert any('Верно: 3 из 3' in message.value for message in app.success)

def test_schema_failure_shows_user_action_instead_of_internal_error(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import studyflow.models
    import prometheus_client
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr(prometheus_client, 'start_http_server', lambda *args, **kwargs: None)
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# ACID\nCOMMIT фиксирует транзакцию.')
    m.close()
    class FailingModels(Models):
        def chat(self, model, system, context, schema):
            raise ModelError('Локальная модель дважды вернула ответ неверного формата (Plan). Ответ не сохранён.')
        def close(self):
            pass
    monkeypatch.setattr(studyflow.models, 'Ollama', FailingModels)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    app.text_area[0].set_value('Проверь ACID')
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert any('повторите запрос кнопкой под полем' in message.value for message in app.error)
    assert not any('схеме' in message.value for message in app.error)

def test_quick_quiz_button_starts_quiz(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import studyflow.models
    import prometheus_client
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr(prometheus_client, 'start_http_server', lambda *args, **kwargs: None)
    m = Memory(tmp_path / 'memory.sqlite')
    m.ingest('course.md', '# ACID\nCOMMIT фиксирует транзакцию.')
    m.close()
    model = Models()
    model.close = lambda: None
    monkeypatch.setattr(studyflow.models, 'Ollama', lambda: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    select_activity(app, 'Пройти тест')
    click_button(app, 'Создать тест')
    assert not app.exception
    assert quiz_radios(app)[0].label == 'Что фиксирует транзакцию?'
