from pathlib import Path

from streamlit.testing.v1 import AppTest

from studyflow.memory import Memory
from test_studyflow import Models
from ui_helpers import click_button, select_activity, select_page
from studyflow.presentation import display_topic, format_plan

ROOT = Path(__file__).resolve().parents[1]


def test_plan_format_splits_inline_days_into_headings():
    text = 'День 1: Изучить темы. День 2: Повторить материал. День 3 — пройти тест.'
    formatted = format_plan(text)
    assert formatted.count('### День') == 3
    assert '### День 1\n\nИзучить темы.' in formatted
    assert '### День 2\n\nПовторить материал.' in formatted
    assert '### День 3\n\nпройти тест.' in formatted
    assert format_plan('Прочитать раздел и повторить') == 'Прочитать раздел и повторить'
    assert display_topic('3. Основные HTTP-методы') == 'Основные HTTP-методы'
    assert display_topic('Транзакции') == 'Транзакции'


def test_first_visit_example_and_task_actions(monkeypatch, tmp_path):
    import studyflow.models
    class NoGeneration:
        def __init__(self):
            raise AssertionError('Navigation and loading the example must not generate')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setattr(studyflow.models, 'Ollama', NoGeneration)
    app = AppTest.from_file(ROOT / 'app.py').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Получить ответ').disabled
    assert any('Начните с конспекта' in i.value for i in app.info)
    click_button(app, 'Попробовать на примере')
    assert not app.exception
    assert not next(b for b in app.button if b.label == 'Получить ответ').disabled
    memory = Memory(tmp_path / 'memory.sqlite')
    assert len(memory.documents()) == 1
    memory.close()
    select_activity(app, 'Составить план')
    assert app.number_input[0].value == 3
    assert any(b.label == 'Составить план' for b in app.button)
    select_activity(app, 'Пройти тест')
    assert not app.number_input
    assert app.text_area[0].label == 'Тема теста (необязательно)'
    assert any(b.label == 'Создать тест' for b in app.button)
    select_page(app, 'История')
    assert not app.text_area
    assert any('сохраняться здесь' in i.value for i in app.info)


def test_plan_action_uses_topic_and_number_of_days(monkeypatch, tmp_path):
    import studyflow.models
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    memory = Memory(tmp_path / 'memory.sqlite')
    memory.ingest('course.md', '# Транзакции\nCOMMIT фиксирует транзакцию.')
    memory.close()
    class ExplicitPlan(Models):
        def route(self, request):
            raise AssertionError('The plan button already specifies the mode')
    model = ExplicitPlan(mode='plan', query='COMMIT')
    model.close = lambda: None
    monkeypatch.setattr(studyflow.models, 'Ollama', lambda: model)
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=20).run()
    select_activity(app, 'Составить план')
    app.text_area[0].set_value('COMMIT')
    app.number_input[0].set_value(5)
    click_button(app, 'Составить план')
    assert not app.exception
    context = next(context for name, context in model.calls if name == model.writer)
    assert context['mode'] == 'plan'
    assert 'COMMIT' in context['request'] and '5 дней' in context['request']
    assert any('План готов' in s.value for s in app.success)
    select_activity(app, 'Пройти тест')
    assert not any('План готов' in s.value for s in app.success)


def test_empty_question_explains_next_action(monkeypatch, tmp_path):
    import studyflow.models
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    memory = Memory(tmp_path / 'memory.sqlite')
    memory.ingest('course.md', '# Тема\nФакт из конспекта.')
    memory.close()
    class NoGeneration:
        def __init__(self):
            raise AssertionError('An empty question must not call a model')
    monkeypatch.setattr(studyflow.models, 'Ollama', NoGeneration)
    app = AppTest.from_file(ROOT / 'app.py').run()
    click_button(app, 'Получить ответ')
    assert not app.exception
    assert any('Напишите вопрос в поле выше' in w.value for w in app.warning)
