import json
import logging
import os
import sqlite3
import threading
from uuid import uuid4
from pathlib import Path
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from studyflow.memory import Memory
from studyflow.models import Ollama, ModelError
from studyflow.flow import build_graph, run
from studyflow.telemetry import configure
from studyflow.presentation import display_topic, format_plan

st.set_page_config(page_title='StudyFlow', page_icon='📚', layout='wide', initial_sidebar_state='expanded')
st.title('StudyFlow')
st.caption('Разберитесь в конспекте, составьте план подготовки и проверьте себя.')
data = Path(os.getenv('DATA_DIR', 'data'))
data.mkdir(exist_ok=True, parents=True)
configure(data)
memory = Memory(data / 'memory.sqlite')
thread = 'student'

@st.cache_resource
def request_lock():
    return threading.Lock()

def open_quiz(run_id):

    st.session_state['open_quiz_run'] = run_id
    st.session_state['page'] = 'Занятие'
    st.session_state['activity'] = 'Пройти тест'

def use_example():
    st.session_state['load_example'] = True
    st.session_state['page'] = 'Занятие'

def change_activity():
    st.session_state.pop('result', None)
    st.session_state.pop('request_error', None)

try:
    example = Path(__file__).resolve().parent / 'examples' / 'databases.md'
    if st.session_state.pop('load_example', False):
        memory.ingest('Пример — базы данных.md', example.read_text())
        st.session_state['material_notice'] = 'Пример добавлен. Теперь выберите задачу и нажмите кнопку под ней.'
    pending_quiz = st.session_state.pop('open_quiz_run', None)
    if pending_quiz:
        try:
            st.session_state['result'] = {'run': pending_quiz, 'status': 'approved', 'sources': memory.artifact_sources(pending_quiz)}
            st.session_state.pop('request_error', None)
        except ValueError:
            st.session_state.pop('result', None)
            st.session_state['request_error'] = 'Этот тест больше недоступен: материалы изменились. Создайте новый тест.'
    with st.sidebar:
        page = st.radio('Раздел', ['Занятие', 'История', 'Прогресс', 'Состояние сервиса'], key='page')
        st.header('1. Добавьте конспект')
        uploaded = st.file_uploader('Файл TXT или Markdown, до 1 МБ', type=['md', 'txt'], max_upload_size=1, help='Загружайте файлы по одному. Они сохраняются между посещениями приложения.')
        if uploaded:
            try:
                version = memory.ingest(uploaded.name, uploaded.getvalue().decode('utf-8'))
                upload_key = (uploaded.name, version)
                if st.session_state.get('studyflow_uploaded') != upload_key:
                    st.session_state['studyflow_uploaded'] = upload_key
                    st.success('Конспект сохранён. Можно начинать занятие.')
            except (ValueError, UnicodeError) as exc:
                st.error(str(exc))
        docs = memory.documents()
        if docs:
            st.caption(f'Сохранено материалов: {len(docs)}')
            with st.expander('Ваши материалы'):
                for document in docs:
                    st.write(document['name'])
        st.caption('Материалы и результаты сохраняются автоматически.')

    if page == 'Занятие':
        if not docs:
            st.info('Начните с конспекта: загрузите файл в панели слева или попробуйте готовый пример по базам данных.')
            if example.exists():
                st.button('Попробовать на примере', on_click=use_example, key='use_example')
        else:
            notice = st.session_state.pop('material_notice', None)
            if notice:
                st.success(notice)
        st.subheader('2. Что хотите сделать?')
        activity = st.selectbox('Выберите задачу', ['Задать вопрос', 'Составить план', 'Пройти тест'], key='activity', on_change=change_activity)
        quiz_mode = activity == 'Пройти тест'
        plan_mode = activity == 'Составить план'
        if quiz_mode:
            st.write('Получите 3 вопроса по конспекту. После ответов увидите оценку и объяснения.')
            field_label, placeholder, action = 'Тема теста (необязательно)', 'Например: транзакции или первичные ключи', 'Создать тест'
        elif plan_mode:
            st.write('Распределим темы по дням и включим повторение.')
            field_label, placeholder, action = 'Тема плана (необязательно)', 'Например: базы данных', 'Составить план'
        else:
            st.write('Спросите о теме из конспекта. В ответе можно будет открыть использованные разделы.')
            field_label, placeholder, action = 'Ваш вопрос', 'Например: чем COMMIT отличается от ROLLBACK?', 'Получить ответ'
        with st.form('request_form'):
            request = st.text_area(field_label, placeholder=placeholder, key='study_request')
            days = st.number_input('На сколько дней составить план?', min_value=1, max_value=7, value=3, help='Короткий план на 1–7 дней.') if plan_mode else None
            if quiz_mode or plan_mode:
                st.caption('Можно оставить тему пустой — используем доступные разделы загруженных материалов.')
            st.caption('Подготовка может занять несколько минут. Дождитесь результата; повторное нажатие может прервать текущий запрос.')
            submitted = st.form_submit_button(action, type='primary', disabled=not bool(docs))
        quiz_clicked = submitted and quiz_mode
        start_clicked = submitted and not quiz_mode
        if start_clicked or quiz_clicked:
            effective_request = ('Создай тест по теме: ' + request.strip() if request.strip() else 'Создай тест по конспекту') if quiz_clicked else request
            if plan_mode:
                day_word = 'день' if days == 1 else ('дня' if days < 5 else 'дней')
                topic = 'по теме: ' + request.strip() if request.strip() else 'по конспекту'
                effective_request = f'Составь план подготовки {topic} на {days} {day_word}'
            if not docs:
                st.warning('Сначала загрузите конспект в боковой панели.')
            elif not thread.strip():
                st.warning('Укажите идентификатор диалога.')
            elif not effective_request.strip():
                st.warning('Напишите вопрос в поле выше, например: «Что такое транзакция?»')
            elif not request_lock().acquire(blocking=False):
                st.warning('В этом приложении уже выполняется запрос. Дождитесь его завершения, прежде чем запускать следующий.')
            else:
                models = None
                stage = st.empty()
                current_stage = []
                def show_stage(message):
                    current_stage[:] = [message]
                    stage.info(message)
                request_run = uuid4().hex
                history_recorded = False
                st.session_state.pop('result', None)
                st.session_state.pop('request_error', None)
                try:
                    models = Ollama()
                    with st.spinner('Готовим результат по конспекту. Пожалуйста, дождитесь завершения…'):
                        with SqliteSaver.from_conn_string(str(data / 'checkpoints.sqlite')) as saver:
                            st.session_state['result'] = run(build_graph(memory, models, saver, on_stage=show_stage), effective_request, thread, mode='quiz' if quiz_clicked else ('plan' if plan_mode else None), run_id=request_run)
                    memory.record_interaction(st.session_state['result'], effective_request, thread)
                    history_recorded = True
                except (ModelError, ValueError, sqlite3.Error) as exc:
                    st.session_state.pop('result', None)
                    message = str(exc)
                    if 'неверного формата' in message or 'не соответствующий схеме' in message:
                        message = 'Не удалось подготовить результат. Сократите вопрос или тему и повторите запрос кнопкой под полем. Материалы сохранены.'
                    if current_stage:
                        message += ' Этап: ' + current_stage[0] + '.'
                    st.session_state['request_error'] = message
                    try:
                        memory.record_interaction({'run': request_run, 'mode': 'quiz' if quiz_clicked else 'request', 'status': 'error', 'verdict': {'issues': [message]}}, effective_request, thread)
                    except sqlite3.Error:
                        st.warning('Не удалось записать ошибку в историю. Сообщение остаётся на этой странице.')
                    history_recorded = True
                finally:
                    try:
                        if not history_recorded:
                            cancel_message = 'Запрос прерван до получения результата. Дождитесь завершения одного запроса перед повторным запуском.'
                            st.session_state['request_error'] = cancel_message
                            try:
                                memory.record_interaction({'run': request_run, 'mode': 'quiz' if quiz_clicked else 'request', 'status': 'error', 'verdict': {'issues': [cancel_message]}}, effective_request, thread)
                            except sqlite3.Error as exc:
                                logging.getLogger('studyflow').warning(json.dumps({'event': 'history_write_error', 'run': request_run, 'error_type': type(exc).__name__}))
                    finally:
                        try:
                            if models:
                                models.close()
                        finally:
                            request_lock().release()
                    stage.empty()
        if st.session_state.get('request_error'):
            detail = st.session_state['request_error']
            summary = detail.split(' Этап:')[0]
            if summary.startswith('Ollama:'):
                summary = 'Сервис подготовки ответов сейчас недоступен. Попробуйте позже; ваши материалы сохранены.'
            elif 'database is locked' in summary:
                summary = 'Хранилище сейчас занято. Дождитесь завершения другого действия и повторите запрос.'
            st.error(summary)
            if summary != detail:
                with st.expander('Подробности ошибки'):
                    st.write(detail)
        result = st.session_state.get('result')
        if result:
            st.divider()
            st.subheader('Результат занятия')
            if result['status'] == 'insufficient':
                st.warning('Не удалось найти подходящий раздел. Уточните тему словами из конспекта или добавьте материал по этому вопросу.')
            elif result['status'] == 'rejected':
                st.warning('Не удалось подготовить результат, который подтверждается конспектом. Уточните тему или добавьте более подробный материал.')
                with st.expander('Почему результат не показан'):
                    for issue in result['verdict']['issues']:
                        st.write(issue)
            else:
                artifact = memory.artifact(result['run'])
                if artifact and artifact['stale']:
                    st.warning('Материал обновился. Повторите запрос: этот результат устарел.')
                elif artifact:
                    draft = json.loads(artifact['body'])
                    if artifact['kind'] != 'quiz':
                        st.success('План готов' if artifact['kind'] == 'plan' else 'Ответ готов')
                        st.markdown(format_plan(draft['text']) if artifact['kind'] == 'plan' else draft['text'])
                    if artifact['kind'] == 'quiz':
                        grades = memory.grades(result['run'])
                        if grades:
                            st.success(f"Верно: {sum(g['correct'] for g in grades)} из {len(grades)}")
                            st.caption('Результат сохранён. Этот тест уже пройден; для новой попытки создайте новый тест.')
                            by_question = {g['question']: g for g in grades}
                            for i, q in enumerate(draft['questions']):
                                grade = by_question[i]
                                st.markdown(f"**{i + 1}. {q['text']}**")
                                selected = grade['selected']
                                st.write('Ваш ответ: ' + (q['options'][selected] if selected is not None else 'не сохранён в старой версии'))
                                st.write('Правильный ответ: ' + q['options'][q['correct']])
                                st.write('Верно' if grade['correct'] else 'Ошибка')
                                st.write(q['explanation'])
                        else:
                            st.success('Тест готов')
                            st.caption(f"Вопросов: {len(draft['questions'])}")
                            st.info('Выберите по одному варианту в каждом вопросе, затем нажмите «Проверить ответы». Оценку получите сразу.')
                            with st.form('quiz_' + result['run']):
                                answers = []
                                for i, q in enumerate(draft['questions']):
                                    st.caption(f"Вопрос {i + 1} из {len(draft['questions'])}")
                                    answers.append(st.radio(q['text'], range(len(q['options'])), format_func=lambda j, q=q: q['options'][j], index=None, key=f"{result['run']}_{i}"))
                                if st.form_submit_button('Проверить ответы'):
                                    try:
                                        memory.grade(result['run'], answers)
                                        st.rerun()
                                    except ValueError as exc:
                                        st.error(str(exc))
                    with st.expander('Использованные разделы конспекта'):
                        sources_by_id = {source['id']: source for source in result['sources']}
                        shown = set()
                        for c in draft['citations']:
                            source = sources_by_id.get(c['source_id'])
                            if not source or c['source_id'] in shown:
                                continue
                            shown.add(c['source_id'])
                            heading = source['heading']
                            if heading.endswith(' [1]') and len(source['body']) < 1800:
                                heading = heading[:-4]
                            st.markdown(f'**{heading}**')
                            st.markdown(source['body'])
    if page == 'История':
        st.subheader('История занятий')
        st.caption('Здесь остаются результаты и незавершённые запросы. Откройте тест, чтобы пройти его или посмотреть оценку.')
        entries = memory.history(thread)
        if not entries:
            st.info('Ваши планы, объяснения и тесты будут сохраняться здесь.')
        else:
            for entry in entries:
                kind = {'plan': 'План', 'explain': 'Объяснение', 'quiz': 'Тест', 'request': 'Запрос'}.get(entry['kind'], entry['kind'])
                request_title = entry['request'] or 'Ранее сохранённый ответ'
                timestamp = (entry['created'] + ' UTC') if entry['created'] else 'ранее'
                status_label = {'approved': 'Готово', 'rejected': 'Не прошёл проверку', 'insufficient': 'Материал не найден', 'error': 'Не завершён'}.get(entry['status'], 'Не завершён')
                with st.expander(f'{kind} · {status_label} · {timestamp} · {request_title[:70]}'):
                    if entry['request']:
                        st.caption('Запрос')
                        st.write(entry['request'])
                    else:
                        st.caption('Старый ответ: исходный запрос тогда не сохранялся.')
                    if entry['status'] == 'rejected':
                        st.warning('Ответ не прошёл проверку и не был сохранён как материал.')
                        issues = json.loads(entry['response']).get('issues', [])
                        for issue in issues:
                            st.write(issue)
                        continue
                    elif entry['status'] == 'insufficient':
                        st.info('Для этого запроса не нашлось подходящих разделов.')
                        continue
                    elif entry['status'] != 'approved':
                        st.warning('Запрос не был завершён.')
                        for issue in json.loads(entry['response']).get('issues', []):
                            st.write(issue)
                        continue
                    if entry['kind'] == 'quiz' and not entry['stale']:
                        previous_grades = memory.grades(entry['run'])
                        st.caption(f"Пройдено · {sum(g['correct'] for g in previous_grades)} из {len(previous_grades)}" if previous_grades else 'Тест ещё не пройден')
                        st.button('Открыть тест', key='open_' + entry['run'], on_click=open_quiz, args=(entry['run'],))
                    response = json.loads(entry['response'])
                    if response.get('text'):
                        st.markdown(response['text'])
                    for question in response.get('questions', []):
                        st.markdown(f"**{question['text']}**")
                        st.markdown('\n'.join(f'- {option}' for option in question['options']))
                    citations = response.get('citations', [])
                    if citations:
                        headings = [c['section'].removesuffix(' [1]') for c in citations]
                        st.caption('Использованные разделы: ' + ', '.join(dict.fromkeys(headings)))
                    if entry['stale']:
                        st.warning('Источники обновились после этого ответа.')

    if page == 'Прогресс':
        st.subheader('Ваш прогресс')
        progress = memory.progress()
        if progress:
            st.dataframe([{'Тема': display_topic(row['topic']), 'Ответов': row['attempts'], 'Правильных, %': round(row['mastery'] * 100)} for row in progress])
            st.caption('Учитывается первый ответ на каждый вопрос. Повторное открытие теста не меняет результат.')
            st.write('Темы для повторения:', ', '.join(display_topic(r['topic']) for r in progress if r['mastery'] < 0.7) or 'нет слабых тем')
        else:
            st.info('Чтобы увидеть результаты, выберите «Пройти тест» в разделе «Занятие», создайте тест, ответьте на вопросы и нажмите «Проверить ответы».')
    if page == 'Состояние сервиса':
        st.subheader('Состояние сервиса')
        st.caption('Здесь сохраняются сообщения о доступности и ошибках сервиса.')
        notifications = data / 'alerts.jsonl'
        if notifications.exists():
            for line in notifications.read_text().splitlines()[-20:]:
                try:
                    with st.expander('Подробности уведомления'):
                        st.json(json.loads(line))
                except json.JSONDecodeError:
                    st.warning('Одно уведомление повреждено или ещё записывается. Остальные данные приложения доступны.')
        else:
            st.info('Новых уведомлений нет.')
finally:
    memory.close()
