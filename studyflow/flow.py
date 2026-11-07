import json
import re
from pathlib import Path
from typing import TypedDict
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from .contracts import Citation, Draft, Plan, QuizDraft, QuizProposal, Verdict
from .models import ModelError
from .telemetry import RUNS, REVISIONS, event

ROOT = Path(__file__).resolve().parent.parent / 'resources'
QUIZ_INTRO = 'Тест по загруженным материалам. Выберите один ответ на каждый вопрос.'

GENERIC_PLAN_WORDS = {
    'составь', 'составить', 'составление', 'подготовь', 'подготовить', 'подготовки',
    'подготовку', 'план', 'плана', 'планирование', 'учебный', 'учебную', 'учиться',
    'подготовься', 'подготовиться', 'по', 'для', 'на', 'из', 'моим', 'моим',
    'материалам', 'материала', 'конспекту', 'конспектам', 'теме', 'темам', 'курсу',
    'курс', 'экзамену', 'экзамена', 'экзамен', 'зачёту', 'зачету', 'день', 'дня',
    'дней', 'неделя', 'недели', 'недель', 'неделю', 'месяц', 'месяца', 'месяцев',
    'час', 'часа', 'часов', 'два', 'три', 'четыре', 'пять', 'мне', 'пожалуйста',
    'весь', 'всего', 'всему', 'всем', 'рассчитай', 'распредели', 'распиши', 'с',
    'расчётом', 'расчетом', 'учётом', 'учетом', 'занятие', 'занятия', 'занятий',
    'создай', 'создать', 'сделай', 'сделать', 'тест', 'теста', 'тесты', 'тестов',
    'проверь', 'проверить', 'вопросы', 'вопросов', 'знания', 'викторину', 'опрос',
    'загруженных', 'разделам', 'материал', 'материалы',
}


def has_explicit_topic(request):
    words = re.findall(r'\w+', request.casefold())
    return any(word not in GENERIC_PLAN_WORDS and not word.isdigit() for word in words)

class State(TypedDict, total=False):
    request: str
    mode: str
    run: str
    plan: dict
    search_query: str
    sources: list[dict]
    draft: dict
    verdict: dict
    attempts: int
    status: str
    conversation: str
    requested_mode: str | None


def prompt(role, skill=None):
    text = (ROOT / 'SOUL.md').read_text() + '\n' + (ROOT / 'MEMORY.md').read_text()
    text += '\n' + (ROOT / 'prompts' / f'{role}.md').read_text()
    if skill:
        text += '\n' + (ROOT / 'skills' / f'{skill}.md').read_text()
    return text


def bounded_sources(sources, byte_budget=6000):
    selected = []
    for source in sources:
        size = len(source['body'].encode('utf-8'))
        if size <= byte_budget:
            selected.append(source)
            byte_budget -= size
    return selected


def quiz_evidence(sources, byte_budget=4500):
    evidence = []
    for source in sources:
        for text in re.split(r'(?<=[.!?])\s+|\n+', source['body']):
            text = text.strip()
            size = len(text.encode('utf-8'))
            if text and size <= byte_budget:
                evidence.append({'id': len(evidence), 'source_id': source['id'], 'text': text})
                byte_budget -= size
            if len(evidence) == 30:
                return evidence
    return evidence


def review_context(request, mode, sources, draft, plan=None):
    statements = [] if mode == 'quiz' and draft['text'] == QUIZ_INTRO else [text.strip() for text in re.split(r'(?<=[.!?])\s+|\n+', draft['text']) if text.strip()]
    for question in draft.get('questions', []):
        statements.append(f"{question['text']} Правильный ответ: {question['options'][question['correct']]}. {question['explanation']}")
    cited_ids = {citation['source_id'] for citation in draft.get('citations', [])}
    cited_sources = [source for source in sources if source['id'] in cited_ids]
    return {'request': request, 'mode': mode, 'plan': plan or {}, 'sources': cited_sources, 'draft': draft,
            'statements': [{'id': i, 'text': text} for i, text in enumerate(statements)]}


def quote_matches(quote, body):
    def plain(text):
        text = re.sub(r'[^\w]+', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().casefold()
    normalized_quote = plain(quote)
    return bool(normalized_quote) and normalized_quote in plain(body)


def grounded_verdict(verdict, context):
    if not verdict.approved:
        return verdict
    issues = []
    expected = {statement['id'] for statement in context['statements']}
    ids = [check.statement_id for check in verdict.checks]
    if set(ids) != expected or len(ids) != len(expected):
        issues.append('Валидатор должен проверить каждое утверждение ровно один раз')
    sources = {source['id']: source for source in context['sources']}
    cited = {citation['source_id'] for citation in context['draft']['citations']}
    for check in verdict.checks:
        source = sources.get(check.source_id)
        if not source or not quote_matches(check.quote, source['body']):
            issues.append(f'Для утверждения {check.statement_id} нет точной цитаты из источника')
        if check.source_id not in cited:
            issues.append(f'Источник {check.source_id} подтверждения отсутствует в citations ответа')
        if not check.supported:
            issues.append(f'Утверждение {check.statement_id} не подтверждено: {check.reason}')
    return Verdict(checks=verdict.checks, approved=False, issues=issues) if issues else verdict


def build_graph(memory, models, checkpointer=None, on_stage=None):
    def announce(message):
        if on_stage:
            on_stage(message)

    def planner(state):
        announce('1/4 · Выбираю режим и тему поиска')
        with event('planner', run=state['run']):
            models.begin_request()
            mode = state.get('requested_mode') or models.route(state['request'])
            if mode in {'quiz', 'plan'} and not has_explicit_topic(state['request']):
                plan = Plan(query='все разделы загруженных материалов')
            else:
                plan = models.chat(models.small, prompt('planner'), {'request': state['request'], 'mode': mode, 'documents': memory.documents(), 'progress': memory.progress(), 'previous_context': state.get('conversation', '')}, Plan)
            steps = {
                'explain': [f'Найти разделы по теме «{plan.query}»', 'Подготовить объяснение с точными ссылками на найденные материалы'],
                'plan': [f'Найти учебные разделы по теме «{plan.query}»', 'Распределить темы с учётом указанного пользователем срока', 'Включить повторение и самопроверку по материалам'],
                'quiz': [f'Найти разделы по теме «{plan.query}»', 'Составить вопросы только по найденным материалам', 'Проверить правильные ответы и объяснения по источникам'],
            }[mode]
            return {'mode': mode, 'plan': {'query': plan.query, 'steps': steps}, 'attempts': 0, 'status': 'running', 'draft': {}, 'verdict': {}, 'sources': []}

    def researcher(state):
        announce('2/4 · Ищу разделы в загруженных материалах')
        with event('researcher', run=state['run']):
            query = state['plan']['query']
            if state['mode'] in {'quiz', 'plan'} and not has_explicit_topic(state['request']):
                sources = bounded_sources(memory.recent_sections())
                return {'sources': sources, 'search_query': query, 'status': 'running' if sources else 'insufficient'}
            sources = memory.search(query)
            if not sources and query.casefold() != state['request'].casefold():
                query = state['request']
                sources = memory.search(query)
            sources = bounded_sources(sources)
            return {'sources': sources, 'search_query': query, 'status': 'running' if sources else 'insufficient'}

    def teacher(state):
        announce('3/4 · Готовлю ответ' if not state['attempts'] else '3/4 · Исправляю ответ по замечаниям (последняя попытка)')
        with event('teacher', run=state['run']):
            skill = {'explain': 'explain', 'plan': 'plan', 'quiz': 'quiz'}[state['mode']]
            context = {'request': state['request'], 'mode': state['mode'], 'plan': state['plan'], 'sources': state['sources'], 'progress': memory.progress()}
            if state['attempts']:
                context.update(previous_draft=state['draft'], feedback={'issues': state['verdict']['issues']})
            if state['mode'] == 'quiz':
                evidence = quiz_evidence(state['sources'])
                if not evidence:
                    raise ModelError('В конспекте не найдено коротких фрагментов для теста. Добавьте определения или разделите текст на предложения.')
                context = {'request': state['request'], 'mode': 'quiz', 'evidence': evidence, 'progress': memory.progress()}
                if state['attempts']:
                    context['feedback'] = {'issues': state['verdict']['issues']}
                    context['previous_questions'] = [{key: question[key] for key in ('text', 'options', 'correct')} for question in state['draft']['questions']]
                proposal = models.chat(models.writer, prompt('teacher', skill), context, QuizProposal)
                by_id = {item['id']: item for item in evidence}
                sources_by_id = {source['id']: source for source in state['sources']}
                questions, cited_ids = [], set()
                for question in proposal.questions:
                    fact = by_id.get(question.evidence_id)
                    if fact is None:
                        raise ModelError('Модель выбрала отсутствующий фрагмент конспекта. Тест не сохранён.')
                    source = sources_by_id[fact['source_id']]
                    questions.append({**question.model_dump(exclude={'evidence_id'}), 'explanation': fact['text'], 'topic': re.sub(r' \[\d+\]$', '', source['heading'])})
                    cited_ids.add(source['id'])
                draft = QuizDraft(text=QUIZ_INTRO, citations=[Citation(source_id=source_id) for source_id in sorted(cited_ids)], questions=questions)
            else:
                draft = models.chat(models.writer, prompt('teacher', skill), context, Draft)


            source_by_id = {source['id']: source for source in state['sources']}
            citations = []
            for citation in draft.citations:
                source = source_by_id.get(citation.source_id)
                citations.append(
                    Citation(source_id=source['id'], version=source['version'], section=source['heading'])
                    if source else citation
                )
            draft = draft.model_copy(update={'citations': citations})
            return {'draft': draft.model_dump(), 'attempts': state['attempts'] + 1}

    def validator(state):
        announce('4/4 · Проверяю ответ по источникам')
        with event('validator', run=state['run']):
            draft = Draft.model_validate(state['draft'])
            allowed_citations = {(source['id'], source['version'], source['heading']) for source in state['sources']}
            issues = []
            invalid_citations = [
                (citation.source_id, citation.version, citation.section)
                for citation in draft.citations
                if (citation.source_id, citation.version, citation.section) not in allowed_citations
            ]
            if invalid_citations:
                allowed_source_ids = sorted(source['id'] for source in state['sources'])
                issues.append(f'Недействительная ссылка; допустимые source_id: {allowed_source_ids}')
            if state['mode'] == 'quiz' and len(draft.questions) != 3:
                issues.append('Тест должен содержать ровно 3 вопроса')
            if state['mode'] != 'quiz' and draft.questions:
                issues.append('Поле questions должно быть пустым вне режима quiz; вопрос для самопроверки можно включить в text')
            if issues:
                verdict = Verdict(checks=[], approved=False, issues=issues)
            else:
                context = review_context(state['request'], state['mode'], state['sources'], state['draft'], state['plan'])
                verdict = grounded_verdict(models.chat(models.validator, prompt('validator'), context, Verdict), context)
            if not verdict.approved:
                REVISIONS.inc()
            return {'verdict': verdict.model_dump(), 'status': 'approved' if verdict.approved else 'rejected'}

    def persist(state):
        with event('persist', run=state['run']):
            memory.save(state['run'], state['mode'], Draft.model_validate(state['draft']))
        return {'conversation': (state['request'] + '\n' + state['draft']['text'])[:2000]}

    graph = StateGraph(State)
    for name, node in [('planner', planner), ('researcher', researcher), ('teacher', teacher), ('validator', validator), ('persist', persist)]:
        graph.add_node(name, node)
    graph.add_edge(START, 'planner')
    graph.add_edge('planner', 'researcher')
    graph.add_conditional_edges('researcher', lambda state: 'teacher' if state['sources'] else END)
    graph.add_edge('teacher', 'validator')
    graph.add_conditional_edges('validator', lambda state: 'persist' if state['status'] == 'approved' else ('teacher' if state['attempts'] < 2 else END))
    graph.add_edge('persist', END)
    return graph.compile(checkpointer=checkpointer)


def run(graph, request, thread_id, mode=None, run_id=None):
    if not request.strip() or len(request) > 2000:
        raise ValueError('Запрос должен содержать от 1 до 2000 символов')
    if mode not in {None, 'explain', 'plan', 'quiz'}:
        raise ValueError('Неизвестный режим запроса')
    with event('workflow', thread=thread_id):
        try:
            result = graph.invoke({'request': request, 'run': run_id or uuid4().hex, 'requested_mode': mode}, {'configurable': {'thread_id': thread_id}, 'recursion_limit': 16})
            RUNS.labels(result['status']).inc()
            return result
        except Exception:
            RUNS.labels('error').inc()
            raise
