import json
import math
import os
import re
import time
import httpx
from .contracts import Draft, Plan, QuizDraft, QuizProposal, Verdict
from .telemetry import CALLS, TOKENS, LATENCY, event

OUTPUT_LIMITS = {Plan: 96, Draft: 600, QuizDraft: 900, QuizProposal: 600, Verdict: 700}
MAX_REVIEW_TOKENS = 2000

class ModelError(RuntimeError):
    pass

class Ollama:
    def __init__(self, base_url=None, timeout=180):
        self.http = httpx.Client(base_url=base_url or os.getenv('OLLAMA_URL', 'http://localhost:11434'), timeout=timeout, trust_env=False)
        self.small = os.getenv('MODEL_SMALL', 'qwen3.5:0.8b')
        self.writer = os.getenv('MODEL_WRITER', 'qwen3.5:2b')
        self.validator = os.getenv('MODEL_VALIDATOR', 'gemma4:e2b')
        self.router = os.getenv('MODEL_ROUTER', 'd3:2b')
        self.context = int(os.getenv('MODEL_CONTEXT', '4096'))
        self.threads = int(os.getenv('MODEL_THREADS', '6'))
        self.keep_alive = os.getenv('MODEL_KEEP_ALIVE', '5m')
        self.call_timeout = timeout
        self.request_timeout = float(os.getenv('REQUEST_TIMEOUT_SECONDS', '180'))
        self.deadline = None
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            self.http.close()
            raise ValueError('REQUEST_TIMEOUT_SECONDS должен быть положительным')
        if re.match(r'[a-z]+', self.writer).group() == re.match(r'[a-z]+', self.validator).group():
            self.http.close()
            raise ValueError('Валидатор должен использовать другое семейство моделей')

    def close(self):
        self.http.close()

    def begin_request(self):
        self.deadline = time.monotonic() + self.request_timeout

    def metadata(self):
        try:
            return {name: self.http.get(path).raise_for_status().json()
                    for name, path in [('version', '/api/version'), ('models', '/api/tags')]}
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError('Не удалось сохранить версию Ollama и digest моделей') from exc

    def post(self, path, payload, decode):
        model = payload['model']
        with event('llm', model=model, endpoint=path), LATENCY.labels(model).time():
            try:
                remaining = self.call_timeout if self.deadline is None else min(self.call_timeout, self.deadline - time.monotonic())
                if remaining <= 0:
                    raise ModelError('Истекло общее время запроса. Ответ не сохранён. Дождитесь освобождения модели и повторите запрос.')
                response = self.http.post(path, json=payload, timeout=remaining)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or 'error' in data:
                    raise ValueError('invalid model response')
                result = decode(data)
            except ModelError:
                CALLS.labels(model, 'error').inc()
                raise
            except httpx.TimeoutException as exc:
                CALLS.labels(model, 'error').inc()
                raise ModelError('Истекло время ожидания модели. Ответ не сохранён. Попробуйте более короткий запрос после завершения текущей обработки.') from exc
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                CALLS.labels(model, 'error').inc()
                raise ModelError(f'Ollama: {type(exc).__name__}. Проверьте сервис, наличие модели и контракт {path}.') from exc
            CALLS.labels(model, 'ok').inc()
            return result

    def route(self, request):
        def decode(data):
            try:
                choice = data['answers']['mode']['choice']
                if choice not in {'explain', 'plan', 'quiz'}:
                    raise ValueError('unknown choice')
                for direction, key in [('input', 'input_tokens'), ('output', 'output_tokens')]:
                    TOKENS.labels(self.router, direction).inc(data.get('usage', {}).get(key, 0))
                return choice
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError('Некорректный ответ System One') from exc
        return self.post('/v1/systemone', {'model': self.router, 'state': request, 'questions': {'mode': {'type': 'choice', 'instructions': 'Выберите режим учебного помощника', 'criteria': {'explain': 'Объяснить материал или ответить на вопрос', 'plan': 'Составить расписание подготовки', 'quiz': 'Создать проверочный тест'}}}}, decode)

    def chat(self, model, system, context, schema):
        def decode(data):
            try:
                result = schema.model_validate_json(data['message']['content'])
                for direction, key in [('input', 'prompt_eval_count'), ('output', 'eval_count')]:
                    TOKENS.labels(model, direction).inc(data.get(key, 0))
                return result
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError('Модель вернула ответ, не соответствующий схеме') from exc
        context_size = max(self.context, 8192) if schema is Verdict else self.context
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)},
        ]


        output_limit = OUTPUT_LIMITS[schema]
        if schema is Verdict:
            output_limit = min(MAX_REVIEW_TOKENS, max(output_limit, 160 * len(context.get('statements', [])) + 120))
        output_schema = schema.model_json_schema()
        if schema is QuizProposal:
            output_schema['$defs']['ProposedQuestion']['properties']['evidence_id']['enum'] = [item['id'] for item in context['evidence']]
        payload = {'model': model, 'messages': messages, 'format': output_schema, 'stream': False, 'think': False, 'keep_alive': self.keep_alive, 'options': {'num_ctx': context_size, 'num_thread': self.threads, 'num_predict': output_limit, 'temperature': 0.1}}
        try:
            return self.post('/api/chat', payload, decode)
        except ModelError as exc:


            if str(exc) != 'Модель вернула ответ, не соответствующий схеме':
                raise
            payload = {**payload, 'messages': [
                {'role': 'system', 'content': system + '\nПредыдущий ответ не прошёл проверку структуры. Верните один JSON-объект, строго соответствующий переданной JSON Schema; не добавляйте текст до или после него.'},
                messages[1],
            ]}
            try:
                return self.post('/api/chat', payload, decode)
            except ModelError as retry_exc:
                if str(retry_exc) == 'Модель вернула ответ, не соответствующий схеме':
                    raise ModelError(
                        f'Локальная модель дважды вернула ответ неверного формата ({schema.__name__}). '
                        'Ответ не сохранён. Повторите запрос; если ошибка повторится, упростите формулировку.'
                    ) from retry_exc
                raise
