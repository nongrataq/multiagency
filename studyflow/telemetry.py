import json
import logging
import os
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

CALLS = Counter('studyflow_model_calls', 'Model calls', ['model', 'outcome'])
TOKENS = Counter('studyflow_tokens', 'Tokens', ['model', 'direction'])
LATENCY = Histogram('studyflow_model_seconds', 'Model latency', ['model'])
RUNS = Counter('studyflow_runs', 'Workflow outcomes', ['outcome'])
EVAL = Gauge('studyflow_eval_pass_ratio', 'Last evaluated pass ratio')
EVAL.set(float('nan'))
AVAILABLE = Gauge('studyflow_ollama_available', 'Ollama health')
AVAILABLE.set(float('nan'))
EVAL_VALID = Gauge('studyflow_eval_result_valid', 'Last eval result is readable and valid')
EVAL_VALID.set(float('nan'))
REVISIONS = Counter('studyflow_revisions', 'Validator requested revisions')


def configure(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('studyflow')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(Path(directory) / 'events.jsonl', maxBytes=2_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        if os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'):
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)


@contextmanager
def event(name, **attributes):
    start = time.monotonic()
    with trace.get_tracer('studyflow').start_as_current_span(name, attributes=attributes, record_exception=False, set_status_on_exception=False) as span:
        outcome = 'ok'
        try:
            yield span
        except Exception as exc:
            outcome = 'error'

            span.set_attribute('error.type', type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            logging.getLogger('studyflow').info(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=name, trace_id=format(span.get_span_context().trace_id, '032x'), outcome=outcome, seconds=time.monotonic()-start, **attributes), ensure_ascii=False))
