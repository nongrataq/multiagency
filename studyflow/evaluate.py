import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
from .contracts import Verdict
from .flow import ROOT, build_graph, run, prompt, review_context, grounded_verdict
from .memory import Memory
from .models import Ollama, ModelError, OUTPUT_LIMITS, MAX_REVIEW_TOKENS
from .telemetry import EVAL, configure, event

PROJECT = ROOT.parent

def atomic_json(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summary(records):
    per_model = {}
    for row in records:
        counts = per_model.setdefault(row['model'], {'passed': 0, 'total': 0})
        counts['total'] += 1
        counts['passed'] += int(row['passed'])
    for counts in per_model.values():
        counts['pass_ratio'] = counts['passed'] / counts['total']
    return {'pass_ratio': min((m['pass_ratio'] for m in per_model.values()), default=0),
            'overall_pass_ratio': sum(r['passed'] for r in records) / len(records) if records else 0,
            'per_model': per_model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--dataset', default=str(PROJECT / 'evals/cases.jsonl'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be at least 1')
    if args.repeat < 1:
        parser.error('--repeat must be at least 1')
    dataset = Path(args.dataset).read_bytes()
    cases = [json.loads(line) for line in dataset.decode('utf-8').splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error('dataset must contain at least one case')
    data = Path(os.getenv('DATA_DIR', 'data'))
    configure(data)
    evaluation_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    archive = data / 'evals' / evaluation_id
    archive.mkdir(parents=True)
    candidates = ['qwen3.5:0.8b', 'qwen3.5:2b', 'gemma4:e2b'] if args.compare else [None]
    records = []
    models = Ollama()
    writer, validator = models.writer, models.validator
    try:
        source_paths = sorted([*PROJECT.glob('studyflow/*.py'), *ROOT.rglob('*.md')])
        metadata = {'runtime': models.metadata(), 'python': platform.python_version(),
                    'dependencies': {name: version(name) for name in ['langgraph', 'pydantic', 'httpx']},
                    'dataset_sha256': hashlib.sha256(dataset).hexdigest(),
                    'lock_sha256': hashlib.sha256((PROJECT / 'requirements.lock').read_bytes()).hexdigest(),
                    'source_sha256': {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                    'models': {'planner': models.small, 'writer': writer, 'validator': validator, 'router': models.router},
                    'options': {'num_ctx': models.context, 'review_num_ctx': max(models.context, 8192), 'num_thread': models.threads, 'num_predict': {schema.__name__: limit for schema, limit in OUTPUT_LIMITS.items()}, 'review_token_policy': f'min({MAX_REVIEW_TOKENS}, max(700, 160 * statements + 120))', 'request_timeout_seconds': getattr(models, 'request_timeout', 180), 'temperature': 0.1},
                    'repeat': args.repeat}
        report = {'evaluation_id': evaluation_id, 'complete': False, 'metadata': metadata, 'records': records}
        atomic_json(archive / 'results.json', report)
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / 'memory.sqlite')
            try:
                memory.ingest('databases.md', (PROJECT / 'examples/databases.md').read_text())
                for candidate in candidates:
                    models.writer = candidate or writer
                    models.validator = 'qwen3.5:2b' if models.writer.startswith('gemma') else validator
                    for repetition in range(1, args.repeat + 1):
                        for case in cases:
                            start = time.monotonic()
                            row = {'case': case['id'], 'request': case['request'], 'rubric': case['rubric'],
                                   'model': models.writer, 'validator': models.validator, 'repetition': repetition,
                                   'expected_mode': case['mode'], 'expected_status': case['status']}
                            with event('eval', case=case['id'], model=models.writer):
                                try:
                                    result = run(build_graph(memory, models), case['request'], case['id'])
                                    row.update(run=result['run'], result=result)
                                    text = result.get('draft', {}).get('text', '').lower()
                                    checks = {'mode': result['mode'] == case['mode'], 'status': result['status'] == case['status']}
                                    if case.get('keywords'):
                                        checks['keywords'] = all(w.lower() in text for w in case['keywords'])
                                    if result['status'] == 'approved':
                                        context = review_context(case['request'], result['mode'], result['sources'], result['draft'], result['plan'])
                                        context['evaluation_rubric'] = case['rubric']
                                        judge = grounded_verdict(models.chat(models.validator, prompt('validator'), context, Verdict), context)
                                        row['judge'] = judge.model_dump()
                                        checks['judge'] = judge.approved
                                    row.update(checks=checks, passed=all(checks.values()))
                                except (ModelError, ValueError) as exc:
                                    row.update(passed=False, error=str(exc))
                            row['seconds'] = time.monotonic() - start
                            records.append(row)
                            report.update(summary(records))
                            atomic_json(archive / 'results.json', report)
                            print(json.dumps({k: v for k, v in row.items() if k not in {'result', 'request', 'rubric'}}, ensure_ascii=False), flush=True)
            finally:
                memory.close()
    finally:
        models.close()
    report['complete'] = True
    atomic_json(archive / 'results.json', report)
    atomic_json(data / 'eval-results.json', report)
    EVAL.set(report['pass_ratio'])
    raise SystemExit(0 if report['pass_ratio'] >= 0.8 else 1)


if __name__ == '__main__':
    main()
