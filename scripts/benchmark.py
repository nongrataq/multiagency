import argparse
import json
import time
from pathlib import Path
import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3.5:2b')
parser.add_argument('--url', default='http://127.0.0.1:11434')
args = parser.parse_args()
with httpx.Client(base_url=args.url, timeout=300, trust_env=False) as client:
    version = client.get('/api/version').raise_for_status().json()
    tags = client.get('/api/tags').raise_for_status().json()
    start = time.monotonic()
    result = client.post('/api/chat', json={'model': args.model, 'messages': [{'role': 'user', 'content': 'Объясни по-русски, что такое транзакция базы данных, в трёх предложениях.'}], 'stream': False, 'think': False, 'keep_alive': 0, 'options': {'num_ctx': 4096, 'num_predict': 160, 'num_thread': 6, 'temperature': 0.1}}).raise_for_status().json()
    duration = time.monotonic() - start
seconds = result.get('eval_duration', 0) / 1e9
record = dict(model=args.model, wall_seconds=duration, tokens_per_second=result.get('eval_count', 0) / seconds if seconds else None, ollama=version, installed_models=tags, response=result)
Path('data').mkdir(exist_ok=True)
safe_name = args.model.replace(':', '-')
Path(f'data/benchmark-{safe_name}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
with Path('data/benchmarks.jsonl').open('a') as output:
    output.write(json.dumps(record, ensure_ascii=False) + '\n')
print(json.dumps(record, ensure_ascii=False, indent=2))
