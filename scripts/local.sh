#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://127.0.0.1:11434
export MODEL_SMALL=qwen3.5:0.8b
export MODEL_WRITER=qwen3.5:2b
export MODEL_VALIDATOR=gemma4:e2b
export MODEL_ROUTER=d3:2b
export MODEL_CONTEXT=4096
export MODEL_THREADS=6
export MODEL_KEEP_ALIVE=5m
exec .venv/bin/python -m studyflow.serve --server.headless=true --server.address=127.0.0.1 --browser.gatherUsageStats=false
