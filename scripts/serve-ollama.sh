#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export OLLAMA_MODELS="$PWD/.runtime/models"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NO_CLOUD=1
exec .runtime/ollama/ollama serve
