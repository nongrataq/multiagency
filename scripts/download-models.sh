#!/bin/sh
set -u
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p data
printf '%s\n' waiting-for-current-download > data/download-status.txt
current_pull_pid=${1:-}
if [ -n "$current_pull_pid" ]; then
    while ps -p "$current_pull_pid" >/dev/null 2>&1; do
        sleep 10
    done
fi
failed=0
for model in qwen3.5:2b qwen3.5:0.8b gemma4:e2b d3:2b; do
    printf 'Downloading %s\n' "$model" > data/download-status.txt
    printf '\nDownloading %s\n' "$model"
    if .runtime/ollama/ollama pull "$model"; then
        printf '\nDownloaded %s\n' "$model"
    else
        printf '\nFAILED %s; continuing with the next model\n' "$model"
        failed=1
    fi
done
if [ "$failed" -eq 0 ]; then
    printf '%s\n' 'All four models downloaded. No inference was started.' > data/download-status.txt
else
    printf '%s\n' 'Queue finished with errors; inspect data/download-queue.log.' > data/download-status.txt
fi
exit "$failed"
