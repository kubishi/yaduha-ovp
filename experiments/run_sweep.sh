#!/bin/bash
# Run the dual-agent experiment across a ladder of forward-translation models.
# Usage: ./run_sweep.sh [model1 model2 ...]  — defaults to the full ladder.
#
# Env vars:
#   BACKWARD_MODEL — decoder (default: gpt-4o-mini)
#   PARALLEL       — concurrency per model (default: 4)
#   TYPES          — comma-separated sentence types to re-run (e.g. 'nominalization').
#                    If set, only those types run; prior rows of those types are
#                    dropped from the output JSONL; other types are preserved.

set -e
cd "$(dirname "$0")/../.."

MODELS=${@:-"gpt-4o-mini llama3.2:1b llama3.2:3b qwen2.5:3b llama3.1:8b qwen2.5:7b"}
BACKWARD="${BACKWARD_MODEL:-gpt-4o-mini}"
PARALLEL="${PARALLEL:-4}"
TYPES_ARG=()
if [ -n "${TYPES:-}" ]; then
    TYPES_ARG=(--types "$TYPES")
    echo "sweep: forward=[$MODELS] backward=$BACKWARD parallel=$PARALLEL types=$TYPES"
else
    echo "sweep: forward=[$MODELS] backward=$BACKWARD parallel=$PARALLEL (all types)"
fi

for model in $MODELS; do
    echo "=== $(date +%H:%M:%S)  forward=$model  backward=$BACKWARD ==="
    uv run --project yaduha-ovp python yaduha-ovp/experiments/run_translations.py \
        --forward-model "$model" --backward-model "$BACKWARD" --parallel "$PARALLEL" \
        "${TYPES_ARG[@]}"
done
echo "=== all forward models done ==="

echo "=== scoring all outputs ==="
for f in yaduha-ovp/experiments/results/*.jsonl; do
    case "$f" in *.scored.jsonl|*_sweep.log) continue ;; esac
    uv run --project yaduha-ovp python yaduha-ovp/experiments/run_metrics.py --input "$f"
done
echo "=== sweep complete ==="
