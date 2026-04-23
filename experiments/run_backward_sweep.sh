#!/bin/bash
# Backward-direction sweep: gpt-4o-mini as the forward translator (trusted,
# high-quality structured-JSON output), each self-hosted open model in the
# BACKWARD slot (rendering structured JSON → natural English).
#
# This is the complement to run_sweep.sh (forward sweep, self-hosted forward +
# gpt-4o-mini backward). Together they give us the first systematic evaluation
# of open-weight models at each position in the Yaduha pipeline.
#
# Output: yaduha-ovp/experiments/results/backward_sweep/<backward_model>.jsonl
#
# Env vars:
#   BACKWARD_MODELS  — models to evaluate (default: the full open ladder)
#   FORWARD          — forward model (default: gpt-4o-mini)
#   PARALLEL         — per-model concurrency (default: 4)

set -e
cd "$(dirname "$0")/../.."

MODELS=${BACKWARD_MODELS:-"llama3.2:1b llama3.2:3b qwen2.5:3b qwen2.5:7b llama3.1:8b"}
FORWARD="${FORWARD:-gpt-4o-mini}"
PARALLEL="${PARALLEL:-4}"

RESULTS="${RESULTS:-yaduha-ovp/experiments/results/backward_sweep}"
mkdir -p "$RESULTS"

echo "=== backward sweep  forward=$FORWARD  backward=[$MODELS]  parallel=$PARALLEL ==="

for bm in $MODELS; do
    tag=$(echo "$bm" | sed 's/:/_/g; s/\//_/g')
    out="$RESULTS/${tag}.jsonl"
    echo
    echo "=== $(date +%H:%M:%S)  backward=$bm  ==="
    uv run --project yaduha-ovp python yaduha-ovp/experiments/run_translations.py \
        --forward-model "$FORWARD" --backward-model "$bm" \
        --parallel "$PARALLEL" --out "$out"
done

echo
echo "=== scoring backward sweep ==="
for f in "$RESULTS"/*.jsonl; do
    case "$f" in *.scored.jsonl) continue ;; esac
    uv run --project yaduha-ovp python yaduha-ovp/experiments/run_metrics.py --input "$f"
done
echo "=== backward sweep complete ==="
