#!/bin/bash
# Run the 5-step datagen pipeline end-to-end.
#
# Output: yaduha-ovp/experiments/datagen/out/{forward,backward}_{train,val}.jsonl
#
# Env vars:
#   N_STRUCTURES      — number of unique structured sentences (default: 2000)
#   PER_LEMMA         — OOV sentences per OOV lemma (default: 4)
#   PER_NAME          — proper-noun records per (name, kind) (default: 3)
#   K_MIN, K_MAX      — paraphrase count range per canonical (default: 4..8)
#   PARA_MODEL        — paraphrase model (default: gpt-4o-mini; gpt-4o for higher quality)
#   BACKWARD_MODEL    — canonical/decoder model (default: gpt-4o-mini)
#   PARALLEL          — concurrency (default: 8)
#   COMET_THRESHOLD   — if set (e.g. 0.5), filter forward paraphrases by COMET sim
#   VAL_FRAC          — val split fraction (default: 0.1)
#   SEED              — RNG seed (default: 0)
#
# Each step is resumable; re-running with the same flags continues where it
# left off. To redo a step, delete its output JSONL.

set -e
cd "$(dirname "$0")/../.."

N_STRUCTURES="${N_STRUCTURES:-2000}"
PER_LEMMA="${PER_LEMMA:-4}"
PER_NAME="${PER_NAME:-3}"
K_MIN="${K_MIN:-4}"
K_MAX="${K_MAX:-8}"
PARA_MODEL="${PARA_MODEL:-gpt-4o-mini}"
BACKWARD_MODEL="${BACKWARD_MODEL:-gpt-4o-mini}"
PARALLEL="${PARALLEL:-8}"
VAL_FRAC="${VAL_FRAC:-0.1}"
SEED="${SEED:-0}"

DATAGEN=yaduha-ovp/experiments/datagen
OUT=$DATAGEN/out
mkdir -p "$OUT"

echo "=== datagen config ==="
echo "  N_STRUCTURES=$N_STRUCTURES  PER_LEMMA=$PER_LEMMA  PER_NAME=$PER_NAME  K=[$K_MIN..$K_MAX]"
echo "  PARA_MODEL=$PARA_MODEL  BACKWARD_MODEL=$BACKWARD_MODEL  PARALLEL=$PARALLEL"
echo "  VAL_FRAC=$VAL_FRAC  SEED=$SEED  COMET_THRESHOLD=${COMET_THRESHOLD:-<skip>}"
echo

UVRUN="uv run --project yaduha-ovp python"

echo "=== step 1: sample_structures ==="
$UVRUN $DATAGEN/sample_structures.py -n "$N_STRUCTURES" --seed "$SEED"

echo "=== step 2: paraphrase ==="
$UVRUN $DATAGEN/paraphrase.py \
    --backward-model "$BACKWARD_MODEL" --para-model "$PARA_MODEL" \
    --k-min "$K_MIN" --k-max "$K_MAX" --parallel "$PARALLEL"

echo "=== step 3: oov_substitutions ==="
$UVRUN $DATAGEN/oov_substitutions.py \
    --per-lemma "$PER_LEMMA" --model "$PARA_MODEL" --parallel "$PARALLEL" --seed "$SEED"

echo "=== step 3b: proper_nouns ==="
$UVRUN $DATAGEN/proper_nouns.py \
    --per-name "$PER_NAME" --model "$PARA_MODEL" --parallel "$PARALLEL" --seed "$SEED"

echo "=== step 4: decoder_pairs ==="
$UVRUN $DATAGEN/decoder_pairs.py \
    --backward-model "$BACKWARD_MODEL" --parallel "$PARALLEL"

echo "=== step 5: assemble ==="
COMET_ARG=()
if [ -n "${COMET_THRESHOLD:-}" ]; then
    COMET_ARG=(--comet-threshold "$COMET_THRESHOLD")
fi
$UVRUN $DATAGEN/assemble.py \
    --val-frac "$VAL_FRAC" --seed "$SEED" "${COMET_ARG[@]}"

echo "=== datagen complete ==="
ls -la "$OUT"/*.jsonl
