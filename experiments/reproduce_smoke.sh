#!/bin/bash
# Full-pipeline sanity check, ~3 min and < $0.05 in OpenAI cost.
#
# Runs the same code paths as a production build, just at tiny scale. Writes
# everything to /tmp/yaduha_smoke/ so it never touches the real datagen/out
# or finetune/adapters directories.
#
# What it exercises:
#   1. sample_structures.py       — stratified structure sampling (incl. multi-clause)
#   2. paraphrase.py              — canonical English + gpt-4o-mini paraphrases
#   3. oov_substitutions.py       — positive + negative OOV pairs
#   4. proper_nouns.py            — single-mention + coref multi-clause
#   5. decoder_pairs.py           — clean + masked backward pairs
#   6. assemble.py                — SFT chat-template JSONL
#   7. validate.py                — schema + chat-template checks
#
# What it SKIPS (expensive; production-only):
#   - train.py                    — 45–60 min on one GPU
#   - run_finetune_eval.py        — requires a trained adapter
#   - run_metrics.py / analyze.py — eval scoring
#
# If this exits 0, the pipeline is wired correctly and any change to the
# datagen scripts hasn't broken a code path.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=/tmp/yaduha_smoke
rm -rf "$OUT" && mkdir -p "$OUT"

UVRUN="uv run --project yaduha-ovp python"
DATAGEN=yaduha-ovp/experiments/datagen

echo "=== 1/7 sample_structures (N=20, multi-clause 30%) ==="
$UVRUN $DATAGEN/sample_structures.py -n 20 --seed 42 --multi-frac 0.3 --oov-frac 0.4 \
    --out "$OUT/structures.jsonl"

echo "=== 2/7 paraphrase (k=2) ==="
$UVRUN $DATAGEN/paraphrase.py --input "$OUT/structures.jsonl" --output "$OUT/paraphrases.jsonl" \
    --k-min 2 --k-max 2 --parallel 8

echo "=== 3/7 oov_substitutions (per_lemma=1, limit=20) ==="
$UVRUN $DATAGEN/oov_substitutions.py --per-lemma 1 --limit 20 --parallel 8 --seed 42 \
    --output "$OUT/oov_substitutions.jsonl"

echo "=== 4/7 proper_nouns (per_name=1, limit=16) ==="
$UVRUN $DATAGEN/proper_nouns.py --per-name 1 --limit 16 --parallel 8 --seed 42 \
    --output "$OUT/proper_nouns.jsonl"

echo "=== 5/7 decoder_pairs ==="
$UVRUN $DATAGEN/decoder_pairs.py --structures "$OUT/structures.jsonl" \
    --paraphrases "$OUT/paraphrases.jsonl" --output "$OUT/decoder_pairs.jsonl" \
    --parallel 8

echo "=== 6/7 assemble ==="
$UVRUN $DATAGEN/assemble.py --paraphrases "$OUT/paraphrases.jsonl" \
    --oov "$OUT/oov_substitutions.jsonl" --proper-nouns "$OUT/proper_nouns.jsonl" \
    --decoder "$OUT/decoder_pairs.jsonl" --out-dir "$OUT" --seed 42

echo "=== 7/7 validate ==="
$UVRUN $DATAGEN/validate.py --out-dir "$OUT" --no-tokenizer

echo
echo "=== SMOKE TEST PASSED ==="
echo "output dir: $OUT"
wc -l "$OUT"/forward_train.jsonl "$OUT"/forward_val.jsonl \
      "$OUT"/backward_train.jsonl "$OUT"/backward_val.jsonl
