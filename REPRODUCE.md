# Reproducing the fine-tuning experiment

End-to-end reproduction of `ft-qwen2.5:3b`, the LoRA-fine-tuned Qwen2.5-3B-Instruct
that beats gpt-4o-mini on the cheat-proof comparator metric
(0.492 vs 0.461 COMET_c). Live status dashboard: <https://yaduha-status.pages.dev>.

## Hardware & environment

- **GPU** with ≥ 20 GB VRAM. We used one NVIDIA RTX 5000 Ada (32 GB); anything
  from an A100 40 GB up works. Qwen2.5-3B + LoRA trains in bf16 without
  4-bit quantization, so no `bitsandbytes` compatibility pain.
- **CUDA** 12+ or 13. The committed `uv.lock` pins torch 2.11.0+cu130.
- **Python** 3.10+.
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- **OpenAI API key** in `.env` as `OPENAI_API_KEY=sk-…` (used for the
  paraphrase/OOV/proper-noun datagen at gpt-4o-mini, and for the strong-model
  decoder at eval time).
- **Internet** to pull `Qwen/Qwen2.5-3B-Instruct` (~6 GB) from HuggingFace on
  first training run.

Total cost: **~$0.40 OpenAI API** (datagen + 150-sentence eval decoding) plus
**~1 GPU-hour**.

## Environment setup

```bash
cd yaduha-project

# Sibling repos; yaduha-ovp's pyproject points at ../yaduha as editable
git clone https://github.com/kubishi/yaduha.git
git clone -b feature/weakmodels https://github.com/kubishi/yaduha-ovp.git
cd yaduha-ovp

uv sync --group dev --group experiments --group training
cp /path/to/.env .env          # OPENAI_API_KEY=sk-…
```

## Sanity check: pipeline smoke test (~2 min, < $0.05)

Confirms every code path runs. Doesn't train — just validates the datagen and
assemble steps at tiny scale. Good first command after cloning, or after any
edit to the datagen scripts.

```bash
bash experiments/reproduce_smoke.sh
```

Expected: `=== SMOKE TEST PASSED ===` and four non-empty JSONL files in
`/tmp/yaduha_smoke/`.

## Full reproduction

Four stages. Each is resumable — re-running with the same flags skips work
already done. Delete the relevant output file to force a redo.

### 1. Generate training data (~10 min, ~$0.30)

```bash
N_STRUCTURES=750 PER_LEMMA=4 PER_NAME=3 K_MIN=2 K_MAX=4 SEED=0 PARALLEL=12 \
    bash experiments/run_datagen.sh
```

Writes six JSONL files to `experiments/datagen/out/`:
- `structures.jsonl`  — sampled structured sentences (stratified)
- `paraphrases.jsonl` — gpt-4o-mini paraphrases
- `oov_substitutions.jsonl` — hypernym substitution pairs
- `proper_nouns.jsonl` — single + coref-multi name pairs
- `decoder_pairs.jsonl` — backward-direction training pairs
- `forward_{train,val}.jsonl`, `backward_{train,val}.jsonl` — assembled SFT data

Verify schema:

```bash
uv run python experiments/datagen/validate.py
```

Expected: `ALL CHECKS PASSED` across four files, token lengths reported for
Qwen2.5-3B and Llama-3.2-3B tokenizers.

### 2. Fine-tune the forward LoRA (~45–60 min)

```bash
uv run python experiments/finetune/scripts/train.py \
    --direction forward \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 1 --grad-accum 8 --max-seq-length 2304 \
    --seed 42 --no-eval \
    --out-tag qwen2.5-3b-instruct-forward
```

Writes the adapter to `experiments/finetune/adapters/qwen2.5-3b-instruct-forward/`
(~120 MB `adapter_model.safetensors` + tokenizer).

### 3. Evaluate on the 150-sentence held-out set (~8 min)

```bash
uv run python experiments/finetune/scripts/run_finetune_eval.py \
    --adapter experiments/finetune/adapters/qwen2.5-3b-instruct-forward \
    --tag ft-qwen2.5_3b__gpt-4o-mini
```

Writes `experiments/results/ft-qwen2.5_3b__gpt-4o-mini.jsonl` — same schema
as `run_translations.py`'s output, so the existing scoring pipeline picks it up.

### 4. Score + analyze (~2 min on GPU for COMET, 30 s CPU for the rest)

```bash
uv run python experiments/run_metrics.py \
    --input experiments/results/ft-qwen2.5_3b__gpt-4o-mini.jsonl
uv run python experiments/analyze.py
```

Regenerates `experiments/plots/*.png` and `experiments/summary.csv` with
`ft-qwen2.5:3b` alongside the 5 base models + gpt-4o-mini.

## Expected results

With `--seed 42` on the fine-tune and the default gpt-4o-mini decoder, the
published run gave:

| Sentence type | COMET_c (fine-tune) | COMET_c (gpt-4o-mini) |
|---|---|---|
| subject-verb | 0.710 | 0.709 |
| subject-verb-object | 0.441 | 0.344 |
| two-verb | 0.593 | 0.533 |
| two-clause | 0.398 | 0.365 |
| complex | 0.370 | 0.340 |
| nominalization | 0.442 | 0.474 |
| **All** | **0.492** | 0.461 |

Parse failure rate: **0 / 150**.

Exact per-sentence scores drift by up to ~0.01 COMET run-to-run because
`paraphrase.py` uses temperature=0.9 (intentional — diverse training data).
Aggregate per-type medians should be stable to ~0.02.

## Layering in a different target language

The datagen pipeline is language-agnostic once `Sentence.masked_copy()` is
implemented in the target language package (see
`yaduha-ovp/yaduha_ovp/__init__.py` for the reference implementation). Any
package registered under `yaduha.languages` that provides
`sentence_types` + `get_instructions` + per-Sentence `masked_copy` can be
fed through the same `run_datagen.sh` → `train.py` → `run_finetune_eval.py`
pipeline.

## Files of note

- `experiments/run_datagen.sh` — end-to-end datagen orchestrator
- `experiments/reproduce_smoke.sh` — 2-minute sanity run (no training)
- `experiments/datagen/` — 6-step datagen pipeline (language-agnostic)
- `experiments/finetune/scripts/train.py` — HF TRL + PEFT LoRA trainer
- `experiments/finetune/scripts/run_finetune_eval.py` — drop-in replacement
  for `run_translations.py` using the fine-tuned HF model as the forward agent
- `experiments/results/` — eval output JSONL (one per model) +
  `archive/` for intermediate v1/v2 runs
