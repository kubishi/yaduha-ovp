"""Score translation JSONL with BLEU, chrF, chrF++, and COMET.

Computes drift between `source` (original English) and each of:
  simple       — STRONG decode of structured OVP Sentence(s)
  comparator   — STRONG decode of structure with OOV vocab masked
  backwards    — STRONG decode of the OVP-language target string (round-trip)

For each metric+arm, adds <metric>_<arm> to the record (e.g., bleu_simple,
chrfpp_backwards, comet_comparator).

Usage:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/run_metrics.py \\
        --input yaduha-ovp/experiments/results/<model_tag>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sacrebleu


ARMS = ("backwards", "comparator")


def score_bleu(hyp: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0


def score_chrf(hyp: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0


def score_chrfpp(hyp: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Translations JSONL")
    p.add_argument("--output", default=None)
    p.add_argument("--no-comet", action="store_true", help="Skip COMET (fast path)")
    p.add_argument("--comet-batch", type=int, default=32)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".scored.jsonl")

    records: list[dict[str, Any]] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    valid = [r for r in records if r.get("error") is None and r.get("source")]
    print(f"total={len(records)} scoring={len(valid)}", file=sys.stderr)

    # Surface-metric scores (fast)
    for r in valid:
        src = r["source"]
        for arm in ARMS:
            hyp = r.get(arm)
            if not hyp:
                continue
            r[f"bleu_{arm}"] = score_bleu(hyp, src)
            r[f"chrf_{arm}"] = score_chrf(hyp, src)
            r[f"chrfpp_{arm}"] = score_chrfpp(hyp, src)

    # COMET: one batch call per arm, reusing the loaded model
    if not args.no_comet and valid:
        print("computing COMET (loads ~2GB model on first run)...", file=sys.stderr)
        from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]

        path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(path)

        for arm in ARMS:
            pairs: list[tuple[int, dict[str, str]]] = []
            for i, r in enumerate(valid):
                hyp = r.get(arm)
                if hyp:
                    pairs.append((i, {"src": r["source"], "mt": hyp, "ref": r["source"]}))
            if not pairs:
                continue
            data = [d for _, d in pairs]
            out = model.predict(data, batch_size=args.comet_batch, gpus=1)
            for (idx, _), score in zip(pairs, out.scores):
                valid[idx][f"comet_{arm}"] = float(score)

    with out_path.open("w") as fout:
        for r in records:
            fout.write(json.dumps(r) + "\n")

    # Summary table
    def mean(rs: list[dict], key: str) -> float:
        xs = [r[key] for r in rs if key in r]
        return sum(xs) / len(xs) if xs else float("nan")

    def summarize(label: str, rs: list[dict]) -> None:
        if not rs:
            return
        parts = [f"{label:<24s} n={len(rs):3d}"]
        for metric in ("bleu", "chrf", "chrfpp") + (("comet",) if not args.no_comet else ()):
            for arm in ARMS:
                key = f"{metric}_{arm}"
                if any(key in r for r in rs):
                    parts.append(f"{metric}_{arm[0]}={mean(rs, key):.3f}")
        print("  ".join(parts), file=sys.stderr)

    print(f"\n=== {in_path.name} ===", file=sys.stderr)
    summarize("ALL", valid)

    types: dict[str, list[dict]] = {}
    for r in valid:
        types.setdefault(r["type"], []).append(r)
    for t, rs in sorted(types.items()):
        summarize(t, rs)

    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
