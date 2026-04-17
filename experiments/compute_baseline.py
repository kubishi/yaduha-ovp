"""Compute null-distribution baseline scores over all unrelated-sentence pairs.

For each metric (BLEU, chrF, chrF++, COMET), we compute scores treating every
unordered pair (a, b) from the dataset as (src=a, hyp=b, ref=a). The resulting
distribution characterises what "semantically unrelated" looks like under each
metric on this data. Mean and std are saved so analyze.py can overlay
μ and μ+3σ dashed guide lines.

Output: results/baseline.json  — {metric: [mean, std], ...}

Usage:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/compute_baseline.py
    # Skip COMET for a fast CPU-only pass:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/compute_baseline.py --no-comet
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import sacrebleu

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "evaluation_sentences.csv"
OUT = HERE / "results" / "baseline.json"


def load_sentences() -> list[str]:
    with DATA.open() as f:
        return [row["sentence"] for row in csv.DictReader(f)]


def pair_iter(sentences: list[str]):
    """Unordered pairs (i < j). Each pair scored twice (a,b) and (b,a) to
    symmetrise since sentence BLEU/chrF are not symmetric."""
    for a, b in itertools.combinations(sentences, 2):
        yield a, b
        yield b, a


def compute_surface_baseline(sentences: list[str], metric_fn, label: str) -> tuple[float, float]:
    t0 = time.time()
    scores = []
    for a, b in pair_iter(sentences):
        scores.append(metric_fn(b, a))
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores)
    print(f"  {label:<8s} n={len(scores):>6d}  mean={mean:.4f}  std={std:.4f}  "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)
    return mean, std


def compute_comet_baseline(sentences: list[str], batch_size: int) -> tuple[float, float]:
    from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]

    t0 = time.time()
    print("  loading COMET model...", file=sys.stderr)
    path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(path)
    data = []
    for a, b in pair_iter(sentences):
        data.append({"src": a, "mt": b, "ref": a})
    print(f"  scoring {len(data)} pairs on GPU (batch={batch_size})...", file=sys.stderr)
    out = model.predict(data, batch_size=batch_size, gpus=1)
    scores = list(out.scores)
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores)
    print(f"  comet    n={len(scores):>6d}  mean={mean:.4f}  std={std:.4f}  "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)
    return mean, std


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-comet", action="store_true")
    p.add_argument("--comet-batch", type=int, default=64)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    sentences = load_sentences()
    n = len(sentences)
    print(f"dataset: {n} sentences → {n * (n - 1)} ordered pairs "
          f"({n * (n - 1) // 2} unordered)", file=sys.stderr)

    result: dict[str, list[float]] = {}

    print("\nsurface metrics (CPU):", file=sys.stderr)
    result["bleu"] = list(
        compute_surface_baseline(sentences,
                                 lambda h, r: sacrebleu.sentence_bleu(h, [r]).score / 100.0,
                                 "bleu"))
    result["chrf"] = list(
        compute_surface_baseline(sentences,
                                 lambda h, r: sacrebleu.sentence_chrf(h, [r]).score / 100.0,
                                 "chrf"))
    result["chrfpp"] = list(
        compute_surface_baseline(sentences,
                                 lambda h, r: sacrebleu.sentence_chrf(h, [r], word_order=2).score / 100.0,
                                 "chrfpp"))

    if not args.no_comet:
        print("\nCOMET (GPU):", file=sys.stderr)
        result["comet"] = list(compute_comet_baseline(sentences, args.comet_batch))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
