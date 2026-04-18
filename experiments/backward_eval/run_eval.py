"""Score each backward model against the hand-validated gold.

Takes `out/eval_items.gold.csv` (with the `gold_english` column edited by the
annotator), runs each model's `SentenceToEnglishTool` over the same 40
structured sentences, and computes three metrics:

  1. chrF         — character-ngram F-score vs gold.
  2. COMET        — reference-based COMET (src=ref=gold, mt=model output).
  3. Placeholder preservation rate — for masked items, fraction of [NOUN] /
     [VERB] tokens in the input that appear verbatim in the output.

Output: `out/backward_scored.csv` with per-item rows +
`out/backward_summary.csv` with per-model aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import sacrebleu  # type: ignore[import-untyped]

from yaduha.agent import Agent
from yaduha.agent.ollama import OllamaAgent
from yaduha.agent.openai import OpenAIAgent
from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence

load_dotenv()

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

PLACEHOLDER_RE = re.compile(r"\[(?:NOUN|VERB)\]")


def parse_structured(json_str: str):
    d = json.loads(json_str)
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def placeholder_rate(input_str: str, output_str: str) -> tuple[int, int]:
    """Returns (preserved, expected). For each [NOUN]/[VERB] in input, check if
    at least that many copies appear in output."""
    from collections import Counter
    inp = Counter(PLACEHOLDER_RE.findall(input_str))
    out = Counter(PLACEHOLDER_RE.findall(output_str))
    preserved = sum(min(out[k], v) for k, v in inp.items())
    expected = sum(inp.values())
    return preserved, expected


def make_agent(model: str, ollama_url: str) -> Agent:
    if model.startswith("gpt-"):
        return OpenAIAgent(model=model, api_key=os.environ["OPENAI_API_KEY"],
                           temperature=0.0)
    return OllamaAgent(model=model, base_url=ollama_url, temperature=0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default=str(OUT / "eval_items.gold.csv"))
    p.add_argument("--models", nargs="+", default=[
        "gpt-4o-mini",
        "llama3.2:1b", "llama3.2:3b", "llama3.1:8b",
        "qwen2.5:3b", "qwen2.5:7b",
    ])
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--no-comet", action="store_true")
    p.add_argument("--out-scored", default=str(OUT / "backward_scored.csv"))
    p.add_argument("--out-summary", default=str(OUT / "backward_summary.csv"))
    args = p.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"missing gold file: {gold_path}", file=sys.stderr)
        print("Build it first with build_gold_seed.py, then edit the "
              "gold_english column and save as *.gold.csv.", file=sys.stderr)
        return 2

    with gold_path.open() as f:
        items = list(csv.DictReader(f))
    print(f"loaded {len(items)} gold items", file=sys.stderr)

    language = LanguageLoader.load_language("ovp")
    sentence_types = language.sentence_types

    # Pre-parse structured sentences
    parsed: list[Any] = []
    for it in items:
        try:
            parsed.append(parse_structured(it["structured_json"]))
        except Exception as e:
            print(f"failed to parse id={it['id']}: {e}", file=sys.stderr)
            parsed.append(None)

    # Run each model
    per_item_rows: list[dict[str, Any]] = []
    per_model: dict[str, list[dict[str, Any]]] = {}

    for model in args.models:
        print(f"\n=== {model} ===", file=sys.stderr)
        agent = make_agent(model, args.ollama_url)
        s2e = SentenceToEnglishTool(agent=agent, SentenceType=sentence_types)
        per_model[model] = []
        t0 = time.time()
        for i, (it, s) in enumerate(zip(items, parsed)):
            if s is None:
                continue
            try:
                out = s2e(s).content.strip()
                err = None
            except Exception as e:
                out = ""
                err = f"{type(e).__name__}: {e}"

            gold = it.get("gold_english", "").strip()
            chrf = sacrebleu.sentence_chrf(out, [gold]).score / 100.0 if gold and out else 0.0
            preserved, expected = placeholder_rate(it["structured_json"], out)
            per_item_rows.append({
                "model": model,
                "id": it["id"],
                "kind": it["kind"],
                "tags": it["tags"],
                "gold": gold,
                "mt": out,
                "chrf": round(chrf, 3),
                "placeholder_preserved": preserved,
                "placeholder_expected": expected,
                "error": err or "",
            })
            per_model[model].append(per_item_rows[-1])
            print(
                f"  [{i+1}/{len(items)}] {it['kind']:<22s} chrf={chrf:.3f} "
                f"phld={preserved}/{expected} :: {out[:60]}",
                file=sys.stderr,
            )
        print(f"  model done in {time.time() - t0:.1f}s", file=sys.stderr)

    # COMET: batch compute per model
    if not args.no_comet:
        print("\ncomputing COMET (loads model on first run)...", file=sys.stderr)
        from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]

        path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(path)
        data = []
        keep_indices = []
        for idx, r in enumerate(per_item_rows):
            if r["gold"] and r["mt"]:
                data.append({"src": r["gold"], "mt": r["mt"], "ref": r["gold"]})
                keep_indices.append(idx)
        if data:
            pred = comet_model.predict(data, batch_size=32, gpus=1)
            for idx, score in zip(keep_indices, pred.scores):
                per_item_rows[idx]["comet"] = round(float(score), 3)

    # Write per-item scored CSV
    out_scored = Path(args.out_scored)
    out_scored.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "id", "kind", "tags", "gold", "mt", "chrf"]
    if not args.no_comet:
        cols.append("comet")
    cols += ["placeholder_preserved", "placeholder_expected", "error"]
    with out_scored.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_item_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote per-item: {out_scored}", file=sys.stderr)

    # Write per-model summary
    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    summary_rows = []
    for model, rows in per_model.items():
        chrfs = [r["chrf"] for r in rows if isinstance(r["chrf"], float)]
        comets = [r.get("comet") for r in rows if isinstance(r.get("comet"), float)]
        preserved = sum(r["placeholder_preserved"] for r in rows)
        expected = sum(r["placeholder_expected"] for r in rows)
        phld_rate = preserved / expected if expected else 1.0
        summary_rows.append({
            "model": model,
            "n": len(rows),
            "chrf_mean": round(mean(chrfs), 3) if chrfs else "",
            "comet_mean": round(mean(comets), 3) if comets else "",
            "placeholder_rate": round(phld_rate, 3),
            "placeholder_preserved": preserved,
            "placeholder_expected": expected,
        })

    out_summary = Path(args.out_summary)
    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"wrote summary : {out_summary}", file=sys.stderr)

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"{'model':<20s}  {'chrf':>6s}  {'comet':>6s}  {'phld_rate':>10s}",
          file=sys.stderr)
    for r in summary_rows:
        comet_str = f"{r['comet_mean']}" if r['comet_mean'] != "" else "  -   "
        print(f"{r['model']:<20s}  {r['chrf_mean']:>6}  {comet_str:>6}  "
              f"{r['placeholder_rate']:>10.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
