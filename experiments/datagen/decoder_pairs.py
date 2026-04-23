"""Step 4: build decoder (backward) training pairs.

For every structured sentence S we emit:
  - clean:  (S, natural_english)               from paraphrases.jsonl canonical
  - masked: (mask_oov(S), english_with_placeholders)
            via SentenceToEnglishTool on the masked structure (only when S has OOV)

The masked pair is critical: without it the fine-tuned decoder will smooth
over `[NOUN]`/`[VERB]` placeholders instead of preserving them.

Output schema:
    {
        "id": <int>,
        "structure_id": <int>,
        "kind": "clean" | "masked",
        "type": "sv" | "svo",
        "tags": [...],
        "structured": <pydantic dump (masked if kind=masked)>,
        "english": <text>,
        "oov_tokens": [...],   # only for masked kind
        "errors": [...]
    }

Resumable on (structure_id, kind).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, jsonl_append, jsonl_iter, make_openai  # noqa: E402

load_dotenv()


def _parse_one(d: dict):
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def emit_clean(
    structures: list[dict],
    per_struct_canon_by_id: dict[int, list[str]],
) -> list[dict]:
    """Emit one clean decoder pair per *structure* (not per record), so each
    training example teaches the backward model to render one clause."""
    out: list[dict[str, Any]] = []
    for r in structures:
        canons = per_struct_canon_by_id.get(r["id"])
        if not canons:
            continue
        if len(canons) != len(r["structured"]):
            continue  # inconsistent cache; skip
        for i, (s_dump, canon) in enumerate(zip(r["structured"], canons)):
            out.append({
                "structure_id": r["id"],
                "clause_index": i,
                "kind": "clean",
                "type": r["types"][i] if i < len(r.get("types", [])) else "?",
                "tags": r["tags"],
                "structured": s_dump,   # single dict (one clause)
                "english": canon,
                "oov_tokens": [],
                "errors": [],
            })
    return out


def render_masked_clauses(
    rec: dict[str, Any], sentence_types: tuple, backward_model: str
) -> list[dict[str, Any]]:
    """For a multi-clause record, iterate each clause and emit a masked pair
    if that clause has OOV tokens."""
    results: list[dict[str, Any]] = []
    try:
        agent = make_openai(backward_model, temperature=0.0)
        s2e = SentenceToEnglishTool(agent=agent, SentenceType=sentence_types)
        for i, d in enumerate(rec["structured"]):
            try:
                parsed = _parse_one(d)
                masked, oov = parsed.masked_copy()
                if not oov:
                    continue
                r = s2e(masked)
                results.append({
                    "structure_id": rec["id"],
                    "clause_index": i,
                    "kind": "masked",
                    "type": rec["types"][i] if i < len(rec.get("types", [])) else "?",
                    "tags": rec["tags"],
                    "structured": masked.model_dump(mode="json"),
                    "english": r.content.strip(),
                    "oov_tokens": oov,
                    "errors": [],
                })
            except Exception as e:
                results.append({
                    "structure_id": rec["id"],
                    "clause_index": i,
                    "kind": "masked",
                    "type": "?",
                    "tags": rec["tags"],
                    "structured": d,
                    "english": None,
                    "oov_tokens": [],
                    "errors": [f"{type(e).__name__}: {e}"],
                })
    except Exception as e:
        results.append({
            "structure_id": rec["id"],
            "clause_index": -1,
            "kind": "masked",
            "type": "?",
            "tags": rec.get("tags", []),
            "structured": None,
            "english": None,
            "oov_tokens": [],
            "errors": [f"{type(e).__name__}: {e}"],
        })
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--structures", default=str(OUT / "structures.jsonl"))
    p.add_argument("--paraphrases", default=str(OUT / "paraphrases.jsonl"))
    p.add_argument("--output", default=str(OUT / "decoder_pairs.jsonl"))
    p.add_argument("--backward-model", default="gpt-4o-mini")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--limit-masked", type=int, default=None,
                   help="Cap masked-pair generation (LLM-bound). None = all OOV structures.")
    args = p.parse_args()

    out_path = Path(args.output)
    structures = list(jsonl_iter(Path(args.structures)))
    if not structures:
        print(f"missing structures: {args.structures}", file=sys.stderr)
        return 2

    paraphrases = list(jsonl_iter(Path(args.paraphrases)))
    per_struct_canon_by_id = {
        r["id"]: r["per_structure_canonicals"]
        for r in paraphrases
        if r.get("per_structure_canonicals")
    }
    print(
        f"structures={len(structures)} with_canonicals={len(per_struct_canon_by_id)}",
        file=sys.stderr,
    )

    done = {(r["structure_id"], r.get("clause_index", 0), r["kind"]) for r in jsonl_iter(out_path)
            if r.get("english") is not None}

    # 1) emit clean pairs (no LLM)
    clean_recs = emit_clean(structures, per_struct_canon_by_id)
    next_id = sum(1 for _ in jsonl_iter(out_path))
    new_clean = [r for r in clean_recs
                 if (r["structure_id"], r["clause_index"], "clean") not in done]
    for r in new_clean:
        r["id"] = next_id
        next_id += 1
        jsonl_append(out_path, r)
    print(f"appended {len(new_clean)} clean pairs", file=sys.stderr)

    # 2) emit masked pairs (LLM) — one per OOV clause within each record
    language = LanguageLoader.load_language("ovp")
    sentence_types = language.sentence_types

    oov_records = [
        s for s in structures
        if any(t.startswith("oov_") for t in s["tags"])
    ]
    if args.limit_masked:
        oov_records = oov_records[: args.limit_masked]
    print(f"masked candidate records: {len(oov_records)}", file=sys.stderr)
    if not oov_records:
        return 0

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(render_masked_clauses, s, sentence_types, args.backward_model): s
            for s in oov_records
        }
        for fut in as_completed(futures):
            clauses = fut.result()
            for rec in clauses:
                key = (rec["structure_id"], rec["clause_index"], "masked")
                if key in done:
                    continue
                rec["id"] = next_id
                next_id += 1
                jsonl_append(out_path, rec)
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(oov_records) - completed) / rate if rate > 0 else float("inf")
            first = clauses[0] if clauses else {}
            status = "OK" if clauses and clauses[0].get("english") else "SKIP"
            print(
                f"[{completed}/{len(oov_records)}] {status} sid={first.get('structure_id','?')} "
                f"clauses={len(clauses)} {elapsed:6.1f}s, "
                f"{rate:4.2f}/s, ETA {eta:6.0f}s :: {(first.get('english') or '')[:55]}",
                file=sys.stderr,
            )
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
