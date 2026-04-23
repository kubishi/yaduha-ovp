"""Run the dual-agent experimental pipeline over the eval dataset.

Pipeline per input sentence:
    input                  ---[WEAK model]---> structured OVP-Sentence(s)
    structured             ---[deterministic]-> target         (OVP surface; may contain [lemma] for OOV)
    structured             ---[str_masked]---> target_masked   (OOV lemmas → [NOUN]/[VERB])
    structured             ---[BACKWARD model]-> backwards       (backward LLM reads JSON → English)
    mask_oov(structured)   ---[BACKWARD model]-> comparator      (backward LLM reads JSON with OOV head/lemma masked)

The forward model is the experimental variable under test (its job: English → structured).
The backward model only renders structured English JSON to natural English — a trivial
LLM task — and never sees the OVP surface form.

Writes JSONL to results/<forward_model_tag>__<backward_model_tag>.jsonl — resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yaduha.agent import Agent
from yaduha.loader import LanguageLoader
from yaduha.tool.english_to_sentences import EnglishToSentencesTool
from yaduha.tool.sentence_to_english import SentenceToEnglishTool

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
from _common import clean, make_agent  # noqa: E402

load_dotenv()

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "evaluation_sentences.csv"
RESULTS = HERE / "results"


def load_dataset() -> list[dict[str, str]]:
    with DATA.open() as f:
        return [{"sentence": r["sentence"], "type": r["type"]} for r in csv.DictReader(f)]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for i, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                if rec.get("error") is None and rec.get("source"):
                    done.add(rec["source"])
            except Exception as e:
                print(f"warning: skipped malformed line {i} in {path.name}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
    return done


def translate_one(forward: Agent, backward: Agent, row: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    try:
        language = LanguageLoader.load_language("ovp")
        SentenceTypes = language.sentence_types

        # 1) WEAK: English → structured
        e2s = EnglishToSentencesTool(agent=forward, SentenceType=SentenceTypes)
        t_forward0 = time.time()
        forward_resp = e2s(row["sentence"])
        t_forward = time.time() - t_forward0
        structured = forward_resp.content.sentences

        # 2) Deterministic target surfaces (for display only; not scored)
        ovp_targets = [clean(str(s)) for s in structured]
        ovp_targets_masked = [clean(s.str_masked()) for s in structured]
        target = " ".join(ovp_targets)
        target_masked = " ".join(ovp_targets_masked)
        has_placeholders = target != target_masked

        # 3) BACKWARD: structured JSON → English (backwards)
        s2e = SentenceToEnglishTool(agent=backward, SentenceType=SentenceTypes)

        bw_parts: list[str] = []
        bw_pt = bw_ct = 0
        t_bw0 = time.time()
        for s in structured:
            r = s2e(s)
            bw_parts.append(clean(r.content))
            bw_pt += r.prompt_tokens
            bw_ct += r.completion_tokens
        t_bw = time.time() - t_bw0
        backwards = " ".join(bw_parts)

        # 4) BACKWARD: mask_oov(structured) → English (comparator)
        cmp_parts: list[str] = []
        cmp_pt = cmp_ct = 0
        oov_tokens: list[str] = []
        t_cmp0 = time.time()
        if has_placeholders:
            for s in structured:
                masked, oov = s.masked_copy()
                oov_tokens.extend(oov)
                r = s2e(masked)
                cmp_parts.append(clean(r.content))
                cmp_pt += r.prompt_tokens
                cmp_ct += r.completion_tokens
            comparator = " ".join(cmp_parts)
        else:
            comparator = backwards  # no OOV → comparator identical to backwards
        t_cmp = time.time() - t_cmp0

        return {
            "source": row["sentence"],
            "type": row["type"],
            "target": target,
            "target_masked": target_masked,
            "backwards": backwards,
            "comparator": comparator,
            "oov_tokens": oov_tokens,
            "has_placeholders": has_placeholders,
            "num_structured_sentences": len(structured),
            "structured_json": [s.model_dump() for s in structured],
            "forward_prompt_tokens": forward_resp.prompt_tokens,
            "forward_completion_tokens": forward_resp.completion_tokens,
            "bw_prompt_tokens": bw_pt,
            "bw_completion_tokens": bw_ct,
            "cmp_prompt_tokens": cmp_pt,
            "cmp_completion_tokens": cmp_ct,
            "t_forward": t_forward,
            "t_backwards": t_bw,
            "t_comparator": t_cmp,
            "wall_time": time.time() - t0,
            "error": None,
        }
    except Exception as e:
        return {
            "source": row["sentence"],
            "type": row["type"],
            "target": None,
            "target_masked": None,
            "backwards": None,
            "comparator": None,
            "has_placeholders": None,
            "wall_time": time.time() - t0,
            "error": f"{type(e).__name__}: {e}",
        }


def tag_for(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forward-model", required=True, help="Ollama tag or 'gpt-*' for OpenAI")
    p.add_argument("--backward-model", default="gpt-4o-mini")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--types", default=None,
                   help="Comma-separated sentence types to include (e.g. "
                        "'nominalization,complex'). Existing rows of these types "
                        "are removed from the output file so they'll be re-run. "
                        "Rows of other types are preserved.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else RESULTS / f"{tag_for(args.forward_model)}__{tag_for(args.backward_model)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    types_filter: set[str] | None = None
    if args.types:
        types_filter = {t.strip() for t in args.types.split(",") if t.strip()}
        # purge matching-type rows from existing output so they'll be re-run
        if out_path.exists():
            kept: list[str] = []
            dropped = 0
            with out_path.open() as f:
                for i, line in enumerate(f, 1):
                    try:
                        r = json.loads(line)
                        if r.get("type") in types_filter:
                            dropped += 1
                            continue
                    except Exception as e:
                        # Preserve the line as-is if it fails to parse, so we
                        # don't silently lose data, but flag it to the user.
                        print(f"warning: keeping unparseable line {i} "
                              f"(will not be dropped): {type(e).__name__}: {e}",
                              file=sys.stderr)
                    kept.append(line)
            with out_path.open("w") as f:
                f.writelines(kept)
            print(f"--types {sorted(types_filter)}: dropped {dropped} prior rows from {out_path.name}",
                  file=sys.stderr)

    rows = load_dataset()
    if types_filter:
        rows = [r for r in rows if r["type"] in types_filter]
    if args.limit:
        rows = rows[: args.limit]
    done = load_done(out_path)
    todo = [r for r in rows if r["sentence"] not in done]

    print(f"forward={args.forward_model}  backward={args.backward_model}  parallel={args.parallel}",
          file=sys.stderr)
    print(f"total={len(rows)} done={len(done)} todo={len(todo)} out={out_path}", file=sys.stderr)
    if not todo:
        return 0

    forward: Agent = make_agent(args.forward_model, temperature=0.0,
                                ollama_url=args.ollama_url)
    backward: Agent = make_agent(args.backward_model, temperature=0.0,
                               ollama_url=args.ollama_url)

    t_start = time.time()
    completed = 0
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(translate_one, forward, backward, r): r for r in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
            status = "OK" if rec["error"] is None else "ERR"
            print(
                f"[{completed}/{len(todo)}] {status} {rec['type']:>20s} {elapsed:6.1f}s elapsed, "
                f"{rate:4.2f}/s, ETA {eta:6.0f}s :: {rec['source'][:60]}",
                file=sys.stderr,
            )

    print(f"done in {time.time() - t_start:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
