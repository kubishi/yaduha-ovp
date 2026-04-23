"""Re-run only the comparator stage on existing translation JSONLs.

The comparator stage feeds ``mask_oov(structured)`` to the backward decoder.
When the backward decoder hallucinates content over ``[NOUN]``/``[VERB]``
sentinels it inflates the comparator score and erases the placeholder-leakage
signal. After tightening the SentenceToEnglishTool system prompt to forbid
this substitution, we re-run the comparator stage in place. The forward and
``backwards`` stages are unaffected (the forward stage is the model under
test; the backwards stage feeds unmasked JSON), so we leave them alone.

Resumable: rows whose ``cmp_prompt_version`` already matches ``--version``
are skipped. Pass ``--force`` to re-run regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yaduha.agent import Agent
from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
from _common import clean, make_agent, parse_structured  # noqa: E402

load_dotenv()

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Bump when the comparator prompt or pipeline changes so prior rows are re-run.
PROMPT_VERSION = "v2-preserve-placeholders"


def rerun_one(backward: Agent, sentence_types: tuple, row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("has_placeholders"):
        return row
    if row.get("error"):
        return row
    structured_dicts = row.get("structured_json")
    if not structured_dicts:
        return row

    structured = [parse_structured(d) for d in structured_dicts]
    s2e = SentenceToEnglishTool(agent=backward, SentenceType=sentence_types)

    cmp_parts: list[str] = []
    cmp_pt = cmp_ct = 0
    oov_tokens: list[str] = []
    t0 = time.time()
    for s in structured:
        masked, oov = s.masked_copy()
        oov_tokens.extend(oov)
        r = s2e(masked)
        cmp_parts.append(clean(r.content))
        cmp_pt += r.prompt_tokens
        cmp_ct += r.completion_tokens
    t_cmp = time.time() - t0

    row = dict(row)
    row["comparator"] = " ".join(cmp_parts)
    row["oov_tokens"] = oov_tokens
    row["cmp_prompt_tokens"] = cmp_pt
    row["cmp_completion_tokens"] = cmp_ct
    row["t_comparator"] = t_cmp
    row["cmp_prompt_version"] = PROMPT_VERSION
    # Stale comparator metric scores must be wiped so run_metrics re-scores.
    for k in list(row.keys()):
        if k.endswith("_comparator"):
            row.pop(k, None)
    return row


def process_file(
    path: Path,
    backward_model: str,
    parallel: int,
    force: bool,
    ollama_url: str,
) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    todo_idx = []
    for i, r in enumerate(rows):
        if not r.get("has_placeholders"):
            continue
        if r.get("error"):
            continue
        if not force and r.get("cmp_prompt_version") == PROMPT_VERSION:
            continue
        todo_idx.append(i)

    if not todo_idx:
        print(f"[{path.name}] up-to-date, nothing to rerun", file=sys.stderr)
        return 0, 0

    language = LanguageLoader.load_language("ovp")
    sentence_types = language.sentence_types
    backward: Agent = make_agent(backward_model, temperature=0.0, ollama_url=ollama_url)

    print(f"[{path.name}] re-running comparator on {len(todo_idx)}/{len(rows)} rows",
          file=sys.stderr)
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(rerun_one, backward, sentence_types, rows[i]): i for i in todo_idx}
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i] = fut.result()
            completed += 1
            if completed % 10 == 0 or completed == len(todo_idx):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(todo_idx) - completed) / rate if rate > 0 else float("inf")
                print(f"  [{completed}/{len(todo_idx)}] {elapsed:5.1f}s, "
                      f"{rate:4.2f}/s, ETA {eta:5.0f}s",
                      file=sys.stderr)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return len(todo_idx), len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more *.jsonl files to update in place")
    p.add_argument("--backward-model", default="gpt-4o-mini",
                   help="Backward decoder; should match what produced the file "
                        "originally (the second tag in the filename)")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="Re-run even rows whose cmp_prompt_version is current")
    args = p.parse_args()

    total_rerun = 0
    for s in args.inputs:
        path = Path(s)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        rerun, _total = process_file(
            path, args.backward_model, args.parallel, args.force, args.ollama_url,
        )
        total_rerun += rerun
    print(f"done. {total_rerun} rows updated across {len(args.inputs)} files",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
