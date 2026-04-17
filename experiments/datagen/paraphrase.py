"""Step 2: for each structured sentence, render canonical English and request K
paraphrases via a transformation menu.

Output (one record per structure):
    {
        "id": <int>, "type": "sv"|"svo", "tags": [...],
        "structured": <pydantic dump>,
        "canonical": <natural English from SentenceToEnglishTool>,
        "paraphrases": [
            {"text": <english>, "transforms": ["add_adverbial", ...]},
            ...
        ],
        "errors": [...]
    }

Resumable: skips structures whose id already appears in the output JSONL.
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
from pydantic import BaseModel, Field

from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, jsonl_append, jsonl_iter, make_openai  # noqa: E402

load_dotenv()


TRANSFORMS = {
    "add_adverbial": "Add an adverbial phrase (e.g. 'quickly', 'every morning', 'at noon').",
    "add_relative_clause": "Add a non-restrictive relative clause to a noun.",
    "passivize": "Re-cast the sentence in the passive voice (only meaningful for transitive sentences).",
    "coordinate_and_drop": "Add a coordinate clause; the structured sentence still expresses the dominant clause only.",
    "substitute_rare_vocab": "Swap nouns/verbs for rarer English equivalents (e.g. 'dog' -> 'chihuahua', 'eat' -> 'devour').",
    "add_irrelevant_detail": "Add concrete but inessential detail (location, time, instrument).",
    "idiomize": "Re-express via an English idiom whose literal meaning matches the structure.",
    "change_register": "Shift register (formal / colloquial / literary).",
}


class Paraphrase(BaseModel):
    text: str = Field(..., description="An English paraphrase whose meaning matches the input.")
    transforms: list[str] = Field(
        default_factory=list,
        description="The transform names from the menu that were applied.",
    )


class ParaphraseList(BaseModel):
    paraphrases: list[Paraphrase]


def _prompt_sys(n_sentences: int) -> str:
    """coordinate_and_drop only makes sense for single-clause inputs. For
    multi-clause inputs we must preserve ALL clauses."""
    if n_sentences == 1:
        transforms = TRANSFORMS
        clause_note = (
            "The structured target schema is fixed; your paraphrases must preserve "
            "the dominant clause's subject, verb, and object so that the same "
            "structured target still applies."
        )
    else:
        transforms = {k: v for k, v in TRANSFORMS.items() if k != "coordinate_and_drop"}
        clause_note = (
            f"The source comprises {n_sentences} distinct clauses. Your paraphrases "
            "MUST preserve all clauses — the subject/verb/object of each — so the "
            "multi-clause structured target still applies. You may coordinate them "
            "with conjunctions, semicolons, or separate sentences."
        )
    return (
        "You generate English paraphrases of a given English sentence for use as training "
        "data for a low-resource translation system.\n"
        + clause_note
        + "\n\nApply ONE OR TWO transforms from this menu per paraphrase, listing the names "
        "you applied in `transforms`:\n"
        + "\n".join(f"- {k}: {v}" for k, v in transforms.items())
        + "\n\n"
        "Return between 4 and 8 distinct paraphrases that span a range of complexity, "
        "from a near-literal rewording to one with multiple added details."
    )


def _parse_structure(d: dict[str, Any]):
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def render_canonical(s2e: SentenceToEnglishTool, structured: Any) -> str:
    r = s2e(structured)
    return r.content.strip()


def request_paraphrases(agent, canonical: str, n_sentences: int, k_min: int, k_max: int) -> ParaphraseList:
    user = (
        f"Source sentence: {canonical}\n\n"
        f"Produce {k_min}\u2013{k_max} English paraphrases."
    )
    resp = agent.get_response(
        messages=[
            {"role": "system", "content": _prompt_sys(n_sentences)},
            {"role": "user", "content": user},
        ],
        response_format=ParaphraseList,
    )
    return resp.content


def process_one(
    record: dict[str, Any],
    sentence_types: tuple,
    strong_model: str,
    para_model: str,
    k_min: int,
    k_max: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": record["id"],
        "n_sentences": record["n_sentences"],
        "types": record["types"],
        "tags": record["tags"],
        "structured": record["structured"],
        "canonical": None,
        "per_structure_canonicals": None,
        "paraphrases": [],
        "errors": [],
    }
    try:
        parsed = [_parse_structure(d) for d in record["structured"]]

        strong = make_openai(strong_model, temperature=0.0)
        s2e = SentenceToEnglishTool(agent=strong, SentenceType=sentence_types)
        per_struct = [render_canonical(s2e, p) for p in parsed]
        out["per_structure_canonicals"] = per_struct
        # Joined canonical: simple concatenation, trusting gpt's per-clause punctuation.
        canonical = " ".join(per_struct)
        out["canonical"] = canonical

        para_agent = make_openai(para_model, temperature=0.9)
        result = request_paraphrases(para_agent, canonical, record["n_sentences"], k_min, k_max)
        out["paraphrases"] = [p.model_dump() for p in result.paraphrases]
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(OUT / "structures.jsonl"))
    p.add_argument("--output", default=str(OUT / "paraphrases.jsonl"))
    p.add_argument("--strong-model", default="gpt-4o-mini",
                   help="Model used to render canonical English from structured JSON.")
    p.add_argument("--para-model", default="gpt-4o-mini",
                   help="Model used to author paraphrases. Use gpt-4o for higher quality.")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"missing input: {in_path}", file=sys.stderr)
        return 2

    language = LanguageLoader.load_language("ovp")
    sentence_types = language.sentence_types

    records = list(jsonl_iter(in_path))
    if args.limit:
        records = records[: args.limit]

    done_ids = {r["id"] for r in jsonl_iter(out_path) if r.get("canonical") is not None}
    todo = [r for r in records if r["id"] not in done_ids]
    print(
        f"input={len(records)} done={len(done_ids)} todo={len(todo)} "
        f"out={out_path}",
        file=sys.stderr,
    )
    if not todo:
        return 0

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(
                process_one, r, sentence_types,
                args.strong_model, args.para_model, args.k_min, args.k_max,
            ): r for r in todo
        }
        for fut in as_completed(futures):
            rec = fut.result()
            jsonl_append(out_path, rec)
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
            status = "OK" if not rec["errors"] else "ERR"
            n_para = len(rec["paraphrases"])
            print(
                f"[{completed}/{len(todo)}] {status} id={rec['id']:5d} "
                f"k={n_para} {elapsed:6.1f}s, {rate:4.2f}/s, ETA {eta:6.0f}s :: "
                f"{(rec.get('canonical') or '')[:60]}",
                file=sys.stderr,
            )
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
