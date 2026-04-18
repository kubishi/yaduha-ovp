"""Step 5: validate, dedup, and split datagen outputs into SFT-ready files.

Reads:
    out/paraphrases.jsonl
    out/oov_substitutions.jsonl
    out/decoder_pairs.jsonl

Writes (chat-template SFT format):
    out/forward_train.jsonl   out/forward_val.jsonl
    out/backward_train.jsonl  out/backward_val.jsonl

Each output line:
    {
        "messages": [
            {"role": "system",    "content": <system prompt>},
            {"role": "user",      "content": <english | structured-json>},
            {"role": "assistant", "content": <SentenceList JSON | english>}
        ],
        "meta": {"source": "...", "structure_id": ..., "tags": [...], ...}
    }

Validation (optional, --comet-threshold): rejects forward-direction
paraphrases whose COMET similarity to the canonical English is below
threshold (per CLAUDE.md plan: 0.5).

Dedup: per-direction, on normalized English input.
Split: stratified 90/10 on (direction, type, has_oov) bucket.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from yaduha.tool.english_to_sentences import SentenceList
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence
from yaduha_ovp.prompts import get_prompt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, jsonl_iter, jsonl_write, normalize_english  # noqa: E402


def parse_structured(d: dict[str, Any]):
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)

FORWARD_SYSTEM = get_prompt(
    include_vocab=True,
    include_examples=(SubjectVerbSentence, SubjectVerbObjectSentence),
)

# BACKWARD_SYSTEM must match SentenceToEnglishTool's inference system prompt
# EXACTLY so training and serving see the same context. See
# yaduha.tool.sentence_to_english for the source.
BACKWARD_SYSTEM = (
    "You are a translator that transforms structured sentences into natural English. "
    "The sentences may be strange and unusual, but you must translate them as "
    "accurately as possible. "
)


def _backward_example_turns() -> list[dict[str, str]]:
    """Replicate SentenceToEnglishTool's in-context example turns so training
    records match the inference format exactly.

    Format per example (matches sentence_to_english.py):
        user      -> json.dumps(example_sentence.model_dump_json())
        assistant -> english
    """
    turns: list[dict[str, str]] = []
    for SentenceCls in (SubjectVerbSentence, SubjectVerbObjectSentence):
        for english, example in SentenceCls.get_examples():
            turns.append({
                "role": "user",
                "content": json.dumps(example.model_dump_json(), ensure_ascii=False),
            })
            turns.append({"role": "assistant", "content": english})
    return turns


_BACKWARD_EXAMPLES = _backward_example_turns()


def wrap_forward(structured_list: list[dict[str, Any]]) -> str:
    """Emit the assistant target using pydantic's SentenceList serialization.
    `structured_list` is always a list (length 1+); multi-clause records emit
    a SentenceList of the same length."""
    parsed = [parse_structured(d) for d in structured_list]
    return SentenceList(sentences=parsed).model_dump_json()


def build_forward(
    paraphrases_path: Path,
    oov_path: Path,
    proper_nouns_path: Path | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for r in jsonl_iter(paraphrases_path):
        if r.get("errors") or not r.get("canonical"):
            continue
        target = wrap_forward(r["structured"])
        # Use 'types' (list) joined for bucket keys; fall back to legacy 'type'.
        type_key = "+".join(r.get("types") or [r.get("type", "?")])
        meta_base = {
            "source": "paraphrase",
            "structure_id": r["id"],
            "n_sentences": r.get("n_sentences", 1),
            "type": type_key,
            "types": r.get("types", [r.get("type")]),
            "tags": r["tags"],
        }
        out.append({
            "english": r["canonical"],
            "target": target,
            "meta": {**meta_base, "kind": "canonical", "transforms": []},
        })
        for p in r.get("paraphrases", []):
            out.append({
                "english": p["text"],
                "target": target,
                "meta": {
                    **meta_base,
                    "kind": "paraphrase",
                    "transforms": p.get("transforms", []),
                },
            })

    for r in jsonl_iter(oov_path):
        if r.get("errors") or not r.get("english"):
            continue
        target = wrap_forward(r["structured"])
        out.append({
            "english": r["english"],
            "target": target,
            "meta": {
                "source": "oov_substitution",
                "kind": r["kind"],
                "oov_lemma": r["oov_lemma"],
                "in_vocab": r.get("in_vocab"),
                "type": r["type"],
                "n_sentences": 1,
                "tags": [r["kind"]],
            },
        })

    if proper_nouns_path is not None and proper_nouns_path.exists():
        for r in jsonl_iter(proper_nouns_path):
            if r.get("errors") or not r.get("english"):
                continue
            target = wrap_forward(r["structured"])
            type_key = "+".join(r.get("types") or ["?"])
            out.append({
                "english": r["english"],
                "target": target,
                "meta": {
                    "source": "proper_nouns",
                    "kind": r["kind"],
                    "name": r["name"],
                    "type": type_key,
                    "types": r.get("types", []),
                    "n_sentences": r.get("n_sentences", 1),
                    "tags": [r["kind"]],
                },
            })

    return out


def build_backward(decoder_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in jsonl_iter(decoder_path):
        if r.get("errors") or not r.get("english"):
            continue
        out.append({
            "structured": r["structured"],
            "english": r["english"],
            "meta": {
                "source": "decoder",
                "structure_id": r["structure_id"],
                "kind": r["kind"],
                "type": r["type"],
                "tags": r["tags"],
                "oov_tokens": r.get("oov_tokens", []),
            },
        })
    return out


def comet_filter(
    records: list[dict[str, Any]],
    threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Reject paraphrases far from canonical. Keeps canonicals + non-paraphrase sources."""
    by_struct: dict[int, str] = {}
    pending: list[dict[str, Any]] = []
    keep: list[dict[str, Any]] = []
    for r in records:
        meta = r["meta"]
        if meta.get("kind") == "canonical":
            by_struct[meta["structure_id"]] = r["english"]
            keep.append(r)
        elif meta.get("kind") == "paraphrase":
            pending.append(r)
        else:
            keep.append(r)

    if not pending:
        return keep

    print(
        f"COMET-filtering {len(pending)} paraphrases (threshold {threshold})...",
        file=sys.stderr,
    )
    from comet import download_model, load_from_checkpoint  # type: ignore[import-untyped]

    path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(path)
    data = []
    for r in pending:
        canon = by_struct.get(r["meta"]["structure_id"])
        if canon is None:
            continue
        # src=ref=canonical, mt=paraphrase  (mirrors run_metrics.py monolingual trick)
        data.append({"src": canon, "mt": r["english"], "ref": canon})
    pred = model.predict(data, batch_size=batch_size, gpus=1)

    n_drop = 0
    for r, score in zip(pending, pred.scores):
        r["meta"]["comet"] = float(score)
        if score >= threshold:
            keep.append(r)
        else:
            n_drop += 1
    print(f"COMET dropped {n_drop}/{len(pending)} paraphrases", file=sys.stderr)
    return keep


def dedup(records: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in records:
        k = key_fn(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def stratified_split(
    records: list[dict[str, Any]],
    bucket_fn,
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[bucket_fn(r)].append(r)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for bucket, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac)) if items else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def to_messages_forward(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": FORWARD_SYSTEM},
            {"role": "user", "content": r["english"]},
            {"role": "assistant", "content": r["target"]},
        ],
        "meta": r["meta"],
    }


def to_messages_backward(r: dict[str, Any]) -> dict[str, Any]:
    # Match SentenceToEnglishTool's inference format EXACTLY:
    #   system  -> SentenceToEnglishTool's system prompt
    #   *       -> in-context example turns from each Sentence's get_examples()
    #   user    -> json.dumps(sentence.model_dump_json())   (double-encoded)
    #   assist. -> english
    # Prior versions trained with a short custom system prompt and no
    # in-context examples, which mismatched the serving path and caused the
    # adapter to run past the intended turn boundary at inference.
    parsed = parse_structured(r["structured"])
    user = json.dumps(parsed.model_dump_json(), ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": BACKWARD_SYSTEM},
            *_BACKWARD_EXAMPLES,
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["english"]},
        ],
        "meta": r["meta"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paraphrases", default=str(OUT / "paraphrases.jsonl"))
    p.add_argument("--oov", default=str(OUT / "oov_substitutions.jsonl"))
    p.add_argument("--proper-nouns", default=str(OUT / "proper_nouns.jsonl"),
                   help="Optional; skipped if file absent.")
    p.add_argument("--decoder", default=str(OUT / "decoder_pairs.jsonl"))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--comet-threshold", type=float, default=None,
                   help="If set, drop paraphrases whose COMET sim to canonical < threshold (e.g. 0.5).")
    p.add_argument("--comet-batch", type=int, default=32)
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    # --- Forward ---
    forward = build_forward(
        Path(args.paraphrases),
        Path(args.oov),
        Path(args.proper_nouns),
    )
    print(f"forward: built {len(forward)} raw pairs", file=sys.stderr)

    if args.comet_threshold is not None:
        forward = comet_filter(forward, args.comet_threshold, args.comet_batch)

    forward = dedup(
        forward,
        key_fn=lambda r: ("forward", normalize_english(r["english"])),
    )
    print(f"forward: {len(forward)} after dedup", file=sys.stderr)

    f_train, f_val = stratified_split(
        forward,
        bucket_fn=lambda r: (
            r["meta"].get("type", "?"),
            r["meta"].get("source", "?"),
            r["meta"].get("kind", "?"),
        ),
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # --- Backward ---
    backward = build_backward(Path(args.decoder))
    print(f"backward: built {len(backward)} raw pairs", file=sys.stderr)
    backward = dedup(
        backward,
        key_fn=lambda r: (
            "backward",
            r["meta"]["kind"],
            json.dumps(r["structured"], sort_keys=True),
        ),
    )
    print(f"backward: {len(backward)} after dedup", file=sys.stderr)

    b_train, b_val = stratified_split(
        backward,
        bucket_fn=lambda r: (
            r["meta"].get("type", "?"),
            r["meta"].get("kind", "?"),
        ),
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # --- Write ---
    n_ft = jsonl_write(out_dir / "forward_train.jsonl", map(to_messages_forward, f_train))
    n_fv = jsonl_write(out_dir / "forward_val.jsonl",   map(to_messages_forward, f_val))
    n_bt = jsonl_write(out_dir / "backward_train.jsonl", map(to_messages_backward, b_train))
    n_bv = jsonl_write(out_dir / "backward_val.jsonl",   map(to_messages_backward, b_val))

    print(
        f"wrote forward_train={n_ft} forward_val={n_fv} "
        f"backward_train={n_bt} backward_val={n_bv}",
        file=sys.stderr,
    )

    # --- Bucket summary ---
    def show(label: str, recs: list[dict[str, Any]], key) -> None:
        c: dict[Any, int] = defaultdict(int)
        for r in recs:
            c[key(r)] += 1
        print(f"\n{label} buckets:", file=sys.stderr)
        for k, v in sorted(c.items(), key=lambda x: -x[1])[:20]:
            print(f"  {str(k):60s} {v:5d}", file=sys.stderr)

    show("forward train",
         [r for r in forward if r in f_train],
         lambda r: (r["meta"].get("type"), r["meta"].get("source"), r["meta"].get("kind")))
    show("backward train",
         [r for r in backward if r in b_train],
         lambda r: (r["meta"].get("type"), r["meta"].get("kind")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
