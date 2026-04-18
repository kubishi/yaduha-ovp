"""Build a hand-validation seed for the backward-translation eval.

Samples a diverse set of structured sentences (stratified across SV/SVO,
in-vocab/OOV, plain/nominalized, and with/without masking), renders each via
gpt-4o-mini's SentenceToEnglishTool as a seed translation, and writes a CSV
for the human annotator to edit into the gold reference.

Output: eval_items.csv with columns:
    id, kind, tags, structured_json, ovp_surface, seed_english, gold_english, notes

The annotator edits `gold_english` (replacing or keeping the seed as they see
fit) and saves to `eval_items.gold.csv`. `run_eval.py` then scores each
backward model against that gold.

Cost: ~$0.02 at gpt-4o-mini for ~40 renderings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yaduha.agent.openai import OpenAIAgent
from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha_ovp import (
    INTRANSITIVE_VERB_LOOKUP,
    NOUN_LOOKUP,
    TRANSITIVE_VERB_LOOKUP,
    IntransitiveVerb,
    NominalizerTense,
    NominalObject,
    NominalSubject,
    ObjectNoun,
    Plurality,
    Pronoun,
    Proximity,
    SubjectNoun,
    SubjectVerbObjectSentence,
    SubjectVerbSentence,
    TenseAspect,
    TransitiveVerb,
)

load_dotenv()

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

NOUN_KEYS = list(NOUN_LOOKUP.keys())
TRANS_KEYS = list(TRANSITIVE_VERB_LOOKUP.keys())
INTRANS_KEYS = list(INTRANSITIVE_VERB_LOOKUP.keys())
PRONS = [p for p in Pronoun if p != Pronoun.reflexive]
OOV_NOUNS = ["laptop", "chihuahua", "cabin", "mountain_peak", "Tom", "Susan"]
OOV_VERBS = ["program", "sprint", "compose", "hike"]
TENSES = list(TenseAspect)


def _rand_subject(rng: random.Random, *, allow_name=False, allow_oov=False):
    r = rng.random()
    if r < 0.35:
        return rng.choice(PRONS)
    if allow_oov and r < 0.45:
        return SubjectNoun(head=rng.choice(OOV_NOUNS), proximity=rng.choice(list(Proximity)),
                           plurality=Plurality.singular)
    return SubjectNoun(head=rng.choice(NOUN_KEYS), proximity=rng.choice(list(Proximity)),
                       plurality=rng.choice(list(Plurality)))


def _rand_object(rng: random.Random, *, allow_oov=False):
    r = rng.random()
    if r < 0.3:
        return rng.choice(list(Pronoun))
    if allow_oov and r < 0.4:
        return ObjectNoun(head=rng.choice(OOV_NOUNS), proximity=rng.choice(list(Proximity)),
                          plurality=Plurality.singular)
    return ObjectNoun(head=rng.choice(NOUN_KEYS), proximity=rng.choice(list(Proximity)),
                      plurality=rng.choice(list(Plurality)))


def sample_items(seed: int) -> list[dict[str, Any]]:
    """40 items stratified across categories. Deterministic given seed."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    next_id = [0]

    def add(kind: str, tags: list[str], sentence: Any):
        items.append({
            "id": next_id[0],
            "kind": kind,
            "tags": tags,
            "sentence": sentence,
        })
        next_id[0] += 1

    # 8 simple SV (all in-vocab)
    for _ in range(8):
        add("sv_simple", ["sv"],
            SubjectVerbSentence(
                subject=_rand_subject(rng),
                verb=IntransitiveVerb(lemma=rng.choice(INTRANS_KEYS), tense_aspect=rng.choice(TENSES)),
            ))

    # 8 simple SVO (all in-vocab)
    for _ in range(8):
        add("svo_simple", ["svo"],
            SubjectVerbObjectSentence(
                subject=_rand_subject(rng),
                verb=TransitiveVerb(lemma=rng.choice(TRANS_KEYS), tense_aspect=rng.choice(TENSES)),
                object=_rand_object(rng),
            ))

    # 6 with OOV noun head (subject or object)
    for _ in range(6):
        if rng.random() < 0.5:
            # OOV in subject
            s = SubjectVerbSentence(
                subject=SubjectNoun(head=rng.choice(OOV_NOUNS), proximity=rng.choice(list(Proximity)),
                                    plurality=Plurality.singular),
                verb=IntransitiveVerb(lemma=rng.choice(INTRANS_KEYS), tense_aspect=rng.choice(TENSES)),
            )
            add("oov_noun_subject", ["sv", "oov_noun"], s)
        else:
            s = SubjectVerbObjectSentence(
                subject=_rand_subject(rng),
                verb=TransitiveVerb(lemma=rng.choice(TRANS_KEYS), tense_aspect=rng.choice(TENSES)),
                object=ObjectNoun(head=rng.choice(OOV_NOUNS), proximity=rng.choice(list(Proximity)),
                                  plurality=Plurality.singular),
            )
            add("oov_noun_object", ["svo", "oov_noun"], s)

    # 4 with OOV verb lemma
    for _ in range(4):
        if rng.random() < 0.5:
            s = SubjectVerbSentence(
                subject=_rand_subject(rng),
                verb=IntransitiveVerb(lemma=rng.choice(OOV_VERBS), tense_aspect=rng.choice(TENSES)),
            )
            add("oov_verb", ["sv", "oov_verb"], s)
        else:
            s = SubjectVerbObjectSentence(
                subject=_rand_subject(rng),
                verb=TransitiveVerb(lemma=rng.choice(OOV_VERBS), tense_aspect=rng.choice(TENSES)),
                object=_rand_object(rng),
            )
            add("oov_verb", ["svo", "oov_verb"], s)

    # 6 nominalization (NominalSubject and NominalObject)
    for _ in range(6):
        if rng.random() < 0.5:
            s = SubjectVerbSentence(
                subject=NominalSubject(
                    verb_lemma=rng.choice(INTRANS_KEYS + TRANS_KEYS),
                    nominalizer_tense=rng.choice(list(NominalizerTense)),
                    proximity=rng.choice(list(Proximity)),
                    plurality=Plurality.singular,
                ),
                verb=IntransitiveVerb(lemma=rng.choice(INTRANS_KEYS), tense_aspect=rng.choice(TENSES)),
            )
            add("nominalized_subject", ["sv", "nominalized"], s)
        else:
            s = SubjectVerbObjectSentence(
                subject=_rand_subject(rng),
                verb=TransitiveVerb(lemma=rng.choice(TRANS_KEYS), tense_aspect=rng.choice(TENSES)),
                object=NominalObject(
                    verb_lemma=rng.choice(INTRANS_KEYS + TRANS_KEYS),
                    nominalizer_tense=rng.choice(list(NominalizerTense)),
                    proximity=rng.choice(list(Proximity)),
                    plurality=Plurality.singular,
                ),
            )
            add("nominalized_object", ["svo", "nominalized"], s)

    # 8 masked variants: pick 8 structures that have OOV content and mask them
    oov_candidates = [it for it in items
                      if "oov_noun" in it["tags"] or "oov_verb" in it["tags"]]
    rng.shuffle(oov_candidates)
    for parent in oov_candidates[:8]:
        masked, oov_tokens = parent["sentence"].masked_copy()
        add("masked", parent["tags"] + ["masked"], masked)
        items[-1]["source_id"] = parent["id"]
        items[-1]["masked_tokens"] = oov_tokens

    return items


def render_seed(items: list[dict[str, Any]], sentence_types: tuple) -> None:
    """Fill each item with a `seed_english` via gpt-4o-mini + `ovp_surface` via str()."""
    agent = OpenAIAgent(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"],
                        temperature=0.0)
    s2e = SentenceToEnglishTool(agent=agent, SentenceType=sentence_types)
    for it in items:
        try:
            it["ovp_surface"] = str(it["sentence"])
        except Exception:
            it["ovp_surface"] = "(render error)"
        try:
            r = s2e(it["sentence"])
            it["seed_english"] = r.content.strip()
        except Exception as e:
            it["seed_english"] = f"(ERROR: {type(e).__name__}: {e})"


def write_csv(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort so masked items follow their parents, to help the annotator.
    by_id = {it["id"]: it for it in items}
    ordered: list[dict[str, Any]] = []
    masked_children: dict[int, list[dict[str, Any]]] = {}
    for it in items:
        if it["kind"] == "masked":
            masked_children.setdefault(it["source_id"], []).append(it)
    for it in items:
        if it["kind"] == "masked":
            continue
        ordered.append(it)
        for child in masked_children.get(it["id"], []):
            ordered.append(child)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "tags", "source_id", "masked_tokens",
                    "structured_json", "ovp_surface", "parent_english",
                    "seed_english", "gold_english", "notes"])
        for it in ordered:
            parent_eng = ""
            if it["kind"] == "masked":
                parent = by_id.get(it["source_id"])
                if parent:
                    parent_eng = parent.get("seed_english", "")
            w.writerow([
                it["id"],
                it["kind"],
                ",".join(it["tags"]),
                it.get("source_id", ""),
                ",".join(it.get("masked_tokens", [])),
                it["sentence"].model_dump_json(),
                it.get("ovp_surface", ""),
                parent_eng,
                it.get("seed_english", ""),
                # Start gold = seed; annotator overwrites if needed.
                it.get("seed_english", ""),
                "",
            ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT / "eval_items.csv"))
    p.add_argument("--no-seed-english", action="store_true",
                   help="Skip the gpt-4o-mini rendering (faster; gold_english column blank).")
    args = p.parse_args()

    language = LanguageLoader.load_language("ovp")

    items = sample_items(args.seed)
    print(f"sampled {len(items)} items", file=sys.stderr)
    # Composition summary
    kinds: dict[str, int] = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    for k, v in sorted(kinds.items()):
        print(f"  {k:<24s} {v:3d}", file=sys.stderr)

    if not args.no_seed_english:
        print("rendering seed English via gpt-4o-mini...", file=sys.stderr)
        render_seed(items, language.sentence_types)

    out = Path(args.out)
    write_csv(items, out)
    print(f"wrote {out}", file=sys.stderr)
    print(f"\nNext: open {out} in a spreadsheet tool, review/edit the "
          f"'gold_english' column row by row, save as '{out.with_suffix('.gold.csv')}'.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
