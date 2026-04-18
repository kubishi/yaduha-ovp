"""Step 1: enumerate diverse valid structured sentences.

Stratifies across single vs multi-clause, SV/SVO, plain noun vs nominalized
subject/object, and deliberately injects OOV nouns/verbs (drawn from
oov_lemmas) so the model later learns when to emit a placeholder vs a
hypernym.

Each output line is one training *record*. A record may contain one or
multiple structured sentences — multi-clause records teach the model to
emit SentenceList targets of length > 1 (needed for two-clause / complex
eval sentences):

    {
        "id":        <int>,
        "n_sentences": 1 | 2 | 3,
        "types":     ["sv"] | ["sv","svo"] | ...,
        "tags":      ["nominalized_subject", "oov_noun", ...],
        "structured": [<sentence dict>, ...],   # ALWAYS a list
    }

The output is the seed input for paraphrase.py and decoder_pairs.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, jsonl_write  # noqa: E402
from oov_lemmas import (  # noqa: E402
    NEGATIVE_INTRANS_VERBS_F,
    NEGATIVE_NOUNS_F,
    NEGATIVE_TRANS_VERBS_F,
    POSITIVE_INTRANS_VERBS,
    POSITIVE_NOUNS,
    POSITIVE_TRANS_VERBS,
)

NOUN_KEYS = list(NOUN_LOOKUP.keys())
TRANS_KEYS = list(TRANSITIVE_VERB_LOOKUP.keys())
INTRANS_KEYS = list(INTRANSITIVE_VERB_LOOKUP.keys())
PRONOUNS = list(Pronoun)
SUBJECT_PRONOUN_VALUES = [p for p in PRONOUNS if p != Pronoun.reflexive]
OBJECT_PRONOUN_VALUES = list(PRONOUNS)
TENSES = list(TenseAspect)
PROX = list(Proximity)
PLUR = list(Plurality)
NOM_TENSES = list(NominalizerTense)

OOV_NOUN_POOL = list(POSITIVE_NOUNS.keys()) + list(NEGATIVE_NOUNS_F)
OOV_TRANS_POOL = list(POSITIVE_TRANS_VERBS.keys()) + list(NEGATIVE_TRANS_VERBS_F)
OOV_INTRANS_POOL = list(POSITIVE_INTRANS_VERBS.keys()) + list(NEGATIVE_INTRANS_VERBS_F)


def _pick_noun(rng: random.Random, oov: bool, role: str) -> tuple[Any, list[str]]:
    if oov and OOV_NOUN_POOL:
        head = rng.choice(OOV_NOUN_POOL)
        tag = ["oov_noun"]
    else:
        head = rng.choice(NOUN_KEYS)
        tag = []
    cls = SubjectNoun if role == "subject" else ObjectNoun
    return (
        cls(head=head, proximity=rng.choice(PROX), plurality=rng.choice(PLUR)),
        tag,
    )


def _pick_nominal(rng: random.Random, oov: bool, role: str) -> tuple[Any, list[str]]:
    pool = TRANS_KEYS + INTRANS_KEYS
    if oov and (POSITIVE_TRANS_VERBS or POSITIVE_INTRANS_VERBS):
        verb_lemma = rng.choice(
            list(POSITIVE_TRANS_VERBS.keys()) + list(POSITIVE_INTRANS_VERBS.keys())
        )
        tag = ["oov_nominalized_verb"]
    else:
        verb_lemma = rng.choice(pool)
        tag = []
    cls = NominalSubject if role == "subject" else NominalObject
    return (
        cls(
            verb_lemma=verb_lemma,
            nominalizer_tense=rng.choice(NOM_TENSES),
            proximity=rng.choice(PROX),
            plurality=rng.choice(PLUR),
        ),
        tag + [f"nominalized_{role}"],
    )


def _pick_subject(
    rng: random.Random, allow_nominal: bool, oov: bool
) -> tuple[Any, list[str]]:
    """Returns (subject, tags). Subject can be Pronoun, SubjectNoun, or NominalSubject."""
    r = rng.random()
    if allow_nominal and r < 0.25:
        return _pick_nominal(rng, oov, "subject")
    if r < 0.55:
        return rng.choice(SUBJECT_PRONOUN_VALUES), ["pronoun_subject"]
    return _pick_noun(rng, oov, "subject")


def _pick_object(
    rng: random.Random, allow_nominal: bool, oov: bool
) -> tuple[Any, list[str]]:
    r = rng.random()
    if allow_nominal and r < 0.25:
        return _pick_nominal(rng, oov, "object")
    if r < 0.55:
        return rng.choice(OBJECT_PRONOUN_VALUES), ["pronoun_object"]
    return _pick_noun(rng, oov, "object")


def _pick_verb(
    rng: random.Random, kind: str, oov: bool
) -> tuple[Any, list[str]]:
    """kind: 'trans' or 'intrans'."""
    if kind == "trans":
        if oov and OOV_TRANS_POOL:
            lemma = rng.choice(OOV_TRANS_POOL)
            tag = ["oov_verb"]
        else:
            lemma = rng.choice(TRANS_KEYS)
            tag = []
        return (
            TransitiveVerb(lemma=lemma, tense_aspect=rng.choice(TENSES)),
            tag,
        )
    if oov and OOV_INTRANS_POOL:
        lemma = rng.choice(OOV_INTRANS_POOL)
        tag = ["oov_verb"]
    else:
        lemma = rng.choice(INTRANS_KEYS)
        tag = []
    return (
        IntransitiveVerb(lemma=lemma, tense_aspect=rng.choice(TENSES)),
        tag,
    )


def _sample_sv(
    rng: random.Random, allow_nominal: bool, force_oov: bool
) -> tuple[SubjectVerbSentence, list[str]]:
    subj, st = _pick_subject(rng, allow_nominal, force_oov)
    verb_kind = rng.choice(["trans", "intrans"])
    verb, vt = _pick_verb(rng, verb_kind, force_oov)
    tags = ["sv"] + st + vt + ([f"verb_{verb_kind}"])
    return SubjectVerbSentence(subject=subj, verb=verb), tags


def _sample_svo(
    rng: random.Random, allow_nominal: bool, force_oov: bool
) -> tuple[SubjectVerbObjectSentence, list[str]]:
    subj, st = _pick_subject(rng, allow_nominal, force_oov)
    obj, ot = _pick_object(rng, allow_nominal, force_oov)
    verb, vt = _pick_verb(rng, "trans", force_oov)
    tags = ["svo"] + st + ot + vt
    return (
        SubjectVerbObjectSentence(subject=subj, verb=verb, object=obj),
        tags,
    )


def _sample_one(rng: random.Random, svo_frac: float, allow_nominal: bool, force_oov: bool):
    is_svo = rng.random() < svo_frac
    if is_svo:
        s, tags = _sample_svo(rng, allow_nominal, force_oov)
        return s, tags, "svo"
    s, tags = _sample_sv(rng, allow_nominal, force_oov)
    return s, tags, "sv"


def sample_all(
    n: int,
    seed: int,
    nominalization_frac: float,
    oov_frac: float,
    svo_frac: float,
    multi_frac: float,
    max_sentences: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    attempts = 0
    target_attempts = n * 8

    while len(records) < n and attempts < target_attempts:
        attempts += 1
        # Decide multi vs single. Weight multi toward 2 over 3.
        if rng.random() < multi_frac and max_sentences >= 2:
            n_sentences = 2 if rng.random() < 0.75 else min(3, max_sentences)
        else:
            n_sentences = 1
        allow_nominal_record = rng.random() < nominalization_frac
        force_oov_record = rng.random() < oov_frac

        structures: list[dict[str, Any]] = []
        tags: list[str] = []
        types: list[str] = []
        try:
            for i in range(n_sentences):
                # Decorrelate nominalization / OOV slightly across clauses
                allow_nominal = allow_nominal_record and rng.random() < 0.7
                force_oov = force_oov_record and rng.random() < 0.7
                s, t, ty = _sample_one(rng, svo_frac, allow_nominal, force_oov)
                structures.append(s.model_dump(mode="json"))
                tags.extend(t)
                types.append(ty)
        except Exception:
            continue

        if n_sentences > 1:
            tags.append(f"multi_{n_sentences}")

        key = json.dumps(structures, sort_keys=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        records.append(
            {
                "id": len(records),
                "n_sentences": n_sentences,
                "types": types,
                "tags": sorted(set(tags)),
                "structured": structures,
            }
        )

    if len(records) < n:
        print(
            f"warning: only produced {len(records)} unique records after "
            f"{attempts} attempts (target {n})",
            file=sys.stderr,
        )
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000, help="Number of unique structures to sample")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for stratified sampling (reproducible)")
    p.add_argument("--nominalization-frac", type=float, default=0.20,
                   help="Fraction of structures that may use nominalized subject/object")
    p.add_argument("--oov-frac", type=float, default=0.20,
                   help="Fraction of structures that include at least one OOV head/lemma")
    p.add_argument("--svo-frac", type=float, default=0.5,
                   help="Fraction of structures that are SVO (rest are SV)")
    p.add_argument("--multi-frac", type=float, default=0.25,
                   help="Fraction of records containing >1 clause (teaches multi-clause outputs)")
    p.add_argument("--max-sentences", type=int, default=3,
                   help="Upper bound on sentences per record")
    p.add_argument("--out", default=str(OUT / "structures.jsonl"))
    args = p.parse_args()

    out = Path(args.out)
    records = sample_all(
        n=args.n,
        seed=args.seed,
        nominalization_frac=args.nominalization_frac,
        oov_frac=args.oov_frac,
        svo_frac=args.svo_frac,
        multi_frac=args.multi_frac,
        max_sentences=args.max_sentences,
    )
    n = jsonl_write(out, records)

    by_tag: dict[str, int] = {}
    for r in records:
        for t in r["tags"]:
            by_tag[t] = by_tag.get(t, 0) + 1
    print(f"wrote {n} structures to {out}", file=sys.stderr)
    for t, c in sorted(by_tag.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {c:5d}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
