"""Step 3: explicit OOV substitution training pairs.

POSITIVE pairs teach the forward model: when the English contains an OOV word
that has a clean in-vocab hypernym, prefer the hypernym in the structured
target (gpt-4o-mini's biggest demonstrated edge).

NEGATIVE pairs teach the opposite: when there is NO good in-vocab neighbor,
keep the English lemma so the deterministic surface emits a `[lemma]`
placeholder. Without negatives, fine-tuned models hallucinate substitutions.

For each (oov_lemma, in_vocab_hypernym|None), we deterministically build N
random simple structural contexts, then ask the strong LLM for one natural
English sentence per context that uses the OOV word in a way consistent with
the structure.

Output schema:
    {
        "id": <int>, "kind": "positive_noun" | ... | "negative_intrans_verb",
        "oov_lemma": "chihuahua", "in_vocab": "dog" | null,
        "structured": <pydantic dump>, "english": <sentence>, "errors": [...]
    }
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from yaduha_ovp import (
    INTRANSITIVE_VERB_LOOKUP,
    NOUN_LOOKUP,
    TRANSITIVE_VERB_LOOKUP,
    IntransitiveVerb,
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
from _common import OUT, jsonl_append, jsonl_iter, make_openai  # noqa: E402
from oov_lemmas import (  # noqa: E402
    NEGATIVE_INTRANS_VERBS_F,
    NEGATIVE_NOUNS_F,
    NEGATIVE_TRANS_VERBS_F,
    POSITIVE_INTRANS_VERBS,
    POSITIVE_NOUNS,
    POSITIVE_TRANS_VERBS,
)

load_dotenv()

NOUN_KEYS = list(NOUN_LOOKUP.keys())
TRANS_KEYS = list(TRANSITIVE_VERB_LOOKUP.keys())
INTRANS_KEYS = list(INTRANSITIVE_VERB_LOOKUP.keys())
SUBJECT_PRONOUNS = [p for p in Pronoun if p != Pronoun.reflexive]
TENSES = list(TenseAspect)
PROX = list(Proximity)
PLUR = list(Plurality)


class EnglishSentence(BaseModel):
    text: str = Field(
        ...,
        description="A natural English sentence using the requested OOV word.",
    )


SYS_POSITIVE_NOUN = (
    "You produce natural English sentences for low-resource MT training data.\n"
    "You will receive a target structured sentence (using an in-vocabulary hypernym) "
    "plus an OOV word that the English sentence MUST contain in place of the hypernym. "
    "Write ONE natural English sentence whose meaning matches the structure but uses "
    "the OOV word instead of the hypernym. Keep the sentence short."
)

SYS_POSITIVE_VERB = (
    "You produce natural English sentences for low-resource MT training data.\n"
    "You will receive a target structured sentence (using an in-vocabulary hypernym verb) "
    "plus an OOV verb that the English sentence MUST use in place of the hypernym verb. "
    "Write ONE natural English sentence whose dominant predicate uses the OOV verb but "
    "whose meaning still matches the structure. Keep it short."
)

SYS_NEGATIVE_NOUN = (
    "You produce natural English sentences for low-resource MT training data.\n"
    "You will receive a structured sentence containing an OOV English noun that has NO "
    "good in-vocabulary substitute. Write ONE natural English sentence using that noun "
    "in a way consistent with the structure (subject/object slot, tense, etc). Keep it short."
)

SYS_NEGATIVE_VERB = (
    "You produce natural English sentences for low-resource MT training data.\n"
    "You will receive a structured sentence containing an OOV English verb that has NO "
    "good in-vocabulary substitute. Write ONE natural English sentence using that verb "
    "as the main predicate, consistent with the structure. Keep it short."
)


def _rand_subject(rng: random.Random):
    if rng.random() < 0.5:
        return rng.choice(SUBJECT_PRONOUNS)
    return SubjectNoun(
        head=rng.choice(NOUN_KEYS),
        proximity=rng.choice(PROX),
        plurality=rng.choice(PLUR),
    )


def _rand_object(rng: random.Random):
    if rng.random() < 0.4:
        return rng.choice(list(Pronoun))
    return ObjectNoun(
        head=rng.choice(NOUN_KEYS),
        proximity=rng.choice(PROX),
        plurality=rng.choice(PLUR),
    )


def _build_structure_with_noun(
    rng: random.Random, head: str, role: str
) -> Any:
    """Return a sentence that places `head` as either subject or object noun."""
    proximity = rng.choice(PROX)
    plurality = rng.choice(PLUR)
    if role == "subject":
        subj = SubjectNoun(head=head, proximity=proximity, plurality=plurality)
        if rng.random() < 0.5:
            return SubjectVerbSentence(
                subject=subj,
                verb=IntransitiveVerb(
                    lemma=rng.choice(INTRANS_KEYS),
                    tense_aspect=rng.choice(TENSES),
                ),
            )
        return SubjectVerbObjectSentence(
            subject=subj,
            verb=TransitiveVerb(
                lemma=rng.choice(TRANS_KEYS),
                tense_aspect=rng.choice(TENSES),
            ),
            object=_rand_object(rng),
        )
    obj = ObjectNoun(head=head, proximity=proximity, plurality=plurality)
    return SubjectVerbObjectSentence(
        subject=_rand_subject(rng),
        verb=TransitiveVerb(
            lemma=rng.choice(TRANS_KEYS),
            tense_aspect=rng.choice(TENSES),
        ),
        object=obj,
    )


def _build_structure_with_verb(
    rng: random.Random, lemma: str, kind: str
) -> Any:
    if kind == "trans":
        return SubjectVerbObjectSentence(
            subject=_rand_subject(rng),
            verb=TransitiveVerb(lemma=lemma, tense_aspect=rng.choice(TENSES)),
            object=_rand_object(rng),
        )
    return SubjectVerbSentence(
        subject=_rand_subject(rng),
        verb=IntransitiveVerb(lemma=lemma, tense_aspect=rng.choice(TENSES)),
    )


def _ask_english(agent, sys_prompt: str, payload: dict[str, Any]) -> str:
    user = json.dumps(payload, indent=2)
    resp = agent.get_response(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        response_format=EnglishSentence,
    )
    return resp.content.text.strip()


def make_records(
    rng: random.Random,
    per_lemma: int,
) -> list[dict[str, Any]]:
    """Build all (kind, oov_lemma, in_vocab, structured) tuples WITHOUT calling LLM yet."""
    out: list[dict[str, Any]] = []

    def _emit(kind, oov_lemma, in_vocab, role, s, sent_type):
        out.append({
            "kind": kind,
            "oov_lemma": oov_lemma,
            "in_vocab": in_vocab,
            "role": role,
            # List-of-one for consistency with structures.jsonl (multi-clause support).
            "structured": [s.model_dump(mode="json")],
            "n_sentences": 1,
            "types": [sent_type],
            "type": sent_type,  # keep singular alias for stratified split keys
        })

    for oov, hypernym in POSITIVE_NOUNS.items():
        for _ in range(per_lemma):
            role = rng.choice(["subject", "object"])
            s = _build_structure_with_noun(rng, hypernym, role)
            _emit("positive_noun", oov, hypernym, role, s,
                  "svo" if isinstance(s, SubjectVerbObjectSentence) else "sv")
    for oov, hypernym in POSITIVE_TRANS_VERBS.items():
        for _ in range(per_lemma):
            s = _build_structure_with_verb(rng, hypernym, "trans")
            _emit("positive_trans_verb", oov, hypernym, "verb", s, "svo")
    for oov, hypernym in POSITIVE_INTRANS_VERBS.items():
        for _ in range(per_lemma):
            s = _build_structure_with_verb(rng, hypernym, "intrans")
            _emit("positive_intrans_verb", oov, hypernym, "verb", s, "sv")
    for oov in NEGATIVE_NOUNS_F:
        for _ in range(per_lemma):
            role = rng.choice(["subject", "object"])
            s = _build_structure_with_noun(rng, oov, role)
            _emit("negative_noun", oov, None, role, s,
                  "svo" if isinstance(s, SubjectVerbObjectSentence) else "sv")
    for oov in NEGATIVE_TRANS_VERBS_F:
        for _ in range(per_lemma):
            s = _build_structure_with_verb(rng, oov, "trans")
            _emit("negative_trans_verb", oov, None, "verb", s, "svo")
    for oov in NEGATIVE_INTRANS_VERBS_F:
        for _ in range(per_lemma):
            s = _build_structure_with_verb(rng, oov, "intrans")
            _emit("negative_intrans_verb", oov, None, "verb", s, "sv")

    return out


def _system_for(kind: str) -> str:
    if kind == "positive_noun":
        return SYS_POSITIVE_NOUN
    if kind in ("positive_trans_verb", "positive_intrans_verb"):
        return SYS_POSITIVE_VERB
    if kind == "negative_noun":
        return SYS_NEGATIVE_NOUN
    return SYS_NEGATIVE_VERB


def _payload_for(rec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "structured": rec["structured"],
        "oov_word": rec["oov_lemma"],
        "role": rec["role"],
    }
    if rec["in_vocab"]:
        payload["in_vocab_hypernym"] = rec["in_vocab"]
        payload["instruction"] = (
            f"Write an English sentence that uses '{rec['oov_lemma']}' "
            f"in place of '{rec['in_vocab']}'."
        )
    else:
        payload["instruction"] = (
            f"Write an English sentence using '{rec['oov_lemma']}' naturally; "
            f"there is no in-vocabulary equivalent."
        )
    return payload


def process_one(rec: dict[str, Any], model: str) -> dict[str, Any]:
    out = dict(rec)
    out["english"] = None
    out["errors"] = []
    try:
        agent = make_openai(model, temperature=0.7)
        out["english"] = _ask_english(agent, _system_for(rec["kind"]), _payload_for(rec))
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-lemma", type=int, default=4,
                   help="Sentences per OOV lemma (× lemma pool size = total records)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--output", default=str(OUT / "oov_substitutions.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_path = Path(args.output)
    rng = random.Random(args.seed)
    seeds = make_records(rng, args.per_lemma)
    for i, r in enumerate(seeds):
        r["id"] = i
    if args.limit:
        seeds = seeds[: args.limit]

    done_ids = {r["id"] for r in jsonl_iter(out_path) if r.get("english")}
    todo = [r for r in seeds if r["id"] not in done_ids]
    print(
        f"total={len(seeds)} done={len(done_ids)} todo={len(todo)} out={out_path}",
        file=sys.stderr,
    )
    if not todo:
        return 0

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(process_one, r, args.model): r for r in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            jsonl_append(out_path, rec)
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
            status = "OK" if not rec["errors"] else "ERR"
            print(
                f"[{completed}/{len(todo)}] {status} {rec['kind']:>22s} "
                f"{rec['oov_lemma']:>14s} -> {rec['in_vocab'] or '-':<12s} "
                f"{elapsed:6.1f}s, {rate:4.2f}/s, ETA {eta:6.0f}s "
                f":: {(rec.get('english') or '')[:55]}",
                file=sys.stderr,
            )
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
