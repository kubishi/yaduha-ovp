"""Step 3b: teach proper-noun handling.

Covers two behaviors that the general sample_structures + paraphrase steps
don't reliably hit:

1. Single-reference: proper nouns go in `SubjectNoun.head` / `ObjectNoun.head`
   (so the deterministic renderer emits `[Name]` as an OOV placeholder in the
   OVP surface). Example:
       "Susan runs."  →  [SubjectVerbSentence(subject=SubjectNoun(head="Susan"),
                                              verb=IntransitiveVerb(lemma="run"))]

2. Multi-clause coreference: when the English uses the same name across
   clauses, the structured target uses the `Name` noun in the FIRST clause and
   a 3rd-person pronoun in subsequent clauses. Example:
       "Susan is eating and drinking."
           →  [SubjectVerbSentence(subject=SubjectNoun(head="Susan"), verb=eat),
                SubjectVerbSentence(subject=Pronoun.he_she_it_distal, verb=drink)]

Record schema mirrors oov_substitutions.jsonl:
    {
        "id": <int>, "kind": "proper_single_subject" | ... ,
        "name": "Susan", "n_sentences": 1 | 2,
        "types": ["sv", "sv"], "tags": [...],
        "structured": [<sentence dict>, ...],
        "english": <sentence>, "errors": [...]
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
from oov_lemmas import PROPER_NOUNS  # noqa: E402

load_dotenv()

TRANS_KEYS = list(TRANSITIVE_VERB_LOOKUP.keys())
INTRANS_KEYS = list(INTRANSITIVE_VERB_LOOKUP.keys())
TENSES = list(TenseAspect)
# third-person subject pronouns only (he/she/it)
THIRD_SUBJ_PRONOUNS = [Pronoun.he_she_it_proximal, Pronoun.he_she_it_distal]
THIRD_OBJ_PRONOUNS = [Pronoun.he_she_it_proximal, Pronoun.he_she_it_distal]
I_OR_YOU = [Pronoun.I, Pronoun.you, Pronoun.we_inclusive]


class EnglishSentence(BaseModel):
    text: str = Field(..., description="A natural English sentence matching the structured target.")


def _sub_noun(name: str, rng: random.Random) -> SubjectNoun:
    return SubjectNoun(
        head=name,
        proximity=rng.choice([Proximity.proximal, Proximity.distal]),
        plurality=Plurality.singular,
    )


def _obj_noun(name: str, rng: random.Random) -> ObjectNoun:
    return ObjectNoun(
        head=name,
        proximity=rng.choice([Proximity.proximal, Proximity.distal]),
        plurality=Plurality.singular,
    )


def _coref_subj_pronoun(first_subject: SubjectNoun) -> Pronoun:
    """Pick a 3rd-person subject pronoun with matching proximity."""
    if first_subject.proximity == Proximity.proximal:
        return Pronoun.he_she_it_proximal
    return Pronoun.he_she_it_distal


def _coref_obj_pronoun(first_object: ObjectNoun) -> Pronoun:
    if first_object.proximity == Proximity.proximal:
        return Pronoun.he_she_it_proximal
    return Pronoun.he_she_it_distal


def _sv(subject, verb_kind: str, rng: random.Random) -> SubjectVerbSentence:
    if verb_kind == "trans":
        v = TransitiveVerb(lemma=rng.choice(TRANS_KEYS), tense_aspect=rng.choice(TENSES))
    else:
        v = IntransitiveVerb(lemma=rng.choice(INTRANS_KEYS), tense_aspect=rng.choice(TENSES))
    return SubjectVerbSentence(subject=subject, verb=v)


def _svo(subject, obj, rng: random.Random) -> SubjectVerbObjectSentence:
    v = TransitiveVerb(lemma=rng.choice(TRANS_KEYS), tense_aspect=rng.choice(TENSES))
    return SubjectVerbObjectSentence(subject=subject, verb=v, object=obj)


# ---------- record builders ----------

def build_single_subject(name: str, rng: random.Random) -> tuple[list[Any], list[str]]:
    """[SV or SVO] with `name` as subject."""
    subj = _sub_noun(name, rng)
    if rng.random() < 0.5:
        # SV (intransitive)
        s = _sv(subj, "intrans", rng)
        return [s], ["sv"]
    # SVO with a non-name object
    # (mix of random common-noun object or pronoun object)
    if rng.random() < 0.5:
        from oov_lemmas import POSITIVE_NOUNS as _  # noqa: F401
        from yaduha_ovp import NOUN_LOOKUP
        obj_head = rng.choice(list(NOUN_LOOKUP.keys()))
        obj = ObjectNoun(head=obj_head,
                         proximity=rng.choice([Proximity.proximal, Proximity.distal]),
                         plurality=rng.choice([Plurality.singular, Plurality.plural]))
    else:
        obj = rng.choice(THIRD_OBJ_PRONOUNS + [Pronoun.I, Pronoun.you])
    s = _svo(subj, obj, rng)
    return [s], ["svo"]


def build_single_object(name: str, rng: random.Random) -> tuple[list[Any], list[str]]:
    """[SVO] with `name` as object."""
    subj = rng.choice(I_OR_YOU + THIRD_SUBJ_PRONOUNS)
    obj = _obj_noun(name, rng)
    s = _svo(subj, obj, rng)
    return [s], ["svo"]


def build_multi_subject_coref(name: str, rng: random.Random) -> tuple[list[Any], list[str]]:
    """[SV/SVO with name subject, SV/SVO with coref pronoun subject]"""
    subj_name = _sub_noun(name, rng)
    # First clause: SV or SVO
    if rng.random() < 0.5:
        s1 = _sv(subj_name, rng.choice(["trans", "intrans"]), rng)
    else:
        # SVO with random object
        from yaduha_ovp import NOUN_LOOKUP
        obj = ObjectNoun(
            head=rng.choice(list(NOUN_LOOKUP.keys())),
            proximity=rng.choice([Proximity.proximal, Proximity.distal]),
            plurality=rng.choice([Plurality.singular, Plurality.plural]),
        )
        s1 = _svo(subj_name, obj, rng)
    # Second clause: coref pronoun subject
    subj_pron = _coref_subj_pronoun(subj_name)
    if rng.random() < 0.5:
        s2 = _sv(subj_pron, rng.choice(["trans", "intrans"]), rng)
    else:
        from yaduha_ovp import NOUN_LOOKUP
        obj2 = ObjectNoun(
            head=rng.choice(list(NOUN_LOOKUP.keys())),
            proximity=rng.choice([Proximity.proximal, Proximity.distal]),
            plurality=rng.choice([Plurality.singular, Plurality.plural]),
        )
        s2 = _svo(subj_pron, obj2, rng)
    types = ["svo" if isinstance(x, SubjectVerbObjectSentence) else "sv" for x in (s1, s2)]
    return [s1, s2], types


def build_multi_object_coref(name: str, rng: random.Random) -> tuple[list[Any], list[str]]:
    """[SVO with name as object, SVO with coref pronoun as object]."""
    obj_name = _obj_noun(name, rng)
    subj1 = rng.choice(I_OR_YOU + THIRD_SUBJ_PRONOUNS)
    s1 = _svo(subj1, obj_name, rng)
    # Second clause: same or different subject; coref pronoun object
    obj_pron = _coref_obj_pronoun(obj_name)
    subj2 = rng.choice(I_OR_YOU + THIRD_SUBJ_PRONOUNS)
    s2 = _svo(subj2, obj_pron, rng)
    return [s1, s2], ["svo", "svo"]


BUILDERS: dict[str, Any] = {
    "proper_single_subject": build_single_subject,
    "proper_single_object": build_single_object,
    "proper_multi_subject_coref": build_multi_subject_coref,
    "proper_multi_object_coref": build_multi_object_coref,
}


# ---------- LLM prompting ----------

SYS_SINGLE = (
    "You write natural English sentences for low-resource MT training.\n"
    "You receive a structured sentence that uses a proper name as the subject or "
    "object. Write ONE natural English sentence that matches the structured target "
    "and uses the proper name verbatim. Keep it short."
)

SYS_MULTI_SUBJ = (
    "You write natural English sentences for low-resource MT training.\n"
    "You receive a 2-clause structured target where:\n"
    "  - clause 1 has a PROPER NAME as subject\n"
    "  - clause 2 has a 3rd-person PRONOUN (he/she/it) as subject, coreferent with the name\n"
    "Write ONE natural English sentence that covers BOTH clauses. In your English, "
    "the FIRST mention of the person uses the name; the SECOND mention uses a pronoun "
    "(he/she/it/her/him). Example:\n"
    "  structure = [Susan eats, <pronoun> drinks]\n"
    "  english   = 'Susan is eating, and then she drinks.' "
    "(or 'Susan eats and drinks.' — implicit coreference via dropped subject)\n"
    "Keep it short and natural."
)

SYS_MULTI_OBJ = (
    "You write natural English sentences for low-resource MT training.\n"
    "You receive a 2-clause structured target where:\n"
    "  - clause 1 has a PROPER NAME as object\n"
    "  - clause 2 has a 3rd-person PRONOUN (he/she/it → him/her/it) as object, "
    "coreferent with the name\n"
    "Write ONE natural English sentence that covers BOTH clauses. First object "
    "mention uses the name; second uses a pronoun. Example:\n"
    "  structure = [I see Tom, I talk_to <pronoun>]\n"
    "  english   = 'I see Tom and talk to him.'\n"
    "Keep it short and natural."
)


def _ask(agent, sys_prompt: str, structured: list[Any], name: str) -> str:
    payload = {
        "name": name,
        "structured": [s.model_dump(mode="json") for s in structured],
    }
    resp = agent.get_response(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        response_format=EnglishSentence,
    )
    return resp.content.text.strip()


def _system_for(kind: str) -> str:
    if kind.startswith("proper_single"):
        return SYS_SINGLE
    if kind == "proper_multi_subject_coref":
        return SYS_MULTI_SUBJ
    return SYS_MULTI_OBJ


# ---------- driver ----------

def make_seeds(rng: random.Random, per_name: int, names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in names:
        for kind, builder in BUILDERS.items():
            for _ in range(per_name):
                try:
                    structured, types = builder(name, rng)
                except Exception:
                    continue
                out.append({
                    "kind": kind,
                    "name": name,
                    "n_sentences": len(structured),
                    "types": types,
                    "structured_obj": structured,
                })
    rng.shuffle(out)
    return out


def process_one(seed: dict[str, Any], model: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": seed["id"],
        "kind": seed["kind"],
        "name": seed["name"],
        "n_sentences": seed["n_sentences"],
        "types": seed["types"],
        "structured": [s.model_dump(mode="json") for s in seed["structured_obj"]],
        "english": None,
        "errors": [],
    }
    try:
        agent = make_openai(model, temperature=0.7)
        out["english"] = _ask(agent, _system_for(seed["kind"]), seed["structured_obj"], seed["name"])
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-name", type=int, default=3,
                   help="Records per (name, kind). Total ≈ len(PROPER_NOUNS)*4*per_name.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for structure sampling (reproducible)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="English-author model (gpt-* or any Ollama tag)")
    p.add_argument("--parallel", type=int, default=8,
                   help="Concurrent LLM calls")
    p.add_argument("--output", default=str(OUT / "proper_nouns.jsonl"),
                   help="JSONL output path (resumable)")
    p.add_argument("--names", default=None,
                   help="Comma-separated list of names to use; defaults to PROPER_NOUNS.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total records (smoke-test use)")
    args = p.parse_args()

    names = args.names.split(",") if args.names else PROPER_NOUNS
    rng = random.Random(args.seed)
    seeds = make_seeds(rng, args.per_name, names)
    for i, s in enumerate(seeds):
        s["id"] = i
    if args.limit:
        seeds = seeds[: args.limit]

    out_path = Path(args.output)
    done_ids = {r["id"] for r in jsonl_iter(out_path) if r.get("english")}
    todo = [s for s in seeds if s["id"] not in done_ids]
    print(f"names={len(names)} per_name={args.per_name} total={len(seeds)} "
          f"done={len(done_ids)} todo={len(todo)} out={out_path}", file=sys.stderr)
    if not todo:
        return 0

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(process_one, s, args.model): s for s in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            jsonl_append(out_path, rec)
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
            status = "OK" if not rec["errors"] else "ERR"
            print(
                f"[{completed}/{len(todo)}] {status} {rec['kind']:>28s} "
                f"{rec['name']:>8s} n={rec['n_sentences']} "
                f"{elapsed:5.1f}s {rate:4.2f}/s ETA {eta:5.0f}s "
                f":: {(rec.get('english') or '')[:60]}",
                file=sys.stderr,
            )
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
