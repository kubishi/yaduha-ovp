"""Curated OOV English lemmas used to teach the forward model when to substitute
an in-vocab hypernym vs. when to emit a placeholder.

POSITIVE_*  — OOV lemma maps to an in-vocab head/lemma the model should pick.
NEGATIVE_*  — OOV lemma has no good in-vocab neighbor; target should keep the
              English lemma so the deterministic surface emits `[lemma]`.

Anything that turns out to actually be in-vocab is filtered at import time.
"""

from __future__ import annotations

from yaduha_ovp import (
    INTRANSITIVE_VERB_LOOKUP,
    NOUN_LOOKUP,
    TRANSITIVE_VERB_LOOKUP,
)

_POSITIVE_NOUNS_RAW: dict[str, str] = {
    "chihuahua": "dog",
    "poodle": "dog",
    "labrador": "dog",
    "puppy": "dog",
    "kitten": "cat",
    "stallion": "horse",
    "pony": "horse",
    "donkey": "mule",
    "wheat": "rice",
    "barley": "rice",
    "creek": "river",
    "stream": "river",
    "stool": "chair",
    "bench": "chair",
    "hill": "mountain",
    "peak": "mountain",
    "ridge": "mountain",
    "meal": "food",
    "lunch": "food",
    "dinner": "food",
    "breakfast": "food",
    "snack": "food",
    "oak": "tree",
    "pine": "tree",
    "cedar": "tree",
    "willow": "tree",
    "cabin": "house",
    "cottage": "house",
    "mansion": "house",
    "shed": "house",
    "mug": "cup",
    "glass": "cup",
    "stick": "wood",
    "log": "wood",
    "branch": "wood",
    "stone": "rock",
    "boulder": "rock",
    "pebble": "rock",
    "rabbit": "cottontail",
    "hare": "jackrabbit",
    "fruit": "apple",
    "robin": "bird",
    "sparrow": "bird",
    "crow": "bird",
    "hawk": "eagle",
    "salmon": "fish",
    "trout": "fish",
    "bass": "fish",
    "tea": "coffee",
    "espresso": "coffee",
    "cub": "bear",
    "blade": "knife",
    "dagger": "knife",
}

_POSITIVE_TRANS_VERBS_RAW: dict[str, str] = {
    "consume": "eat",
    "devour": "eat",
    "munch": "eat",
    "spot": "see",
    "notice": "see",
    "observe": "see",
    "view": "see",
    "glimpse": "see",
    "sip": "drink",
    "gulp": "drink",
    "guzzle": "drink",
    "sniff": "smell",
    "strike": "hit",
    "punch": "hit",
    "smack": "hit",
    "slap": "hit",
    "address": "talk_to",
    "greet": "talk_to",
    "pursue": "chase",
    "follow": "chase",
    "ascend": "climb",
    "scale": "climb",
    "prepare": "cook",
    "bake": "cook",
    "fry": "cook",
    "boil": "cook",
    "study": "read",
    "peruse": "read",
    "scribble": "write",
    "draft": "write",
    "compose": "write",
    "stop_by": "visit",
    "discover": "find",
    "locate": "find",
    "spot_out": "find",
}

_POSITIVE_INTRANS_VERBS_RAW: dict[str, str] = {
    "rest": "sit",
    "perch": "sit",
    "slumber": "sleep",
    "doze": "sleep",
    "nap": "sleep",
    "snooze": "sleep",
    "sprint": "run",
    "jog": "run",
    "dash": "run",
    "scamper": "run",
    "depart": "go",
    "leave": "go",
    "stroll": "walk",
    "hike": "walk",
    "march": "walk",
    "amble": "walk",
    "rise": "stand",
    "recline": "lie_down",
    "speak": "talk",
    "chat": "talk",
    "converse": "talk",
    "tumble": "fall",
    "topple": "fall",
    "labor": "work",
    "toil": "work",
    "grin": "smile",
    "beam": "smile",
    "chant": "sing",
    "hum": "sing",
    "croon": "sing",
    "chuckle": "laugh",
    "giggle": "laugh",
    "snicker": "laugh",
    "frolic": "play",
    "leap": "jump",
    "hop": "jump",
    "twirl": "dance",
    "waltz": "dance",
    "paddle": "swim",
}

NEGATIVE_NOUNS: list[str] = [
    "laptop", "phone", "computer", "television", "radio",
    "car", "truck", "airplane", "bicycle", "motorcycle",
    "guitar", "piano", "violin", "drum",
    "doctor", "lawyer", "engineer", "scientist", "astronaut",
    "robot", "wizard", "dragon", "unicorn",
    "internet", "email", "website", "app",
    "pizza", "hamburger", "sandwich", "burrito",
    "factory", "skyscraper", "subway", "highway",
]

NEGATIVE_TRANS_VERBS: list[str] = [
    "program", "google", "tweet", "text", "photograph",
    "email", "patent", "audit", "broadcast", "recycle",
]

NEGATIVE_INTRANS_VERBS: list[str] = [
    "skateboard", "snowboard", "vlog", "podcast", "meditate",
    "vibrate", "glitch", "buffer", "ferment",
]

# Personal names. Teach the model that names are kept as head (→ [Name] in the
# surface form) and, in multi-clause contexts, replaced by a 3rd-person pronoun
# on subsequent references (coreference).
PROPER_NOUNS: list[str] = [
    # traditionally-female
    "Susan", "Alice", "Mary", "Emma", "Sophia", "Olivia",
    "Anna", "Elsa", "Sarah", "Lisa", "Emily", "Rachel",
    # traditionally-male
    "John", "Tom", "Bob", "Mike", "David", "James",
    "Peter", "Jack", "Mark", "Paul", "Daniel", "Ben",
    # gender-neutral or either
    "Alex", "Jordan", "Taylor", "Sam", "Kim", "Casey",
]


def _filter_in_vocab(d: dict[str, str], lookup: dict) -> dict[str, str]:
    return {k: v for k, v in d.items() if k not in lookup}


def _filter_list(xs: list[str], *lookups: dict) -> list[str]:
    return [x for x in xs if all(x not in lk for lk in lookups)]


POSITIVE_NOUNS = _filter_in_vocab(_POSITIVE_NOUNS_RAW, NOUN_LOOKUP)
POSITIVE_TRANS_VERBS = _filter_in_vocab(_POSITIVE_TRANS_VERBS_RAW, TRANSITIVE_VERB_LOOKUP)
POSITIVE_INTRANS_VERBS = _filter_in_vocab(_POSITIVE_INTRANS_VERBS_RAW, INTRANSITIVE_VERB_LOOKUP)
NEGATIVE_NOUNS_F = _filter_list(NEGATIVE_NOUNS, NOUN_LOOKUP)
NEGATIVE_TRANS_VERBS_F = _filter_list(NEGATIVE_TRANS_VERBS, TRANSITIVE_VERB_LOOKUP)
NEGATIVE_INTRANS_VERBS_F = _filter_list(NEGATIVE_INTRANS_VERBS, INTRANSITIVE_VERB_LOOKUP)
