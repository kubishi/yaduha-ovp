"""Shared helpers for the datagen pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from yaduha.agent import Agent
from yaduha.agent.openai import OpenAIAgent
from yaduha_ovp import (
    INTRANSITIVE_VERB_LOOKUP,
    NOUN_LOOKUP,
    TRANSITIVE_VERB_LOOKUP,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def make_openai(model: str, temperature: float = 0.7) -> Agent:
    return OpenAIAgent(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temperature,
    )


def jsonl_iter(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def jsonl_write(path: Path, records: Iterable[dict[str, Any]], mode: str = "w") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open(mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def mask_oov(sentence: Any) -> tuple[Any, list[str]]:
    """Mirror of experiments/run_translations.py::mask_oov.

    Replaces OOV noun heads and verb lemmas with role-tagged sentinels in the
    structured JSON, so the strong LLM cannot cheat by reading the original
    English lemma.

    NOTE: like the eval pipeline, this currently does not mask `verb_lemma`
    on NominalSubject/NominalObject — keep behaviors aligned. If that is ever
    fixed in run_translations.py, mirror the change here.
    """
    clone = sentence.model_copy(deep=True)
    oov: list[str] = []
    for field in ("subject", "object"):
        part = getattr(clone, field, None)
        if part is not None and hasattr(part, "head"):
            if part.head not in NOUN_LOOKUP:
                oov.append(part.head)
                part.head = "[NOUN]"
    verb = getattr(clone, "verb", None)
    if verb is not None and hasattr(verb, "lemma"):
        in_vocab = (
            verb.lemma in TRANSITIVE_VERB_LOOKUP
            or verb.lemma in INTRANSITIVE_VERB_LOOKUP
        )
        if not in_vocab:
            oov.append(verb.lemma)
            verb.lemma = "[VERB]"
    return clone, oov


def normalize_english(s: str) -> str:
    """Cheap normalization for dedup keys."""
    return " ".join(s.lower().split()).rstrip(".!?")
