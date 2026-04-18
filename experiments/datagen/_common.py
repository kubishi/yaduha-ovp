"""Shared helpers for the datagen pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from yaduha.agent import Agent
from yaduha.agent.ollama import OllamaAgent
from yaduha.agent.openai import OpenAIAgent

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def make_openai(model: str, temperature: float = 0.7) -> Agent:
    return OpenAIAgent(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temperature,
    )


def make_agent(model: str, temperature: float = 0.7,
               ollama_url: str = "http://localhost:11434") -> Agent:
    """OpenAI for 'gpt-*' tags, otherwise Ollama. Lets datagen stages swap
    between closed and open models via a tag string."""
    if model.startswith("gpt-"):
        return make_openai(model, temperature=temperature)
    return OllamaAgent(model=model, base_url=ollama_url, temperature=temperature)


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


def normalize_english(s: str) -> str:
    """Cheap normalization for dedup keys."""
    return " ".join(s.lower().split()).rstrip(".!?")


def clean(s: str) -> str:
    """Strip, ensure terminal punctuation, and capitalize the first letter.
    Used across the eval pipeline to normalize LLM outputs before scoring."""
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    if s:
        s = s[0].upper() + s[1:]
    return s


def parse_structured(d: dict[str, Any]):
    """Parse a structured-sentence dict into the correct OVP Sentence
    subclass. The presence of an ``object`` field distinguishes
    SubjectVerbObjectSentence from SubjectVerbSentence."""
    # Imported locally so this module stays importable without the OVP
    # language package installed (useful for framework-level tooling).
    from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)
