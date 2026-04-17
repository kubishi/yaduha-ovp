"""Shared helpers for the datagen pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from yaduha.agent import Agent
from yaduha.agent.openai import OpenAIAgent

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


def normalize_english(s: str) -> str:
    """Cheap normalization for dedup keys."""
    return " ".join(s.lower().split()).rstrip(".!?")
