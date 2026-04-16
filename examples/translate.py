"""Translate English → Owens Valley Paiute using the local yaduha + yaduha-ovp packages."""

import os
from pathlib import Path

from yaduha.agent.openai import OpenAIAgent
from yaduha.translator.pipeline import PipelineTranslator

from dotenv import load_dotenv

load_dotenv()

agent = OpenAIAgent(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.0,
)

translator = PipelineTranslator.from_language(language_code="ovp", agent=agent)

sentences = [
    "The dog is sleeping.",
    "I see the mountain.",
    "The coyote runs.",
]

for text in sentences:
    result = translator(text)
    print(f"EN: {result.source}")
    print(f"PA: {result.target}")
    print(f"BT: {result.back_translation.source}")
    print(f"   ({result.translation_time:.2f}s, "
          f"{result.prompt_tokens + result.completion_tokens} tokens)")
    print()
