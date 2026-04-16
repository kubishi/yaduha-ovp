"""Translate English → Owens Valley Paiute using a local Ollama model.

Requires Ollama running at http://localhost:11434 with the target model pulled:
    ollama pull llama3.2:3b
"""

import os

from dotenv import load_dotenv

from yaduha.agent.ollama import OllamaAgent
from yaduha.translator.pipeline import PipelineTranslator

load_dotenv()

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

agent = OllamaAgent(model=MODEL, base_url=BASE_URL, temperature=0.0)

translator = PipelineTranslator.from_language(language_code="ovp", agent=agent)

sentences = [
    "The dog is sleeping.",
    "I see the mountain.",
    "The coyote runs.",
]

print(f"Model: {MODEL} @ {BASE_URL}\n")

for text in sentences:
    result = translator(text)
    print(f"EN: {result.source}")
    print(f"PA: {result.target}")
    print(f"BT: {result.back_translation.source}")
    print(f"   ({result.translation_time:.2f}s, "
          f"{result.prompt_tokens + result.completion_tokens} tokens)")
    print()
