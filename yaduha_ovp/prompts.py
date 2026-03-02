from typing import Iterable, List, Type, TYPE_CHECKING
from yaduha_ovp.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS
from yaduha_ovp import (
    LENIS_MAP, SUBJECT_PRONOUNS, OBJECT_PRONOUNS,
    SubjectVerbSentence, SubjectVerbObjectSentence,
)

if TYPE_CHECKING:
    from yaduha.language import Sentence

SYSTEM_PROMPT_PREFIX = (
    "You are a translator that translates English sentences into Owens Valley Paiute. "
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible. "
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n"
)

TOOL_USE_INSTRUCTION = (
    "You may also have access to tools that can help you produce a better translation. "
    "Use these tools as needed. You can make one or many tool calls (in parallel and/or sequentially) "
    "until you decide to respond.\n"
)

VOCABULARY_PROMPT = (
    "You use the following vocabulary to translate user input sentences from English to Owens Valley Paiute.\n" +
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible.\n" +
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n" +
    "# Vocabulary\n" +
    "## Nouns: \n" + "\n".join([f"{noun.target}: {noun.english}" for noun in NOUNS]) + "\n" +
    "## Transitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in TRANSITIVE_VERBS]) + "\n" +
    "## Intransitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in INTRANSITIVE_VERBS]) + "\n"
)

SENTENCE_STRUCTURE_PROMPT = (
    "# Grammar\n"
    "\n"
    "## Sentence Types\n"
    "\n"
    "### Subject-Verb (SV) — intransitive or transitive verb without an object\n"
    "Word order: SUBJECT VERB\n"
    "  [subject] [verb stem]-[tense suffix]\n"
    "The subject is either a subject pronoun or a noun with a subject suffix.\n"
    "\n"
    "### Subject-Object-Verb (SOV) — transitive verb with a noun object\n"
    "Word order: SUBJECT OBJECT VERB\n"
    "  [subject] [object noun]-[object suffix] [object pronoun prefix]-[lenis verb stem]-[tense suffix]\n"
    "The object noun appears between subject and verb. The verb is prefixed with a 3rd-person\n"
    "object pronoun matching the object noun's proximity and number, and the verb stem\n"
    "undergoes initial consonant lenition (see below).\n"
    "\n"
    "### Pronoun-Object Transitive — transitive verb with a pronoun object (no object noun)\n"
    "Word order: VERB SUBJECT\n"
    "  [object pronoun prefix]-[lenis verb stem]-[tense suffix] [subject]\n"
    "When the object is a pronoun (not a full noun), the verb comes first.\n"
    "The verb is prefixed with the object pronoun and undergoes lenition.\n"
    "\n"
    "## Subject Forms\n"
    "A subject is either a pronoun or a noun with a proximity suffix:\n"
    "- Pronoun subject: use the subject pronoun directly\n"
    "- Noun subject: [noun]-[subject suffix]\n"
    "  - Proximal (this/these): -ii\n"
    "  - Distal (that/those): -uu\n"
    "\n"
    "## Object Noun Suffixes\n"
    "Object nouns take a suffix based on proximity. The suffix form depends on\n"
    "whether the noun stem ends in a glottal stop ('):\n"
    "- Proximal — glottal-final: -eika, otherwise: -neika\n"
    "- Distal — glottal-final: -uka, otherwise: -noka\n"
    "\n"
    "## Tense/Aspect Verb Suffixes\n"
    "- Past simple: -ku\n"
    "- Past continuous: -ti\n"
    "- Present continuous: -ti\n"
    "- Present perfect: -pü\n"
    "- Present simple: -dü\n"
    "- Future simple: -wei\n"
    "\n"
    "## Subject Pronouns\n"
    + "\n".join(f"- {p.value}: {form}" for p, form in SUBJECT_PRONOUNS.items())
    + "\n\n"
    "## Object Pronoun Prefixes\n"
    "In transitive sentences, the verb is prefixed with an object pronoun.\n"
    "When the object is a full noun, use the 3rd-person pronoun matching its proximity/number.\n"
    + "\n".join(f"- {p.value}: {form}-" for p, form in OBJECT_PRONOUNS.items())
    + "\n\n"
    "## Initial Consonant Lenition (Fortis → Lenis)\n"
    "When a verb takes an object pronoun prefix, its initial consonant changes:\n"
    + ", ".join(f"{f} → {l}" for f, l in LENIS_MAP.items())
    + "\n"
    "If the verb does not start with one of these consonants, it is unchanged.\n"
)

def get_prompt(include_vocab: bool,
               has_tools: bool = False,
               include_examples: Iterable[Type["Sentence"]] | None = None) -> str:
    include_examples = include_examples or []
    system_prompt = SYSTEM_PROMPT_PREFIX
    if has_tools:
        system_prompt += TOOL_USE_INSTRUCTION
    if include_vocab:
        system_prompt += VOCABULARY_PROMPT
    system_prompt += SENTENCE_STRUCTURE_PROMPT
    for sentence_cls in include_examples:
        for source, example_sentence in sentence_cls.get_examples():
            system_prompt += (
                "\n# Example\n"
                f"English: {source}\n"
                f"Owens Valley Paiute: {example_sentence}\n"
            )

    return system_prompt
