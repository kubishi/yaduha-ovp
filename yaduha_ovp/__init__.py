from pydantic import BaseModel, Field
from typing import Dict, Generator, List, Optional, Tuple, Type, Union
from enum import Enum
from random import choice, randint

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_ovp.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Lookup dictionaries for easy access
NOUN_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in INTRANSITIVE_VERBS}


def get_noun_target(lemma: str) -> str:
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
    return f"[{lemma}]"

def get_transitive_verb_target(lemma: str) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"

def get_intransitive_verb_target(lemma: str) -> str:
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"

def get_verb_target(lemma: str) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"

LENIS_MAP = {
    'p': 'b',
    't': 'd',
    'k': 'g',
    's': 'z',
    'm': 'w̃'
}

def to_lenis(word: str) -> str:
    """Convert a word to its lenis form"""
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word


# ============================================================================
# GRAMMATICAL ENUMERATIONS
# ============================================================================

class Proximity(str, Enum):
    proximal = "proximal"
    distal = "distal"

    def get_object_suffix(self, does_end_in_glottal: bool) -> str:
        if self == Proximity.proximal:
            return "eika" if does_end_in_glottal else "neika"
        else:
            return "uka" if does_end_in_glottal else "noka"

    def get_subject_suffix(self) -> str:
        if self == Proximity.proximal:
            return "ii"
        else:
            return "uu"

class Plurality(str, Enum):
    singular = "singular"
    dual = "dual"
    plural = "plural"

class TenseAspect(str, Enum):
    past_simple = "past_simple"
    past_continuous = "past_continuous"
    present_perfect = "present_perfect"
    present_simple = "present_simple"
    present_continuous = "present_continuous"
    future_simple = "future_simple"

    def get_suffix(self) -> str:
        if self == TenseAspect.past_simple:
            return "ku"
        elif self in (TenseAspect.past_continuous, TenseAspect.present_continuous):
            return "ti"
        elif self == TenseAspect.present_perfect:
            return "pü"
        elif self == TenseAspect.present_simple:
            return "dü"
        elif self == TenseAspect.future_simple:
            return "wei"

        raise ValueError("Invalid tense/aspect combination")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Pronoun(str, Enum):
    I = "I"
    we_two = "we (two)"
    we_inclusive = "we (inclusive)"
    we_exclusive = "we (exclusive)"
    you = "you"
    you_all = "you (plural)"
    he_she_it_proximal = "he/she/it (proximal)"
    he_she_it_distal = "he/she/it (distal)"
    they_proximal = "they (proximal)"
    they_distal = "they (distal)"
    reflexive = "self (reflexive)"

SUBJECT_PRONOUNS: Dict[Pronoun, str] = {
    Pronoun.I: 'nüü',
    Pronoun.we_two: 'taa',
    Pronoun.we_inclusive: 'taagwa',
    Pronoun.we_exclusive: 'nüügwa',
    Pronoun.you: 'üü',
    Pronoun.you_all: 'üügwa',
    Pronoun.he_she_it_proximal: 'mahu',
    Pronoun.he_she_it_distal: 'uhu',
    Pronoun.they_proximal: 'mahuw̃a',
    Pronoun.they_distal: 'uhuw̃a',
}

OBJECT_PRONOUNS: Dict[Pronoun, str] = {
    Pronoun.I: 'i',
    Pronoun.we_two: 'ta',
    Pronoun.we_inclusive: 'tei',
    Pronoun.we_exclusive: 'ni',
    Pronoun.you: 'ü',
    Pronoun.you_all: 'üi',
    Pronoun.he_she_it_proximal: 'a',
    Pronoun.he_she_it_distal: 'u',
    Pronoun.they_proximal: 'ai',
    Pronoun.they_distal: 'ui',
    Pronoun.reflexive: 'na',
}

def _third_person_object_pronoun(proximity: Proximity, plurality: Plurality) -> Pronoun:
    if plurality == Plurality.singular:
        return Pronoun.he_she_it_proximal if proximity == Proximity.proximal else Pronoun.he_she_it_distal
    else:
        return Pronoun.they_proximal if proximity == Proximity.proximal else Pronoun.they_distal

class Verb(BaseModel):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'description': 'A verb lemma (transitive or intransitive). '
                f'Known verbs: {", ".join(entry.english for entry in TRANSITIVE_VERBS + INTRANSITIVE_VERBS)}. '
                'If the exact verb is not in this list, use the English lemma as a placeholder.'
        }
    )
    tense_aspect: TenseAspect

class TransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'description': 'A transitive verb lemma. '
                f'Known transitive verbs: {", ".join(entry.english for entry in TRANSITIVE_VERBS)}. '
                'If the exact verb is not in this list, use the English lemma as a placeholder.'
        }
    )

class IntransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'description': 'An intransitive verb lemma. '
                f'Known intransitive verbs: {", ".join(entry.english for entry in INTRANSITIVE_VERBS)}. '
                'If the exact verb is not in this list, use the English lemma as a placeholder.'
        }
    )

class Noun(BaseModel):
    head: str = Field(
        ...,
        json_schema_extra={
            'description': 'A noun lemma. '
                f'Known nouns: {", ".join(entry.english for entry in NOUNS)}. '
                'If the exact noun is not in this list, use the English lemma as a placeholder.'
        }
    )
    possessive_determiner: Optional[Pronoun] = None
    proximity: Proximity
    plurality: Plurality

class SubjectNoun(Noun):
    pass

class ObjectNoun(Noun):
    def get_matching_pronoun_prefix(self) -> str:
        pronoun = _third_person_object_pronoun(self.proximity, self.plurality)
        return OBJECT_PRONOUNS[pronoun]

class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    subject: Union[SubjectNoun, Pronoun]
    verb: Union[TransitiveVerb, IntransitiveVerb]

    def __str__(self) -> str:
        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = SUBJECT_PRONOUNS[self.subject]
        elif isinstance(self.subject, SubjectNoun):
            target_word = get_noun_target(self.subject.head)
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{target_word}-{subject_suffix}"

        verb_stem = get_verb_target(self.verb.lemma)
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_str = f"{verb_stem}-{verb_suffix}"

        return f"{subject_str} {verb_str}"

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbSentence', None, None]:
        """Generate n sample sentences (string representations)"""
        for _ in range(n):
            # Random subject
            if randint(0, 1) == 0:
                subject = choice(list(SUBJECT_PRONOUNS.keys()))
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )

            # Random verb
            if randint(0, 1) == 0:
                verb = IntransitiveVerb(
                    lemma=choice(list(INTRANSITIVE_VERB_LOOKUP.keys())),
                    tense_aspect=choice(list(TenseAspect))
                )
            else:
                verb = TransitiveVerb(
                    lemma=choice(list(TRANSITIVE_VERB_LOOKUP.keys())),
                    tense_aspect=choice(list(TenseAspect))
                )

            yield cls(subject=subject, verb=verb)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        examples = [
            (
                "I sleep.",
                SubjectVerbSentence(
                    subject=Pronoun.I,
                    verb=IntransitiveVerb(
                        lemma="sleep",
                        tense_aspect=TenseAspect.present_simple
                    )
                )
            ),
            (
                "The coyote runs.",
                SubjectVerbSentence(
                    subject=SubjectNoun(
                        head="coyote",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=IntransitiveVerb(
                        lemma="run",
                        tense_aspect=TenseAspect.present_simple
                    )
                )
            ),
            (
                "The mountains will hit.",
                SubjectVerbSentence(
                    subject=SubjectNoun(
                        head="mountain",
                        proximity=Proximity.distal,
                        plurality=Plurality.plural
                    ),
                    verb=IntransitiveVerb(
                        lemma="hit",
                        tense_aspect=TenseAspect.future_simple
                    )
                )
            )
        ]

        return examples

class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    subject: Union[SubjectNoun, Pronoun]
    verb: TransitiveVerb
    object: Union[ObjectNoun, Pronoun]

    def __str__(self) -> str:
        object_pronoun_prefix = None
        if isinstance(self.object, Pronoun):
            object_pronoun_prefix = OBJECT_PRONOUNS[self.object]
        elif isinstance(self.object, ObjectNoun):
            object_pronoun_prefix = self.object.get_matching_pronoun_prefix()

        verb_stem = get_transitive_verb_target(self.verb.lemma) if self.object is not None else get_intransitive_verb_target(self.verb.lemma)
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_stem = to_lenis(verb_stem)
        verb_str = f"{object_pronoun_prefix}-{verb_stem}-{verb_suffix}"

        object_str = None
        if isinstance(self.object, ObjectNoun):
            target_word = get_noun_target(self.object.head)
            does_end_in_glottal = target_word.endswith("'")
            object_suffix = self.object.proximity.get_object_suffix(does_end_in_glottal)
            object_str = f"{target_word}-{object_suffix}"

        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = SUBJECT_PRONOUNS[self.subject]
        elif isinstance(self.subject, SubjectNoun):
            target_word = get_noun_target(self.subject.head)
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{target_word}-{subject_suffix}"

        if object_str is None:
            return f"{verb_str} {subject_str}"
        else:
            return f"{subject_str} {object_str} {verb_str}"

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbObjectSentence', None, None]:
        """Generate n sample sentences (string representations)"""
        for _ in range(n):
            # Random subject
            if randint(0, 1) == 0:
                subject = choice(list(SUBJECT_PRONOUNS.keys()))
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )

            # Random verb
            verb_lemma = choice(list(TRANSITIVE_VERB_LOOKUP.keys()))
            verb = TransitiveVerb(
                lemma=verb_lemma,
                tense_aspect=choice(list(TenseAspect))
            )

            # Random object for transitive verbs
            if randint(0, 1) == 0:
                obj = ObjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )
            else:
                obj = choice(list(OBJECT_PRONOUNS.keys()))

            yield cls(subject=subject, verb=verb, object=obj)

    @classmethod
    def sample(cls, n: int) -> List['SubjectVerbObjectSentence']:
        """Generate n sample sentences (string representations)"""
        return list(cls.sample_iter(n))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        examples = [
            (
                "You read the mountains.",
                SubjectVerbObjectSentence(
                    subject=Pronoun.you,
                    verb=TransitiveVerb(
                        lemma="read",
                        tense_aspect=TenseAspect.present_simple
                    ),
                    object=ObjectNoun(
                        head="mountain",
                        proximity=Proximity.distal,
                        plurality=Plurality.plural
                    )
                ),
            ),
            (
                "That worm will hear it.",
                SubjectVerbObjectSentence(
                    subject=SubjectNoun(
                        head="worm",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=TransitiveVerb(
                        lemma="hear",
                        tense_aspect=TenseAspect.future_simple
                    ),
                    object=Pronoun.he_she_it_distal
                )
            ),
            (
                "That food cooks this weasle.",
                SubjectVerbObjectSentence(
                    subject=SubjectNoun(
                        head="food",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=TransitiveVerb(
                        lemma="cook",
                        tense_aspect=TenseAspect.present_simple
                    ),
                    object=ObjectNoun(
                        head="weasle",
                        proximity=Proximity.proximal,
                        plurality=Plurality.singular
                    )
                )
            )
        ]

        return examples


language = Language(
    code="ovp",
    name="Owens Valley Paiute",
    sentence_types=(SubjectVerbSentence, SubjectVerbObjectSentence),
)
