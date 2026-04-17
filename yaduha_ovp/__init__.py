from pydantic import BaseModel, Field
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from enum import Enum
from random import choice, randint

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_ovp.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Lookup dictionaries for easy access
NOUN_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in INTRANSITIVE_VERBS}


def get_noun_target(lemma: str, mask: bool = False) -> str:
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
    return "[NOUN]" if mask else f"[{lemma}]"

def get_transitive_verb_target(lemma: str, mask: bool = False) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    return "[VERB]" if mask else f"[{lemma}]"

def get_intransitive_verb_target(lemma: str, mask: bool = False) -> str:
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return "[VERB]" if mask else f"[{lemma}]"

def get_verb_target(lemma: str, mask: bool = False) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return "[VERB]" if mask else f"[{lemma}]"


def _noun_in_vocab(lemma: str) -> bool:
    return lemma in NOUN_LOOKUP


def _verb_in_vocab(lemma: str) -> bool:
    return lemma in TRANSITIVE_VERB_LOOKUP or lemma in INTRANSITIVE_VERB_LOOKUP

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


class NominalizerTense(str, Enum):
    """Tense for agentive verb nominalization ('the one who ___')."""
    present = "present"   # -dü   ('the one who runs')
    future = "future"     # -weidü ('the one who will run')

    def get_suffix(self) -> str:
        return "dü" if self == NominalizerTense.present else "weidü"

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


# ============================================================================
# AGENTIVE VERB NOMINALIZATION
#   Any verb can function as a noun using the nominalizer suffix:
#     poyoha-dü-ii     "the one who runs"   (proximal subject)
#     poyoha-dü-uu     "the one who runs"   (distal subject)
#     poyoha-weidü-ii  "the one who will run" (proximal subject)
#     poyoha-dü-oka → poyoha-doka  (distal object; morphophonemic ü-drop)
#     poyoha-dü-neika → poyoha-deika (proximal object)
# ============================================================================

class NominalBase(BaseModel):
    """A noun formed by nominalizing a verb ('the one who ___')."""
    verb_lemma: str = Field(
        ...,
        json_schema_extra={
            'description': (
                'A verb lemma from the OVP vocabulary, used here as an '
                'agentive noun ("the one who ___"). '
                f'Known transitive verbs: {", ".join(entry.english for entry in TRANSITIVE_VERBS)}. '
                f'Known intransitive verbs: {", ".join(entry.english for entry in INTRANSITIVE_VERBS)}. '
                'If the exact verb is not in this list, use the English lemma as a placeholder.'
            )
        }
    )
    nominalizer_tense: NominalizerTense = Field(
        ...,
        json_schema_extra={
            'description': (
                'Tense of the nominalizer. "present" renders as -dü ("the one who runs"); '
                '"future" renders as -weidü ("the one who will run").'
            )
        },
    )
    proximity: Proximity
    plurality: Plurality
    possessive_determiner: Optional[Pronoun] = None


class NominalSubject(NominalBase):
    """Agentive-nominalized subject (e.g. 'the runner'/'the one who runs')."""
    pass


class NominalObject(NominalBase):
    """Agentive-nominalized object (e.g. 'the runner' as direct object)."""
    def get_matching_pronoun_prefix(self) -> str:
        pronoun = _third_person_object_pronoun(self.proximity, self.plurality)
        return OBJECT_PRONOUNS[pronoun]

    def get_object_suffix(self, does_end_in_glottal: bool) -> str:
        """Object suffix fused with the nominalizer: -dü + -neika/-noka → -deika/-doka.

        The vowel in the nominalizer drops before the vowel-initial object suffix,
        so we drop the last character of the nominalizer and append eika/oka.
        """
        nom = self.nominalizer_tense.get_suffix()  # "dü" or "weidü"
        if self.proximity == Proximity.proximal:
            return f"{nom[:-1]}eika"    # dü → deika, weidü → weideika
        else:
            return f"{nom[:-1]}oka"      # dü → doka, weidü → weidoka

def _mask_subject_field(subject: Any, oov: List[str]) -> None:
    """Mutate a subject slot in-place, replacing OOV heads/lemmas with sentinels."""
    if isinstance(subject, SubjectNoun) and not _noun_in_vocab(subject.head):
        oov.append(subject.head)
        subject.head = "[NOUN]"
    elif isinstance(subject, NominalSubject) and not _verb_in_vocab(subject.verb_lemma):
        oov.append(subject.verb_lemma)
        subject.verb_lemma = "[VERB]"


def _mask_object_field(obj: Any, oov: List[str]) -> None:
    if isinstance(obj, ObjectNoun) and not _noun_in_vocab(obj.head):
        oov.append(obj.head)
        obj.head = "[NOUN]"
    elif isinstance(obj, NominalObject) and not _verb_in_vocab(obj.verb_lemma):
        oov.append(obj.verb_lemma)
        obj.verb_lemma = "[VERB]"


def _mask_verb_field(verb: Any, oov: List[str]) -> None:
    if not _verb_in_vocab(verb.lemma):
        oov.append(verb.lemma)
        verb.lemma = "[VERB]"


class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    subject: Union[SubjectNoun, NominalSubject, Pronoun]
    verb: Union[TransitiveVerb, IntransitiveVerb]

    def _render(self, mask: bool) -> str:
        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = SUBJECT_PRONOUNS[self.subject]
        elif isinstance(self.subject, NominalSubject):
            verb_stem = get_verb_target(self.subject.verb_lemma, mask=mask)
            nom_suffix = self.subject.nominalizer_tense.get_suffix()
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{verb_stem}-{nom_suffix}-{subject_suffix}"
        elif isinstance(self.subject, SubjectNoun):
            target_word = get_noun_target(self.subject.head, mask=mask)
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{target_word}-{subject_suffix}"

        verb_stem = get_verb_target(self.verb.lemma, mask=mask)
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_str = f"{verb_stem}-{verb_suffix}"

        return f"{subject_str} {verb_str}"

    def __str__(self) -> str:
        return self._render(mask=False)

    def str_masked(self) -> str:
        return self._render(mask=True)

    def masked_copy(self) -> Tuple["SubjectVerbSentence", List[str]]:
        clone = self.model_copy(deep=True)
        oov: List[str] = []
        _mask_subject_field(clone.subject, oov)
        _mask_verb_field(clone.verb, oov)
        return clone, oov

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
            ),
            (
                "The runner is singing.",
                SubjectVerbSentence(
                    subject=NominalSubject(
                        verb_lemma="run",
                        nominalizer_tense=NominalizerTense.present,
                        proximity=Proximity.distal,
                        plurality=Plurality.singular,
                    ),
                    verb=IntransitiveVerb(
                        lemma="sing",
                        tense_aspect=TenseAspect.present_continuous,
                    ),
                ),
            ),
            (
                "The one who will cook is laughing.",
                SubjectVerbSentence(
                    subject=NominalSubject(
                        verb_lemma="cook",
                        nominalizer_tense=NominalizerTense.future,
                        proximity=Proximity.proximal,
                        plurality=Plurality.singular,
                    ),
                    verb=IntransitiveVerb(
                        lemma="laugh",
                        tense_aspect=TenseAspect.present_continuous,
                    ),
                ),
            ),
        ]

        return examples

class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    subject: Union[SubjectNoun, NominalSubject, Pronoun]
    verb: TransitiveVerb
    object: Union[ObjectNoun, NominalObject, Pronoun]

    def _render(self, mask: bool) -> str:
        object_pronoun_prefix = None
        if isinstance(self.object, Pronoun):
            object_pronoun_prefix = OBJECT_PRONOUNS[self.object]
        elif isinstance(self.object, (ObjectNoun, NominalObject)):
            object_pronoun_prefix = self.object.get_matching_pronoun_prefix()

        if self.object is not None:
            verb_stem = get_transitive_verb_target(self.verb.lemma, mask=mask)
        else:
            verb_stem = get_intransitive_verb_target(self.verb.lemma, mask=mask)
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_stem = to_lenis(verb_stem)
        verb_str = f"{object_pronoun_prefix}-{verb_stem}-{verb_suffix}"

        object_str = None
        if isinstance(self.object, NominalObject):
            obj_verb_stem = get_verb_target(self.object.verb_lemma, mask=mask)
            # morphophonemic fusion: nominalizer + object suffix (dü+oka → doka)
            does_end_in_glottal = obj_verb_stem.endswith("'")
            object_suffix = self.object.get_object_suffix(does_end_in_glottal)
            object_str = f"{obj_verb_stem}-{object_suffix}"
        elif isinstance(self.object, ObjectNoun):
            target_word = get_noun_target(self.object.head, mask=mask)
            does_end_in_glottal = target_word.endswith("'")
            object_suffix = self.object.proximity.get_object_suffix(does_end_in_glottal)
            object_str = f"{target_word}-{object_suffix}"

        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = SUBJECT_PRONOUNS[self.subject]
        elif isinstance(self.subject, NominalSubject):
            subj_verb_stem = get_verb_target(self.subject.verb_lemma, mask=mask)
            nom_suffix = self.subject.nominalizer_tense.get_suffix()
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{subj_verb_stem}-{nom_suffix}-{subject_suffix}"
        elif isinstance(self.subject, SubjectNoun):
            target_word = get_noun_target(self.subject.head, mask=mask)
            subject_suffix = self.subject.proximity.get_subject_suffix()
            subject_str = f"{target_word}-{subject_suffix}"

        if object_str is None:
            return f"{verb_str} {subject_str}"
        else:
            return f"{subject_str} {object_str} {verb_str}"

    def __str__(self) -> str:
        return self._render(mask=False)

    def str_masked(self) -> str:
        return self._render(mask=True)

    def masked_copy(self) -> Tuple["SubjectVerbObjectSentence", List[str]]:
        clone = self.model_copy(deep=True)
        oov: List[str] = []
        _mask_subject_field(clone.subject, oov)
        _mask_verb_field(clone.verb, oov)
        _mask_object_field(clone.object, oov)
        return clone, oov

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
            ),
            (
                "I see the runner.",
                SubjectVerbObjectSentence(
                    subject=Pronoun.I,
                    verb=TransitiveVerb(
                        lemma="see",
                        tense_aspect=TenseAspect.present_simple,
                    ),
                    object=NominalObject(
                        verb_lemma="run",
                        nominalizer_tense=NominalizerTense.present,
                        proximity=Proximity.distal,
                        plurality=Plurality.singular,
                    ),
                ),
            ),
            (
                "The one who cooks sees the one who will eat.",
                SubjectVerbObjectSentence(
                    subject=NominalSubject(
                        verb_lemma="cook",
                        nominalizer_tense=NominalizerTense.present,
                        proximity=Proximity.distal,
                        plurality=Plurality.singular,
                    ),
                    verb=TransitiveVerb(
                        lemma="see",
                        tense_aspect=TenseAspect.present_simple,
                    ),
                    object=NominalObject(
                        verb_lemma="eat",
                        nominalizer_tense=NominalizerTense.future,
                        proximity=Proximity.distal,
                        plurality=Plurality.singular,
                    ),
                ),
            ),
        ]

        return examples


def _get_instructions() -> str:
    from yaduha_ovp.prompts import get_prompt
    return get_prompt(
        include_vocab=True,
        include_examples=(SubjectVerbSentence, SubjectVerbObjectSentence),
    )

language = Language(
    code="ovp",
    name="Owens Valley Paiute",
    sentence_types=(SubjectVerbSentence, SubjectVerbObjectSentence),
    get_instructions=_get_instructions,
)
