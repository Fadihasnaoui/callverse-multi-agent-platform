"""
"Portrait humain" of a persona: who they are, why they travel now, how they talk, what moves them,
what they keep to themselves, and how they decide.

Stored as JSON in customer_profiles.human_profile. Every field is optional: what a persona does not
define comes from its type (archetypes.HUMAN_DEFAULTS), so old and generated personas still work.
The prompt gets it as a first-person identity card plus a voice guide, never as a list of fields.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from app.customers.archetypes import get_archetype


class Companion(BaseModel):
    name: str
    role: str = ""                  # "mari", "fille de 8 ans", "ami"
    decides: bool = False           # has a say in the booking


class Identity(BaseModel):
    name: str = ""                  # real name when the profile name is a label ("Famille Trabelsi")
    age: Optional[int] = None
    job: str = ""
    city: str = ""
    family: str = ""
    companions: List[Companion] = Field(default_factory=list)


class Story(BaseModel):
    why_now: str = ""               # the reason for THIS trip, now
    past_trips: List[str] = Field(default_factory=list)
    tunisia_link: str = ""


class BigFive(BaseModel):
    """1 (very low) to 5 (very high)."""
    openness: int = Field(3, ge=1, le=5)
    conscientiousness: int = Field(3, ge=1, le=5)
    extraversion: int = Field(3, ge=1, le=5)
    agreeableness: int = Field(3, ge=1, le=5)
    neuroticism: int = Field(3, ge=1, le=5)


class Voice(BaseModel):
    address: str = "vouvoie"        # vouvoie | tutoie
    sentence_length: str = "moyen"  # court | moyen | long
    tics: List[str] = Field(default_factory=list)          # "bon", "franchement"
    derja_words: List[str] = Field(default_factory=list)   # Tunisian words, only for Tunisian personas
    never_says: List[str] = Field(default_factory=list)


class Emotions(BaseModel):
    reassured_by: List[str] = Field(default_factory=list)
    annoyed_by: List[str] = Field(default_factory=list)
    delighted_by: List[str] = Field(default_factory=list)


class HiddenFact(BaseModel):
    fact: str
    reveal_if: str                  # what the advisor must ask or do to hear it


class Decision(BaseModel):
    must_haves: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)
    delighters: List[str] = Field(default_factory=list)
    threshold: Optional[float] = Field(None, ge=0.3, le=0.95)
    patience: Optional[int] = Field(None, ge=1, le=15)
    decides_alone: bool = True
    payment: str = ""               # "acompte", "carte", "virement"


class HumanProfile(BaseModel):
    # The situation this persona comes with and how hard it is to convince (empty: its type's usual ones)
    scenario: str = ""
    difficulty: str = ""
    identity: Identity = Field(default_factory=Identity)
    story: Story = Field(default_factory=Story)
    big_five: BigFive = Field(default_factory=BigFive)
    voice: Voice = Field(default_factory=Voice)
    emotions: Emotions = Field(default_factory=Emotions)
    hidden: List[HiddenFact] = Field(default_factory=list)
    decision: Decision = Field(default_factory=Decision)


def load_human(profile) -> HumanProfile:
    """The persona's portrait, with the gaps filled from its type."""
    raw = getattr(profile, "human_profile", None) or {}
    try:
        human = HumanProfile.model_validate(raw)
    except ValueError:
        human = HumanProfile()  # a malformed portrait never breaks a simulation
    preset = get_archetype(getattr(profile, "archetype", None))["human"]
    d = human.decision
    d.must_haves = d.must_haves or list(preset["must_haves"])
    d.threshold = d.threshold if d.threshold is not None else preset["threshold"]
    d.patience = d.patience if d.patience is not None else preset["patience"]
    from app.conversation.scenarios import DIFFICULTIES, SCENARIOS
    human.scenario = human.scenario if human.scenario in SCENARIOS else preset["scenario"]
    human.difficulty = human.difficulty if human.difficulty in DIFFICULTIES else preset["difficulty"]
    return human


# Big Five level -> how it shows in conversation (only marked traits are described)
_TRAITS = {
    "openness": ("je préfère ce que je connais et je me méfie des suggestions inhabituelles", "je suis curieux(se) et ouvert(e) aux suggestions"),
    "conscientiousness": ("je ne m'attarde pas sur les détails", "je veux des détails précis et je note tout"),
    "extraversion": ("je parle peu, mes réponses sont brèves", "je suis bavard(e) et je raconte volontiers ma vie"),
    "agreeableness": ("je peux être sec(sèche), voire cassant(e)", "je reste poli(e) et chaleureux(se), même agacé(e)"),
    "neuroticism": ("je suis serein(e), rien ne m'inquiète vraiment", "je m'inquiète vite et j'imagine le pire"),
}


def personality_text(big_five: BigFive) -> str:
    parts = []
    for trait, (low, high) in _TRAITS.items():
        level = getattr(big_five, trait)
        if level <= 2:
            parts.append(low)
        elif level >= 4:
            parts.append(high)
    return "; ".join(parts)


def identity_card(profile, human: HumanProfile) -> str:
    """First-person portrait, in French (the prompt language)."""
    i, s = human.identity, human.story
    lines = [f"Je m'appelle {i.name or profile.name}" + (f", j'ai {i.age} ans" if i.age else f" ({profile.age_range} ans)")
             + (f", je suis {i.job}" if i.job else "") + (f" et j'habite à {i.city}" if i.city else "") + "."]
    if i.family:
        lines.append(f"Ma situation : {i.family}.")
    if i.companions:
        who = ", ".join(f"{c.name} ({c.role})" if c.role else c.name for c in i.companions)
        lines.append(f"Je pars avec {who}.")
    if s.why_now:
        lines.append(f"Pourquoi ce voyage maintenant : {s.why_now}.")
    for trip in s.past_trips:
        lines.append(f"Souvenir de voyage qui compte pour moi : {trip}.")
    if s.tunisia_link:
        lines.append(f"Mon lien avec la Tunisie : {s.tunisia_link}.")
    traits = personality_text(human.big_five)
    if traits:
        lines.append(f"Mon caractère : {traits}.")
    e = human.emotions
    if e.reassured_by:
        lines.append(f"Ce qui me rassure : {', '.join(e.reassured_by)}.")
    if e.annoyed_by:
        lines.append(f"Ce qui m'agace : {', '.join(e.annoyed_by)}.")
    if e.delighted_by:
        lines.append(f"Ce qui me ferait craquer : {', '.join(e.delighted_by)}.")
    return "\n".join(lines)


def voice_guide(human: HumanProfile) -> str:
    v = human.voice
    length = {"court": "phrases courtes, souvent une seule", "long": "tu développes, phrases plus longues",
              }.get(v.sentence_length, "phrases de longueur variable")
    lines = [f"- Tu {'tutoies' if v.address == 'tutoie' else 'vouvoies'} l'agent ; {length}."]
    if v.tics:
        lines.append(f"- Tes petits mots à toi (de temps en temps, pas à chaque message) : {', '.join(v.tics)}.")
    if v.derja_words:
        lines.append(f"- Il t'arrive de glisser un mot de tunisien, rarement et naturellement : {', '.join(v.derja_words)}.")
    if v.never_says:
        lines.append(f"- Tu ne dis jamais : {', '.join(v.never_says)}.")
    return "\n".join(lines)


def hidden_facts_text(human: HumanProfile, revealed: list) -> str:
    """Secrets still kept, with what unlocks them. Revealed ones go to memory instead."""
    kept = [h for h in human.hidden if h.fact not in (revealed or [])]
    if not kept:
        return "- aucune"
    return "\n".join(f"- {h.fact} (tu ne le dis QUE si : {h.reveal_if})" for h in kept)


if __name__ == "__main__":
    from types import SimpleNamespace
    p = SimpleNamespace(name="Nadia", age_range="40-50", archetype="anxious", human_profile={
        "identity": {"age": 47, "job": "pharmacienne", "city": "Sfax", "companions": [{"name": "Riadh", "role": "mari", "decides": True}]},
        "big_five": {"neuroticism": 5, "extraversion": 2},
        "voice": {"tics": ["bon"]}, "hidden": [{"fact": "budget réel 2000 TND", "reveal_if": "on demande sa marge"}]})
    h = load_human(p)
    assert h.decision.patience == 8 and h.decision.must_haves == ["annulation possible", "sécurité de la zone"]
    card = identity_card(p, h)
    assert "j'ai 47 ans" in card and "Riadh (mari)" in card and "je m'inquiète vite" in card
    assert "budget réel" in hidden_facts_text(h, []) and hidden_facts_text(h, ["budget réel 2000 TND"]) == "- aucune"
    assert load_human(SimpleNamespace(name="X", age_range="30", archetype="vip", human_profile={"big_five": {"openness": 9}})).decision.patience == 5
    print("human ok")
