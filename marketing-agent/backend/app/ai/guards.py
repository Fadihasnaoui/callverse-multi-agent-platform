"""Deterministic checks run on model output. A guard returns a list of plain-language problems;
the client feeds them back to the model for one repair, or rejects the output. This turns
"the prompt says never invent a price" into something enforced in code."""
import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".").rstrip("0").rstrip(".") if "." in n.replace(",", ".") else n for n in _NUMBER.findall(text or "")}


def no_invented_numbers(*allowed_sources: str, fields=("caption", "video_script")):
    """Every number in the model's text must already appear in the material it was given
    (the brief, the owner's idea). Catches invented prices, dates, star ratings."""
    allowed = set()
    for source in allowed_sources:
        allowed |= _numbers(source)

    def guard(output) -> list[str]:
        problems = []
        for field in fields:
            invented = sorted(_numbers(getattr(output, field, "") or "") - allowed)
            if invented:
                problems.append(
                    f"'{field}' contains number(s) {invented} that do not appear in the information you were "
                    "given. Remove them or rephrase without any figure that was not provided."
                )
        return problems

    return guard


def length_close_to(original: str, tolerance: float = 0.35):
    """For proofreading: the corrected text may fix words, not rewrite the message."""
    def guard(output) -> list[str]:
        text = getattr(output, "text", "")
        if abs(len(text) - len(original)) > tolerance * max(len(original), 1):
            return ["The corrected text differs too much in length from the original. Fix only misspelled "
                    "words; do not add, remove or reword anything."]
        if abs(len(text.split()) - len(original.split())) > max(3, 0.25 * len(original.split())):
            return ["The corrected text has a very different number of words. Keep the same words, fixing only spelling."]
        return []
    return guard


def only_values(field: str, allowed: list[str]):
    def guard(output) -> list[str]:
        value = getattr(output, field, None)
        return [] if value in allowed else [f"'{field}' must be one of {allowed}, got {value!r}"]
    return guard


_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def plain_speech(field: str):
    """Text that will be read aloud: no hashtags, links, or emojis."""
    def guard(output) -> list[str]:
        text = getattr(output, field, "") or ""
        if "#" in text or "http" in text.lower() or "www." in text.lower() or _EMOJI.search(text):
            return [f"'{field}' will be spoken aloud: remove hashtags, links and emojis, use plain spoken words only."]
        return []
    return guard


def all_of(*guards):
    def guard(output) -> list[str]:
        problems: list[str] = []
        for g in guards:
            problems += g(output)
        return problems
    return guard


_CTA = re.compile(r"(r[ée]servez|contactez|appelez|[ée]crivez|envoyez|cliquez|inscrivez|commandez|whatsapp|\bdm\b)", re.I)


def no_call_to_action(field: str = "caption"):
    """The app appends its own 'message us' link; a model-written call-to-action would duplicate it."""
    def guard(output) -> list[str]:
        match = _CTA.search(getattr(output, field, "") or "")
        return [f"'{field}' contains a call-to-action ('{match.group(0)}'). Remove it: a message link is added automatically. "
                "A closing question is fine."] if match else []
    return guard


def useful_hashtags(minimum: int = 6):
    """Reject empty/generic hashtag sets while still allowing an owner to disable hashtags."""
    generic = {"travel", "voyage", "vacances", "tourism", "tourisme", "instagood", "love"}

    def guard(output) -> list[str]:
        tags = getattr(output, "hashtags", []) or []
        if len(tags) < minimum:
            return [f"Provide at least {minimum} relevant hashtags with a mix of brand, destination, audience and intent tags."]
        if len({tag.casefold() for tag in tags} - generic) < 4:
            return ["The hashtags are too generic. Add specific destination, Tunisian travel, audience and booking-intent tags."]
        return []

    return guard


def natural_marketing_copy(field: str = "caption"):
    clichés = (
        "inoubliable", "moments magiques", "vous attend", "trésors cachés", "expérience unique",
        "destination de rêve", "paradis", "prêt à", "unforgettable", "dream destination", "paradise awaits",
    )

    def guard(output) -> list[str]:
        text = getattr(output, field, "") or ""
        lower = text.casefold()
        found = [phrase for phrase in clichés if phrase.casefold() in lower]
        problems = [f"'{field}' uses generic AI/tourism cliché(s): {', '.join(found)}. Replace them with precise, natural language."] if found else []
        if "!" in text:
            problems.append(f"'{field}' uses exclamation marks. Remove them and keep the tone confident and natural.")
        return problems

    return guard
