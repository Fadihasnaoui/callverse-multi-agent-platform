"""Owner style rules: the part of 'brand preferences' that code can check, so it is ENFORCED, not requested.

Free-text memory ("please end with a question") is only best-effort for a language model - measured at
1/8 to 5/8 compliance. These structured rules are put in the prompt AND checked on every output; a
violation is sent back to the model for repair, and the post is never saved in violation."""
import re

from pydantic import BaseModel, Field, field_validator

_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")
_QUESTION_ENDS = ("?", "؟")


class StyleRules(BaseModel):
    banned_words: list[str] = Field(default_factory=list, max_length=40)
    max_emojis: int | None = Field(default=None, ge=0, le=20)
    end_with_question: bool = False
    max_caption_chars: int | None = Field(default=None, ge=40, le=2200)
    max_hashtags: int | None = Field(default=None, ge=0, le=15)

    @field_validator("banned_words", mode="before")
    @classmethod
    def _words(cls, value):
        if isinstance(value, str):
            value = re.split(r"[,\n]+", value)
        return [w.strip() for w in (value or []) if str(w).strip()]

    def is_empty(self) -> bool:
        return not (self.banned_words or self.max_emojis is not None or self.end_with_question
                    or self.max_caption_chars or self.max_hashtags is not None)

    def prompt_block(self) -> str:
        lines = []
        if self.banned_words:
            lines.append("never use these words or phrases (any spelling variant): " + ", ".join(self.banned_words))
        if self.max_emojis is not None:
            lines.append(f"at most {self.max_emojis} emoji in the caption")
        if self.end_with_question:
            lines.append("the caption's last sentence must be a question ending in '?' (a trailing emoji after it is fine)")
        if self.max_caption_chars:
            lines.append(f"the caption must be at most {self.max_caption_chars} characters")
        if self.max_hashtags is not None:
            lines.append(f"at most {self.max_hashtags} hashtags")
        if not lines:
            return ""
        return ("\nSTYLE RULES set by the owner - these are checked by software and any violation is rejected:\n- "
                + "\n- ".join(lines) + "\n")

    def violations(self, caption: str, hashtags: list[str]) -> list[str]:
        problems = []
        low = caption.lower()
        for word in self.banned_words:
            if word.lower() in low:
                problems.append(f"the caption contains the banned word/phrase '{word}'; rewrite without it")
        if self.max_emojis is not None and len(_EMOJI.findall(caption)) > self.max_emojis:
            problems.append(f"the caption has more than {self.max_emojis} emoji; remove some")
        if self.end_with_question and not _EMOJI.sub("", caption).rstrip().endswith(_QUESTION_ENDS):
            problems.append("the caption's last sentence must be a question ending in '?'; rewrite it as a question")
        if self.max_caption_chars and len(caption) > self.max_caption_chars:
            problems.append(f"the caption is {len(caption)} characters; the limit is {self.max_caption_chars}")
        if self.max_hashtags is not None and len(hashtags) > self.max_hashtags:
            problems.append(f"use at most {self.max_hashtags} hashtags")
        return problems


def guard_for(rules: StyleRules):
    """A client.run guard for ContentOut."""
    def guard(output) -> list[str]:
        return rules.violations(output.caption, output.hashtags)
    return guard
