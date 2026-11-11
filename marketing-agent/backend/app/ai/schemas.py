"""Output contracts for every AI task. The model's reply is parsed into one of these;
anything that does not fit is repaired or rejected - bad output never reaches the database."""
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_HHMM = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")


class ContentOut(BaseModel):
    caption: str = Field(min_length=20)
    hashtags: list[str] = Field(default_factory=list, max_length=15)
    video_script: str = ""
    visual_brief: str = ""
    overlay_lines: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("hashtags", mode="before")
    @classmethod
    def _clean_tags(cls, value):
        if isinstance(value, str):
            value = re.split(r"[\s,]+", value)
        cleaned = []
        seen = set()
        for tag in value or []:
            normalized = re.sub(r"\s+", "", str(tag).lstrip("#").strip())
            key = normalized.casefold()
            if normalized and key not in seen:
                cleaned.append(normalized)
                seen.add(key)
        return cleaned

    @field_validator("caption")
    @classmethod
    def _strip(cls, value):
        return value.strip()

    @field_validator("overlay_lines", mode="before")
    @classmethod
    def _clean_overlay_lines(cls, value):
        return [str(line).strip()[:42] for line in (value or []) if str(line).strip()][:5]


class SlotOut(BaseModel):
    day_offset: int = Field(ge=0, le=6)
    time: str
    platform: str
    content_type: str
    topic: str = Field(min_length=5)

    @field_validator("time")
    @classmethod
    def _time(cls, value):
        if not _HHMM.match(value.strip()):
            raise ValueError("time must look like HH:MM")
        return value.strip()


class PlanOut(BaseModel):
    slots: list[SlotOut] = Field(min_length=1)


class ScheduleOut(BaseModel):
    proposed_datetime: str
    rationale: str = ""

    @field_validator("proposed_datetime")
    @classmethod
    def _iso(cls, value):
        from datetime import datetime
        datetime.fromisoformat(value.replace("Z", "+00:00"))  # raises ValueError -> repaired
        return value


class SpokenOut(BaseModel):
    spoken_text: str = Field(min_length=10, max_length=900)


class ProofreadOut(BaseModel):
    text: str = Field(min_length=1)


class IntentOut(BaseModel):
    buying_intent: int = Field(ge=0, le=100)


class SummaryOut(BaseModel):
    summary: str = Field(min_length=10)
    recommendations: list[str] = Field(default_factory=list, max_length=6)


class CommentSentimentItem(BaseModel):
    id: str
    sentiment: str = "neutral"
    topics: list[str] = Field(default_factory=list, max_length=5)
    buying_intent: int = Field(default=0, ge=0, le=100)

    @field_validator("sentiment")
    @classmethod
    def _valid_sentiment(cls, value):
        allowed = {"positive", "negative", "neutral", "question"}
        v = value.strip().lower()
        return v if v in allowed else "neutral"


class CommentSentimentOut(BaseModel):
    comments: list[CommentSentimentItem] = Field(default_factory=list)


class CompactOut(BaseModel):
    bullets: list[str] = Field(min_length=1, max_length=12)


class AuditKPISummary(BaseModel):
    reach: str = ""
    engagement_rate: str = ""
    follower_trend: str = ""
    consistency: str = ""


class AuditOut(BaseModel):
    strengths: list[str] = Field(default_factory=list, max_length=6)
    weaknesses: list[str] = Field(default_factory=list, max_length=6)
    opportunities: list[str] = Field(default_factory=list, max_length=5)
    kpi_summary: AuditKPISummary = Field(default_factory=AuditKPISummary)
    platform_comparison: str = ""
    closing_note: str = ""

    # Backwards compat: old audit had "findings"
    findings: list[Any] = Field(default_factory=list, max_length=8)
