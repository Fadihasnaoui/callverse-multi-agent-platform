"""Written run report: one short LLM-written section per agent, from the run's real figures.

The model only writes prose around facts it is given. Every number it writes
must appear in those facts; otherwise the section is retried once with the
offending numbers named, and kept with a warning if it still invents one.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Any, Awaitable, Callable

SECTIONS = [
    ("overview", "Run overview", "the whole run: how many calls, how they ended, the overall quality and the main lesson"),
    ("amira", "Amira — customer advisor", "how Amira handled the calls: completion, repetition loops, quality scores by client type"),
    ("quality", "Quality Analyst", "what the Quality Analyst found: scores, the most frequent findings and the AI-checked issues"),
    ("workforce", "Workforce Manager", "how calls were routed: AI versus human advisers, escalations, waiting for an adviser"),
    ("clients", "Client simulators", "which simulated clients called: sources, languages, situations, and how Amira scored with each"),
    ("voice", "Voice rooms", "the audio side: rooms, turns spoken, failures, audio played"),
]


SOURCE_NAMES = {"jasmin_profile": "Jasmin stored profile", "jasmin_generated": "new Jasmin client", "scripted": "Voice Lab scripted client"}
FINDING_NAMES = {
    "advisor_repetition_loop": "repetition loop", "advisor_repetition_loop_extended": "long repetition loop",
    "same_reply_to_different_questions": "same reply to different questions", "provided_information_ignored": "answer ignored",
    "reply_language_mismatch": "reply in the wrong language", "offers_for_wrong_destination": "offers for the wrong destination",
    "customer_request_unanswered": "client left unanswered", "booking_request_unanswered": "booking request not handled",
    "turn_deadline_exceeded": "slow reply", "possible_unsupported_booking_guarantee": "possible booking guarantee",
}


def _readable(counts: dict[str, Any] | None, names: dict[str, str]) -> dict[str, Any]:
    return {names.get(key, str(key).replace("_", " ")): value for key, value in (counts or {}).items()}


def build_facts(batch: dict[str, Any], reviews: list[dict[str, Any]],
                call_reviews: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """The facts each section may use, taken only from the stored run data."""
    ev = batch.get("evaluation") or {}
    calls = batch.get("calls") or []
    quality, routing, voice = ev.get("quality") or {}, ev.get("routing") or {}, ev.get("voice") or {}
    # Readable names: the model would otherwise copy internal keys such as "jasmin_generated".
    by_source = {SOURCE_NAMES.get(source, source): {"calls": v.get("calls"), "mean_quality": v.get("mean_score")}
                 for source, v in (ev.get("clients") or {}).items()}
    worst = sorted((c for c in calls if isinstance((c.get("quality") or {}).get("score"), (int, float))),
                   key=lambda c: c["quality"]["score"])[:3]
    ai_checked = []
    for review in reviews:
        result = (review.get("quality_review") or {}).get("result") or {}
        weaknesses = ((result.get("checked") or {}).get("verification") or {}).get("weaknesses") or []
        kept = [w for w in weaknesses if w.get("status") != "rejected"]
        if result.get("status") == "free_quality_checked":
            ai_checked.append({
                "client": (review.get("client") or {}).get("name") or (review.get("client") or {}).get("persona"),
                "score": (review.get("quality") or {}).get("score"),
                "issues": [{"title": (w.get("weakness") or {}).get("title"), "suggestion": (w.get("weakness") or {}).get("recommendation")} for w in kept][:3],
            })
    run = {"calls": ev.get("calls"), "finished": ev.get("finished"), "completed": (ev.get("outcomes") or {}).get("completed"),
           "failed": (ev.get("outcomes") or {}).get("failed"), "messages": ev.get("messages")}
    return {
        "overview": {**run, "mean_quality": quality.get("mean_score"), "lowest_quality": quality.get("min_score"),
                     "calls_below_80": quality.get("below_80"), "routing_decisions": routing.get("decisions"),
                     "calls_escalated_to_human": routing.get("calls_escalated"), "voice_turns_spoken": voice.get("turns_spoken"),
                     "voice_failures": voice.get("turn_failures"), "most_frequent_findings": _readable(quality.get("top_findings"), FINDING_NAMES)},
        "amira": {**run, "calls_with_repetition_loop": (ev.get("advisor") or {}).get("calls_with_repeat_loop"),
                  "longest_repetition": (ev.get("advisor") or {}).get("longest_repeat"), "mean_quality": quality.get("mean_score"),
                  "quality_by_client_source": by_source,
                  "lowest_scored_calls": [{"client": (c.get("client") or {}).get("name") or (c.get("client") or {}).get("persona"),
                                           "score": c["quality"]["score"], "findings": [FINDING_NAMES.get(code, code) for code in c["quality"].get("codes") or []]} for c in worst]},
        "quality": {"calls_reviewed": quality.get("reviewed"), "mean_score": quality.get("mean_score"), "lowest_score": quality.get("min_score"),
                    "calls_below_80": quality.get("below_80"), "needing_human_review": quality.get("human_review_required"),
                    "ai_check_completed": quality.get("llm_checked"), "most_frequent_findings": _readable(quality.get("top_findings"), FINDING_NAMES),
                    "ai_checked_calls": ai_checked,
                    **_call_review_facts(call_reviews)},
        "workforce": {"calls": ev.get("calls"),
                      "note": "a routing decision is made for every client turn, so there are many more decisions than calls; "
                              "say 'decisions', never 'calls', for these counts",
                      "routing_decisions": routing.get("decisions"), "decisions_by_queue": _readable(routing.get("by_queue"), {
                          "ai": "Amira (AI)", "customer_advisors": "human adviser", "reservations": "reservations",
                          "waiting_human": "waiting for an adviser"}),
                      "calls_escalated_to_human": routing.get("calls_escalated"), "turns_waiting_for_an_adviser": routing.get("waiting_for_human"),
                      "human_advisers": (batch.get("workforce") or {}).get("human_capacity")},
        "clients": {"calls_by_source": by_source, "languages": (ev.get("client_details") or {}).get("languages"),
                    "situations": _readable({k: v for k, v in ((ev.get("client_details") or {}).get("scenarios") or {}).items() if k != "—"}, {}),
                    "clients": dict(Counter((c.get("client") or {}).get("name") or (c.get("client") or {}).get("persona") or "client" for c in calls))},
        "voice": {"rooms": voice.get("rooms"), "turns_spoken": voice.get("turns_spoken"), "audio_failures": voice.get("turn_failures"),
                  "audio_minutes": voice.get("audio_minutes"), "calls": ev.get("calls")},
    }


def _call_review_facts(call_reviews: list[dict[str, Any]] | None) -> dict[str, Any]:
    if call_reviews is None:
        return {"note": "the AI check needs 10 GB of free RAM; calls without it have the rule-based score only"}
    done = [r for r in call_reviews if r.get("checked_by")]
    issues = Counter(str(w.get("title")) for r in done for w in r.get("weaknesses") or [] if w.get("title"))
    return {
        "note": "every call got an AI reading: the analyst's local model where RAM allowed, an API model for the others",
        "ai_review_by": dict(Counter(r["checked_by"] for r in done)),
        "calls_without_ai_review": len(call_reviews) - len(done),
        "ai_reviews": [{"client": r.get("client"), "rule_score": r.get("rule_score"), "verdict": r.get("verdict"),
                        "weaknesses": [w.get("title") for w in r.get("weaknesses") or []]} for r in done],
        "most_frequent_ai_issues": dict(issues.most_common(5)),
    }


def _numbers(text: str) -> set[str]:
    """Numbers in canonical form, so "81.70", "81,7" and "81.7" compare equal and 10 stays 10."""
    return {f"{float(n.replace(',', '.')):g}" for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def _invented_numbers(text: str, facts: dict[str, Any]) -> set[str]:
    allowed = _numbers(json.dumps(facts, ensure_ascii=False)) | {"80", "10"}  # thresholds named in the facts
    return {n for n in _numbers(text) if n not in allowed}


async def write_section(key: str, title: str, focus: str, facts: dict[str, Any],
                        ask: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    system = (
        "You write one section of a short test report on an AI call-centre simulation (Ozone Voyages, a Tunisian "
        "hotel booking agency), for a teacher's demo. Use ONLY the facts given; copy every number exactly as written "
        "there and never compute or invent one. Plain, confident English for a non-technical reader.\n"
        "Write 4 short labelled parts, each one or two sentences: **What it did**, **What went well**, "
        "**Problems seen**, **Recommendation**. No heading, no preamble, at most 170 words."
    )
    user = f"Section: {title} — {focus}.\nFacts (JSON):\n{json.dumps(facts, ensure_ascii=False)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result = await ask(messages)
    text = str(result.get("reply") or "").strip()
    invented = _invented_numbers(text, facts)
    if text and invented:
        retry = messages + [{"role": "assistant", "content": text},
                            {"role": "user", "content": f"These numbers are not in the facts: {', '.join(sorted(invented))}. "
                                                        "Rewrite the section using only numbers from the facts."}]
        second = await ask(retry)
        if str(second.get("reply") or "").strip():
            result, text = second, str(second.get("reply")).strip()
            invented = _invented_numbers(text, facts)
    return {"key": key, "title": title, "text": text, "provider": result.get("provider"), "model": result.get("model"),
            "warning": f"numbers not found in the run data: {', '.join(sorted(invented))}" if invented else None}


# ---------------------------------------------------------------------------
# Per-call AI quality review. The Quality Analyst's own LLM step runs on the local
# model and is skipped when RAM is short; for those calls the same reading is done
# here by an API model, from the stored transcript and the analyst's rule findings.

REVIEW_SYSTEM = (
    "You are a call-centre quality analyst for Ozone Voyages, a Tunisian hotel booking agency. Its AI advisor, Amira, "
    "answers clients by chat or phone: she finds hotel offers, answers policy questions from the agency FAQ, and hands "
    "bookings, payments and complaints to a human conseiller (payment is never taken in the conversation). "
    "Read the transcript and judge Amira only. Answer with JSON only, no prose around it:\n"
    '{"verdict": "one or two sentences", "strengths": ["..."], '
    '"weaknesses": [{"title": "short", "turn": <message number>, "quote": "exact words copied from that message", '
    '"severity": "minor|major|critical", "recommendation": "what Amira should have done"}]}\n'
    "At most 3 strengths and 4 weaknesses. Every quote must be copied exactly from the transcript. "
    "Write in English whatever the conversation language."
)


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text or "").lower())).strip()


def _parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _local_review(review: dict[str, Any]) -> dict[str, Any]:
    """The analyst's own local-model reading, in the same shape as an API review."""
    checked = ((review.get("quality_review") or {}).get("result") or {}).get("checked") or {}
    proposal = checked.get("proposal") or {}
    weaknesses = (checked.get("verification") or {}).get("weaknesses") or []
    kept = [w.get("weakness") or {} for w in weaknesses if w.get("status") != "rejected"]
    return {
        "verdict": str(proposal.get("handling") or ""),
        "strengths": [],
        "weaknesses": [{"title": w.get("title"), "quote": w.get("evidence") or w.get("quote"), "severity": w.get("severity"),
                        "recommendation": w.get("recommendation")} for w in kept][:4],
    }


async def review_call(review: dict[str, Any], transcript: list[dict[str, Any]],
                      ask: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    client = review.get("client") or {}
    quality = review.get("quality") or {}
    base = {"session_id": str(review.get("id")), "call_id": review.get("call_id"), "language": review.get("language"),
            "client": client.get("name") or client.get("persona") or "client", "scenario": client.get("scenario"),
            "rule_score": quality.get("score"), "rule_findings": [FINDING_NAMES.get(c, c) for c in quality.get("codes") or []]}
    if quality.get("llm_checked"):
        return {**base, "checked_by": "local model (Quality Analyst)", "model": "local", **_local_review(review)}
    lines = [f"{i}. {'Amira' if m.get('role') == 'advisor' else 'Client'}: {m.get('content')}"
             for i, m in enumerate(transcript, 1) if m.get("role") in ("advisor", "client")]
    user = (f"Client situation: {client.get('scenario') or 'unknown'}; language: {review.get('language')}.\n"
            f"Rule-based score: {quality.get('score')}/100; rule findings: {', '.join(base['rule_findings']) or 'none'}.\n"
            "Transcript:\n" + "\n".join(lines))
    messages = [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}]
    result = await ask(messages)
    data = _parse_json(str(result.get("reply") or ""))
    if data is None:  # usually cut off at the gateway's token limit: ask once for a shorter answer
        result = await ask(messages + [{"role": "user", "content": "Your answer was cut off. Reply again with the same "
                                                                 "JSON but much shorter: 2 weaknesses, quotes of 8 words at most."}])
        data = _parse_json(str(result.get("reply") or ""))
    if data is None:
        return {**base, "checked_by": None, "error": "the model did not return a readable review"}
    # Keep only weaknesses whose quote is really in the transcript: no invented evidence.
    said = _plain(" ".join(str(m.get("content") or "") for m in transcript))
    weaknesses, dropped = [], 0
    for w in data.get("weaknesses") or []:
        if isinstance(w, dict) and _plain(w.get("quote")) and _plain(w.get("quote")) in said:
            weaknesses.append({k: w.get(k) for k in ("title", "turn", "quote", "severity", "recommendation")})
        else:
            dropped += 1
    return {**base, "checked_by": "API model", "provider": result.get("provider"), "model": result.get("model"),
            "verdict": str(data.get("verdict") or ""), "strengths": [str(s) for s in data.get("strengths") or []][:3],
            "weaknesses": weaknesses[:4], "unverified_dropped": dropped}


# ---------------------------------------------------------------------------
# Structured AI quality report of one call (button on the Quality Analyst page).
# Two short calls, run together: the gateway caps each answer at 700 tokens.

CRITERIA = [
    ("understanding", "Understood the request"),
    ("accuracy", "Accurate information (offers, prices, FAQ)"),
    ("policy", "Followed policy (no payment, no invented guarantees)"),
    ("pushback", "Handled pushback without repeating"),
    ("tone", "Language and tone"),
    ("handover", "Right handover to a human"),
]
OUTCOMES = {"resolved": "Resolved", "handed_to_conseiller": "Handed to a conseiller",
            "unresolved": "Left unresolved", "client_left": "Client left"}

_CONTEXT = (
    "You are a senior call-centre quality analyst for Ozone Voyages, a Tunisian hotel booking agency. Its AI advisor, "
    "Amira, finds hotel offers, answers policy questions from the agency FAQ, and hands bookings, payments and "
    "complaints to a human conseiller (payment is never taken in the conversation). Judge Amira only, strictly and "
    "fairly. Write in English whatever the conversation language. Answer with JSON only, nothing around it.\n"
)
SCORES_SYSTEM = _CONTEXT + (
    "Give the call outcome and a score from 1 (very poor) to 5 (excellent) for each criterion:\n"
    + "\n".join(f"- {key}: {label}" for key, label in CRITERIA) + "\n"
    "Use 5 for handover or pushback only if it was handled well or genuinely not needed.\n"
    '{"outcome": "resolved|handed_to_conseiller|unresolved|client_left", "outcome_note": "max 20 words", '
    '"criteria": {"<key>": {"score": 1-5, "reason": "max 20 words"}, ...all six keys}, '
    '"verdict": "two sentences", "human_review": true|false}'
)
EVIDENCE_SYSTEM = _CONTEXT + (
    "List what Amira did well and what went wrong, each proven by a quote. A quote is a fragment of at most 12 words "
    "copied exactly from the numbered message you cite. Only keep a problem if the quote itself shows it; explain how.\n"
    '{"strengths": [{"point": "max 15 words", "turn": <message number>, "quote": "..."}], '
    '"problems": [{"title": "max 8 words", "severity": "minor|major|critical", "turn": <message number>, "quote": "...", '
    '"why": "how the quote shows the problem, max 20 words", "better_reply": "what Amira should have said, max 25 words"}], '
    '"improvement": "the one change that would most improve Amira, max 25 words"}\n'
    "At most 2 strengths and 3 problems."
)


def _place_quote(quote: Any, turn: Any, texts: list[str]) -> int | None:
    """The message number that really contains the quote (the cited one first), or None."""
    q = _plain(quote)
    if len(q) < 4:
        return None
    if isinstance(turn, int) and 1 <= turn <= len(texts) and q in _plain(texts[turn - 1]):
        return turn
    return next((i for i, t in enumerate(texts, 1) if q in _plain(t)), None)


async def call_quality_report(session: dict[str, Any], quality: dict[str, Any], transcript: list[dict[str, Any]],
                              ask: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    turns = [m for m in transcript if m.get("role") in ("advisor", "client")]
    texts = [str(m.get("content") or "") for m in turns]
    client = session.get("client") or {}
    lines = [f"{i}. {'Amira' if m.get('role') == 'advisor' else 'Client'}: {m.get('content')}" for i, m in enumerate(turns, 1)]
    user = (f"Client situation: {client.get('scenario') or 'unknown'}; language: {session.get('language')}; "
            f"rule findings: {', '.join(FINDING_NAMES.get(c, c) for c in quality.get('codes') or []) or 'none'}.\n"
            "Transcript:\n" + "\n".join(lines))

    async def one(system: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        result = await ask(messages)
        data = _parse_json(str(result.get("reply") or ""))
        if data is None:
            result = await ask(messages + [{"role": "user", "content": "Your answer was cut off or unreadable. "
                                                                     "Reply again with the same JSON, shorter."}])
            data = _parse_json(str(result.get("reply") or ""))
        return data, result

    (scores, r1), (evidence, r2) = await asyncio.gather(one(SCORES_SYSTEM), one(EVIDENCE_SYSTEM))
    if scores is None and evidence is None:
        raise ValueError("the model did not return a readable report")
    scores, evidence = scores or {}, evidence or {}

    criteria = []
    for key, label in CRITERIA:
        item = (scores.get("criteria") or {}).get(key) or {}
        try:
            score = max(1, min(5, int(item.get("score"))))
        except (TypeError, ValueError):
            score = None
        criteria.append({"key": key, "label": label, "score": score, "reason": str(item.get("reason") or "")})
    given = [c["score"] for c in criteria if c["score"]]

    # Evidence is kept only if its quote is really in the conversation (message number corrected if needed).
    strengths, problems, dropped = [], [], 0
    for s in evidence.get("strengths") or []:
        turn = _place_quote(s.get("quote"), s.get("turn"), texts) if isinstance(s, dict) else None
        if turn:
            strengths.append({"point": s.get("point"), "turn": turn, "quote": s.get("quote")})
        else:
            dropped += 1
    for p in evidence.get("problems") or []:
        turn = _place_quote(p.get("quote"), p.get("turn"), texts) if isinstance(p, dict) else None
        if turn and str(p.get("why") or "").strip():
            problems.append({k: p.get(k) for k in ("title", "severity", "quote", "why", "better_reply")} | {"turn": turn})
        else:
            dropped += 1

    used = r1 if r1.get("model") else r2
    return {
        "outcome": scores.get("outcome") if scores.get("outcome") in OUTCOMES else None,
        "outcome_label": OUTCOMES.get(scores.get("outcome"), "Unknown"),
        "outcome_note": str(scores.get("outcome_note") or ""),
        "criteria": criteria,
        # Computed here from the criteria, never written by the model.
        "ai_score": round(sum(given) / len(given) * 20) if given else None,
        "rule_score": quality.get("score"),
        "verdict": str(scores.get("verdict") or ""),
        "human_review": bool(scores.get("human_review")),
        "strengths": strengths[:2], "problems": problems[:3],
        "improvement": str(evidence.get("improvement") or ""),
        "unproven_dropped": dropped,
        "provider": used.get("provider"), "model": used.get("model"),
    }
