"""Scoped agent memory. Every memory belongs to ONE scope, and each task reads
only the scopes it is wired to, so lead-scoring lessons never leak into
caption writing and vice versa.

Scopes
  global      owner-wide facts every task may use (audience, brand rules)
  content     what makes a post right/wrong for this owner (rejections, caption edits, weekly ideas)
  scheduling  timing habits (calendar choices, preferred days/hours)
  voice       spoken-language lessons (Derja corrections, avatar delivery)
  leads       what is / is not a real lead (owner corrections, which messages became sales)
  learning    conclusions of the weekly learning job + owner notes about the data

The mechanic, in order, for a task asking for context (recall_block):
  1. global      - always, capped.
  2. own scopes  - the scopes this task owns (TASK_WIRING), newest + pinned first.
  3. linked      - other scopes the task may *borrow from*, but only entries
                   that mention a keyword of the current job (its topic). No
                   keyword match, nothing is borrowed - that is how scopes
                   connect "when needed" without flooding the prompt.
Whole block is character-capped. Weekly compaction (compact_all) folds old
unpinned entries of a busy scope into one summary so months of use stay small.
Owner-pinned notes are never summarised or dropped."""
import logging
import re

from app import config, db
from app.ai import prompts, run as ai_run
from app.ai.schemas import CompactOut

log = logging.getLogger("memory")

SCOPES = ("global", "content", "scheduling", "voice", "leads", "learning")
KINDS = ("note", "preference", "feedback", "language", "plan", "schedule", "insight", "summary")

# task -> (scopes it owns, scopes it may borrow from on a keyword match)
TASK_WIRING = {
    "content": (("content",), ("learning",)),
    "weekly": (("content", "scheduling"), ("learning",)),
    "scheduling": (("scheduling",), ("content", "learning")),
    "voice": (("voice",), ("content",)),
    "leads": (("leads",), ()),
    "learning": (("learning",), ("leads", "content")),
}

GLOBAL_CAP = 8
OWN_CAP = 12
LINKED_CAP = 4
MAX_PROMPT_CHARS = 3500
COMPACT_THRESHOLD = 40   # unpinned entries in one scope before it is summarised
COMPACT_KEEP_NEWEST = 15
VERBATIM_SCOPES = {"voice"}  # exact word corrections: never paraphrased by an LLM, only trimmed
VERBATIM_CAP = 200
_SNIP = 300
_STOP = {"avec", "pour", "dans", "cette", "votre", "nous", "vous", "plus", "from", "that", "this", "with", "have", "your"}


def _snip(text: str | None, n: int = _SNIP) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def remember(business_id: str, scope: str, kind: str, content: str, source: str | None = None, pinned: bool = False) -> str | None:
    """Stores one memory. Never raises: memory is a helper, and a failure here
    must not break the task (approve, schedule, ...) that triggered it."""
    content = (content or "").strip()
    if scope not in SCOPES or kind not in KINDS or not content:
        return None
    try:
        with db.get_conn() as conn:
            dup = conn.execute(
                "SELECT id FROM agent_memory WHERE business_id = %s AND scope = %s AND content = %s",
                (business_id, scope, content),
            ).fetchone()
            if dup:
                return str(dup[0])
            row = conn.execute(
                "INSERT INTO agent_memory (business_id, scope, kind, content, source, pinned) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (business_id, scope, kind, content, source, pinned),
            ).fetchone()
            conn.commit()
        return str(row[0])
    except Exception:
        log.exception("could not save memory")
        return None


def _row(r) -> dict:
    return {"id": str(r[0]), "scope": r[1], "kind": r[2], "content": r[3], "source": r[4],
            "pinned": r[5], "created_at": r[6].isoformat()}


def list_memories(business_id: str, scope: str | None = None, limit: int = 300) -> list[dict]:
    query = "SELECT id, scope, kind, content, source, pinned, created_at FROM agent_memory WHERE business_id = %s"
    params: list = [business_id]
    if scope:
        query += " AND scope = %s"
        params.append(scope)
    with db.get_conn() as conn:
        rows = conn.execute(query + " ORDER BY pinned DESC, created_at DESC LIMIT %s", params + [limit]).fetchall()
    return [_row(r) for r in rows]


def delete_memory(business_id: str, memory_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "DELETE FROM agent_memory WHERE id = %s AND business_id = %s RETURNING id", (memory_id, business_id)
        ).fetchone()
        conn.commit()
    return bool(row)


def _keywords(context: str, limit: int = 8) -> list[str]:
    words = {w for w in re.findall(r"\w{4,}", (context or "").lower()) if w not in _STOP and not w.isdigit()}
    return sorted(words, key=len, reverse=True)[:limit]


def _fetch(conn, business_id: str, scopes, cap: int, keywords: list[str] | None = None) -> list[tuple]:
    query = (
        "SELECT scope, kind, content, pinned FROM agent_memory WHERE business_id = %s AND scope = ANY(%s)"
    )
    params: list = [business_id, list(scopes)]
    if keywords is not None:
        query += " AND content ILIKE ANY(%s)"
        params.append([f"%{k}%" for k in keywords])
    query += " ORDER BY pinned DESC, (kind = 'summary') DESC, created_at DESC LIMIT %s"
    return conn.execute(query, params + [cap]).fetchall()


def recall_block(business_id: str, task: str, context: str = "") -> str:
    """Prompt-ready text for one task, or '' when nothing relevant is remembered."""
    own, linked = TASK_WIRING[task]
    try:
        with db.get_conn() as conn:
            sections = [("global", _fetch(conn, business_id, ["global"], GLOBAL_CAP))]
            for scope in own:
                sections.append((scope, _fetch(conn, business_id, [scope], OWN_CAP)))
            keywords = _keywords(context)
            if keywords:
                for scope in linked:
                    sections.append((f"{scope}, borrowed - mentions this topic", _fetch(conn, business_id, [scope], LINKED_CAP, keywords)))
    except Exception:
        log.exception("could not read memory")
        return ""
    out, used = [], 0
    for title, rows in sections:
        if not rows:
            continue
        lines = [f"[{title}]"]
        for _scope, kind, content, pinned in rows:
            line = f"- ({kind}{', owner-pinned' if pinned else ''}) {content}"
            if used + len(line) > MAX_PROMPT_CHARS:
                break
            lines.append(line)
            used += len(line)
        if len(lines) > 1:
            out.append("\n".join(lines))
    if not out:
        return ""
    return (
        "\nWhat you remember about this business from past work with its owner. Entries marked owner-pinned "
        "or phrased as rules are MANDATORY: obey them over your own style defaults. Use the rest to match "
        "their style and avoid repeating mistakes. It is NOT a source of offers, prices or dates - never "
        "state a deal or figure from it unless the current request also gives it:\n"
        + "\n".join(out) + "\n"
    )


# --- capture helpers: called where the owner teaches the agent --------------------

def learn_from_rejection(business_id: str, platform: str, ctype: str, caption: str | None, notes: str | None):
    if notes and notes.strip():
        text = f"Owner rejected a {platform} {ctype} post. Reason: {_snip(notes)}. Post began: \"{_snip(caption, 120)}\""
    else:
        text = f"Owner rejected a {platform} {ctype} post (no reason given). It began: \"{_snip(caption, 120)}\""
    remember(business_id, "content", "feedback", text, source="rejection")


def learn_from_edit(business_id: str, before: str | None, after: str | None):
    if before and after and " ".join(before.split()) != " ".join(after.split()):
        remember(business_id, "content", "feedback",
                 f"Owner rewrote a caption. Agent wrote: \"{_snip(before)}\" -> owner changed it to: \"{_snip(after)}\"",
                 source="caption_edit")


def learn_from_transcript_fix(business_id: str, heard: str | None, fixed: str):
    if heard and " ".join(heard.split()) != " ".join(fixed.split()):
        remember(business_id, "voice", "language",
                 f"Speech-to-text heard \"{_snip(heard)}\" but the owner meant (Tunisian Derja) \"{_snip(fixed)}\"",
                 source="transcript_fix")


def learn_lead_judgement(business_id: str, message: str | None, is_lead: bool, scored: int | None):
    verdict = "a REAL lead" if is_lead else "NOT a lead"
    opposite = "NOT a lead" if is_lead else "a REAL lead"
    try:  # the owner changed their mind: the newest answer replaces the old one
        with db.get_conn() as conn:
            conn.execute(
                "DELETE FROM agent_memory WHERE business_id = %s AND scope = 'leads' AND source = 'lead_feedback' "
                "AND content = %s",
                (business_id, f"Message \"{_snip(message, 200)}\" is {opposite} according to the owner (agent had scored it {scored})."),
            )
            conn.commit()
    except Exception:
        log.exception("could not replace earlier lead judgement")
    remember(business_id, "leads", "feedback",
             f"Message \"{_snip(message, 200)}\" is {verdict} according to the owner (agent had scored it {scored}).",
             source="lead_feedback")


def learn_conversion(business_id: str, message: str | None, notes: str | None):
    if message:
        remember(business_id, "leads", "feedback",
                 f"This message became a paying customer: \"{_snip(message, 200)}\"" + (f" ({_snip(notes, 120)})" if notes else ""),
                 source="conversion")


# --- compaction: keeps a scope small after months of use --------------------------

def compact_scope(business_id: str, scope: str) -> bool:
    """Folds a busy scope into ONE summary entry. Previous summaries are merged
    in (never left to pile up), lines carry dates so recency can resolve
    contradictions, and 'no reason given' rejections - zero information - are
    dropped without being shown to the model."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, kind, content, created_at FROM agent_memory WHERE business_id = %s AND scope = %s "
            "AND pinned = false ORDER BY created_at DESC",
            (business_id, scope),
        ).fetchall()
    fresh = [r for r in rows if r[1] != "summary"]
    if scope in VERBATIM_SCOPES:
        stale = [str(r[0]) for r in fresh[VERBATIM_CAP:]]
        if stale:
            with db.get_conn() as conn:
                conn.execute("DELETE FROM agent_memory WHERE id = ANY(%s::uuid[])", (stale,))
                conn.commit()
        return bool(stale)
    if len(fresh) <= COMPACT_THRESHOLD:
        return False
    summaries = [r for r in rows if r[1] == "summary"]
    old = fresh[COMPACT_KEEP_NEWEST:]
    informative = [r for r in reversed(old) if "no reason given" not in r[2]]  # oldest first
    lines = [f"[previous summary] {r[2]}" for r in reversed(summaries)]
    lines += [f"[{r[3]:%Y-%m-%d}] {r[2]}" for r in informative]
    kept = [f"[{r[3]:%Y-%m-%d}] {r[2]}" for r in reversed(fresh[:COMPACT_KEEP_NEWEST]) if "no reason given" not in r[2]]
    if kept:
        lines += ["[recent - not being merged, these override older lines]"] + kept
    if not informative and not summaries:
        summary = ""
    else:
        try:
            result = ai_run(
                "compact", prompts.COMPACT.system, f"Scope: {scope}\n\n" + "\n".join(lines),
                CompactOut, timeout=90, model=config.COMPACTION_MODEL, version=prompts.COMPACT.version,
            )
        except Exception:  # noqa: BLE001 - leave the scope as it is; the next weekly run retries
            log.exception("memory compaction failed for %s/%s", business_id, scope)
            return False
        bullets = [b.strip().lstrip("- ").strip() for b in result.bullets if b.strip()]
        summary = "\n".join(f"- {x}" for x in bullets)
        if not summary:
            return False
    doomed = [str(r[0]) for r in old] + [str(r[0]) for r in summaries]
    with db.get_conn() as conn:
        conn.execute("DELETE FROM agent_memory WHERE id = ANY(%s::uuid[])", (doomed,))
        if summary:
            conn.execute(
                "INSERT INTO agent_memory (business_id, scope, kind, content, source) VALUES (%s,%s,'summary',%s,'compaction')",
                (business_id, scope, summary),
            )
        conn.commit()
    return True


@db.single_task('marketing-memory-compaction')
def compact_all() -> int:
    """Weekly scheduler entry point."""
    with db.get_conn() as conn:
        pairs = conn.execute(
            "SELECT business_id, scope FROM agent_memory WHERE pinned = false AND kind <> 'summary' "
            "GROUP BY business_id, scope HAVING count(*) > %s", (COMPACT_THRESHOLD,)
        ).fetchall()
    done = 0
    for business_id, scope in pairs:
        try:
            done += compact_scope(str(business_id), scope)
        except Exception:  # noqa: BLE001 - one scope failing must not stop the rest
            log.exception("memory compaction failed for %s/%s", business_id, scope)
    return done
