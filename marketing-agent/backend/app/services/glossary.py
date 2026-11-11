"""Derja glossary - how dictation transcripts get better every time.

Pipeline for a dictated line (see avatar.transcribe_dictation):
  1. speech-to-text gives the RAW transcript;
  2. apply(): word pairs the owner has already confirmed (seen >= AUTO_AFTER
     times) are swapped in deterministically - same fix, same result, always;
  3. proofread(): one LLM pass fixes the remaining misspellings, shown the
     glossary as hints, told to stay in Derja and change nothing else;
  4. the owner reviews and approves;
  5. learn(): raw transcript vs. the approved text is diffed word by word, and
     every changed word becomes (or strengthens) a glossary pair. Accepting a
     machine fix without touching it counts as a confirmation too.
A pair becomes automatic only once it has been confirmed twice, so a single
mistaken edit cannot rewrite future transcripts."""
import difflib
import logging
import re

from app import config, db
from app.ai import guards, prompts, run as ai_run
from app.ai.schemas import ProofreadOut

log = logging.getLogger("glossary")

AUTO_AFTER = 2
HINTS_FOR_LLM = 60
_WORD = re.compile(r"[\w']+", re.UNICODE)

def _tokens(text: str) -> list[str]:
    return _WORD.findall(text or "")


def apply(business_id: str, text: str) -> tuple[str, list[tuple[str, str]]]:
    """Swaps confirmed pairs. Returns (new_text, [(heard, meant), ...] that fired)."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT heard, meant FROM derja_glossary WHERE business_id = %s AND times_seen >= %s "
            "ORDER BY times_seen ASC",  # strongest pair wins when one 'heard' has several 'meant'
            (business_id, AUTO_AFTER),
        ).fetchall()
    mapping = {h: m for h, m in rows}
    fired: list[tuple[str, str]] = []

    def swap(match: re.Match) -> str:
        word = match.group(0)
        meant = mapping.get(word)
        if meant and meant != word:
            fired.append((word, meant))
            return meant
        return word

    return _WORD.sub(swap, text), fired


def proofread(business_id: str, text: str) -> str:
    """Final LLM pass. Any failure or suspicious output falls back to the input."""
    with db.get_conn() as conn:
        hints = conn.execute(
            "SELECT heard, meant FROM derja_glossary WHERE business_id = %s "
            "ORDER BY times_seen DESC, last_seen DESC LIMIT %s",
            (business_id, HINTS_FOR_LLM),
        ).fetchall()
    hint_block = "\n".join(f"- {h} -> {m}" for h, m in hints) or "(none yet)"
    try:
        # The length guard lives in the AI layer: an over-eager rewrite is sent back for one repair, then rejected.
        result = ai_run(
            "proofread", prompts.PROOFREAD.system,
            f"Known corrections for this speaker:\n{hint_block}\n\nTranscript:\n{text}",
            ProofreadOut, timeout=60, model=config.PROOFREAD_MODEL, version=prompts.PROOFREAD.version,
            guard=guards.length_close_to(text),
        )
        return result.text.strip()
    except Exception:  # noqa: BLE001 - proofreading is a bonus; never block the dictation
        log.exception("derja proofread failed")
        return text


def learn(business_id: str, raw: str, final: str) -> int:
    """Diffs raw speech-to-text vs. the approved transcript; upserts word pairs.
    Only 1:1 word replacements are learned - anything else is too ambiguous."""
    a, b = _tokens(raw), _tokens(final)
    pairs: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "replace" and (i2 - i1) == (j2 - j1):
            for heard, meant in zip(a[i1:i2], b[j1:j2]):
                if heard != meant and len(heard) > 1 and not heard.isdigit() and not meant.isdigit():
                    pairs.append((heard, meant))
    if not pairs:
        return 0
    with db.get_conn() as conn:
        for heard, meant in pairs:
            conn.execute(
                "INSERT INTO derja_glossary (business_id, heard, meant) VALUES (%s,%s,%s) "
                "ON CONFLICT (business_id, heard, meant) DO UPDATE "
                "SET times_seen = derja_glossary.times_seen + 1, last_seen = now()",
                (business_id, heard, meant),
            )
        conn.commit()
    return len(pairs)


def list_pairs(business_id: str) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, heard, meant, times_seen, last_seen FROM derja_glossary WHERE business_id = %s "
            "ORDER BY times_seen DESC, last_seen DESC LIMIT 500",
            (business_id,),
        ).fetchall()
    return [{"id": str(r[0]), "heard": r[1], "meant": r[2], "times_seen": r[3],
             "automatic": r[3] >= AUTO_AFTER, "last_seen": r[4].isoformat()} for r in rows]


def add_pair(business_id: str, heard: str, meant: str) -> bool:
    heard, meant = (heard or "").strip(), (meant or "").strip()
    if not heard or not meant or heard == meant:
        return False
    with db.get_conn() as conn:  # owner-typed pairs are trusted: start at the automatic threshold
        conn.execute(
            "INSERT INTO derja_glossary (business_id, heard, meant, times_seen) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (business_id, heard, meant) DO UPDATE SET times_seen = GREATEST(derja_glossary.times_seen, %s)",
            (business_id, heard, meant, AUTO_AFTER, AUTO_AFTER),
        )
        conn.commit()
    return True


def delete_pair(business_id: str, pair_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "DELETE FROM derja_glossary WHERE id = %s AND business_id = %s RETURNING id", (pair_id, business_id)
        ).fetchone()
        conn.commit()
    return bool(row)
