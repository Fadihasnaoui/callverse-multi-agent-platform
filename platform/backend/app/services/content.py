"""Phase 1 - the content engine. Generates a caption/hashtags/video script for
one piece of content, grounded only in facts it's given (same truthfulness
rule as the audit and as the Ozone bot's own system prompt), reads the
business's latest learning_insights before writing anything (the actual
feedback loop that makes this an agent, not a template filler), and appends
a ref-coded m.me call-to-action so every DM that results can be traced back
to this exact piece of content with zero ambiguity - no comment/likes
permission needed for that attribution, see Phase 6."""
import json
import re
from types import SimpleNamespace

from app import config, db
from app.ai import guards, prompts, run as ai_run, AIError
from app.ai.rules import StyleRules, guard_for
from app.ai.schemas import ContentOut
from app.services import memory, strategist


def _detect_content_category(caption: str, topic: str) -> str:
    text = f"{caption} {topic}"
    patterns = {
        "promotion": r"(promo|offre|réduction|discount|prix|%\s*off|remise|تخفيض|عرض)",
        "destination": r"(istanbul|antalya|tunisie|bodrum|sharm|marrakech|dubai|maldives|zanzibar|"
                       r"bali|paris|rome|barcelone|hurghada|destination|voyage|سفر|رحلة)",
        "hotel": r"(hotel|hôtel|resort|spa|palace|all\s*inclusive|séjour|إقامة|فندق)",
        "engagement": r"(comment|avis|partagez|dites.nous|which|choisi|vote|sondage|\?\s*$)",
        "event": r"(soirée|event|fête|gala|concert|célébr|anniversaire|حفل|مهرجان)",
        "testimonial": r"(témoignage|client|review|merci|شكر|تجربة)",
        "brand": r"(ozone|notre\s*équipe|notre\s*agence|bienvenue|about\s*us|qui\s*sommes)",
    }
    for cat, pat in patterns.items():
        if re.search(pat, text, re.I):
            return cat
    return "general"


def _complete_hashtags(existing: list[str], topic: str, business_name: str, platform: str, limit: int) -> list[str]:
    if limit == 0:
        return []
    proper_places = re.findall(r"(?<!^)\b[A-ZÀ-Ý][A-Za-zÀ-ÿ-]{3,}\b", topic)
    target = min(limit, 8)
    candidates = existing + [re.sub(r"\W+", "", business_name), *proper_places,
                             "Tunisie", "VoyageTunisie", "FamilleTunisienne", "VacancesTunisie",
                             "DecouvrirLaTunisie", platform.title()]
    completed = []
    seen = set()
    for candidate in candidates:
        tag = re.sub(r"\s+", "", candidate).lstrip("#")
        if tag and tag.casefold() not in seen:
            completed.append(tag)
            seen.add(tag.casefold())
        if len(completed) >= target:
            break
    return completed

def _latest_learning_insight(conn, business_id: str) -> dict | None:
    row = conn.execute(
        "SELECT insights FROM learning_insights WHERE business_id = %s "
        "AND source <> 'simulation_baseline' ORDER BY created_at DESC LIMIT 1",
        (business_id,),
    ).fetchone()
    return row[0] if row else None


def _page_id(conn, business_id: str, platform: str) -> str | None:
    row = conn.execute(
        "SELECT meta FROM business_credentials WHERE business_id = %s AND platform = %s",
        (business_id, platform),
    ).fetchone()
    return (row[0] or {}).get("page_id") if row else None


def generate_content(business_id: str, platform: str, topic: str, content_type: str = "static_post") -> dict:
    if platform not in ('facebook','instagram') or content_type not in ('static_post','reel','avatar_clip'):
        raise ValueError('unsupported platform or content type')
    with db.get_conn() as conn:
        business = conn.execute(
            "SELECT brand_voice, style_rules, name FROM businesses WHERE id = %s", (business_id,)
        ).fetchone()
        if not business:
            raise ValueError("business not found")
        brand_voice = business[0] or ""
        rules = StyleRules.model_validate(business[1] or {})
        business_name = business[2]

        learning = _latest_learning_insight(conn, business_id)
        page_id = _page_id(conn, business_id, platform)

        # Insert first (empty caption) to get a real id - the CTA needs this
        # id as its ref code, so the row has to exist before the final
        # caption does. features/status get filled in on the update below.
        draft_id = conn.execute(
            "INSERT INTO content_items (business_id, type, platform, status) "
            "VALUES (%s, %s, %s, 'pending') RETURNING id",
            (business_id, content_type, platform),
        ).fetchone()[0]
        conn.commit()

    try:
        strategic_brief = strategist.build_strategic_brief(business_id, context="content")
    except Exception:
        strategic_brief = ""

    try:
        content_directives = strategist.build_content_directives(business_id)
    except Exception:
        content_directives = ""

    learning_guidance = (
        f"What has actually worked for this business before (from a prior audit or "
        f"performance analysis - use this to inform tone/format choices, do not "
        f"state these numbers in the copy itself): {learning}"
        if learning
        else "No prior performance data yet for this business - this may be the first piece of content generated."
    )

    try:
        result = ai_run(
            "content", prompts.CONTENT.system,
            f"Brand voice: {brand_voice or 'friendly, direct, Tunisian'}\n"
            f"Platform: {platform} — write platform-native content (not cross-posted generic)\n"
            f"Content type: {content_type}\n"
            f"Topic/offer to write about: {topic}\n\n"
            f"{'--- STRATEGIC BRIEF (use this to inform your decisions, never quote these numbers in the copy) ---' + chr(10) + strategic_brief + chr(10) if strategic_brief else ''}"
            f"{content_directives + chr(10) if content_directives else ''}"
            f"{learning_guidance}\n"
            f"{rules.prompt_block()}"
            f"{memory.recall_block(business_id, 'content', topic)}",
            ContentOut, model=config.CONTENT_MODEL, version=prompts.CONTENT.version,
            guard=guards.all_of(
                guards.no_invented_numbers(topic, brand_voice, fields=("caption","video_script")),
                lambda output: guards.no_invented_numbers(topic,brand_voice,fields=('text',))(SimpleNamespace(text=' '.join(output.overlay_lines))),
                guards.no_call_to_action(), guards.natural_marketing_copy(), guard_for(rules),
            ),
        )
    except Exception:
        # Never leave an empty, approvable draft behind when the writer failed.
        with db.get_conn() as conn:
            conn.execute("DELETE FROM content_items WHERE id = %s", (str(draft_id),))
            conn.commit()
        raise

    caption = result.caption
    result.hashtags = _complete_hashtags(
        result.hashtags, topic, business_name, platform,
        limit=rules.max_hashtags if rules.max_hashtags is not None else 10,
    )
    cta = f"\n\nEnvoyez-nous un message : m.me/{page_id}?ref={draft_id}" if page_id else ""
    final_caption = caption + cta
    problems = rules.violations(caption,result.hashtags)
    if rules.max_caption_chars and len(final_caption)>rules.max_caption_chars:
        problems.append('Final caption including its message link exceeds the configured limit; shorten the brief or increase the limit.')
    if platform=='instagram' and len(final_caption+' '.join(result.hashtags))+len(result.hashtags)*2>2200:
        problems.append('Final Instagram caption and hashtags exceed 2200 characters')
    if problems:
        with db.get_conn() as conn:
            conn.execute('DELETE FROM content_items WHERE id=%s',(str(draft_id),)); conn.commit()
        raise AIError('; '.join(problems))

    features = {
        "topic": topic,
        "format": content_type,
        "visual_brief": result.visual_brief,
        "overlay_lines": result.overlay_lines,
    }

    detected_category = _detect_content_category(caption, topic)

    with db.get_conn() as conn:
        conn.execute(
            "UPDATE content_items SET caption = %s, hashtags = %s, video_script = %s, features = %s, "
            "content_category = %s WHERE id = %s",
            (
                final_caption,
                result.hashtags,
                result.video_script,
                json.dumps(features),
                detected_category,
                str(draft_id),
            ),
        )
        conn.commit()

    try:
        from app.services.notifications import notify
        notify(business_id, "content_generated",
               f"New {content_type} draft ({platform})",
               f"{topic[:150]}",
               {"content_id": str(draft_id), "platform": platform, "type": content_type})
    except Exception:
        pass

    return {
        "id": str(draft_id),
        "platform": platform,
        "caption": final_caption,
        "hashtags": result.hashtags,
        "video_script": result.video_script,
        "visual_brief": result.visual_brief,
        "overlay_lines": result.overlay_lines,
        "status": "pending",
        "ref_code": str(draft_id) if page_id else None,
    }
