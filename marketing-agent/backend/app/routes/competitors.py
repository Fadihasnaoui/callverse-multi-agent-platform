"""Competitor intelligence endpoints."""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import competitors as comp_service

log = logging.getLogger("competitors.routes")

router = APIRouter(prefix="/businesses/{business_id}", tags=["competitors"])


class AddCompetitor(BaseModel):
    name: str
    page_url: str
    platform: str = "facebook"
    page_id: str | None = None
    followers: int | None = None
    talking_about: int | None = None


class SnapshotUpdate(BaseModel):
    followers: int | None = None
    talking_about: int | None = None
    posts_last_30d: int | None = None
    avg_engagement: float | None = None


@router.get("/competitors")
def list_competitors(business_id: str):
    return comp_service.list_competitors(business_id)


@router.post("/competitors")
def add_competitor(business_id: str, payload: AddCompetitor):
    result = comp_service.add_competitor(
        business_id, payload.name, payload.page_url,
        payload.platform, payload.page_id,
    )
    if payload.followers is not None:
        comp_service.record_snapshot(
            result["id"], followers=payload.followers,
            talking_about=payload.talking_about,
        )
    return result


@router.delete("/competitors/{competitor_id}")
def remove_competitor(business_id: str, competitor_id: str):
    if not comp_service.remove_competitor(business_id, competitor_id):
        raise HTTPException(status_code=404, detail="competitor not found")
    return {"deleted": True}


@router.post("/competitors/{competitor_id}/snapshot")
def update_snapshot(business_id: str, competitor_id: str, payload: SnapshotUpdate):
    snap_id = comp_service.record_snapshot(
        competitor_id, followers=payload.followers,
        talking_about=payload.talking_about,
        posts_last_30d=payload.posts_last_30d,
        avg_engagement=payload.avg_engagement,
    )
    return {"snapshot_id": snap_id}


class RefreshResult(BaseModel):
    competitor_id: str
    snippet: str = ""
    description: str = ""


class RefreshPayload(BaseModel):
    results: list[RefreshResult]


@router.post("/competitors/refresh")
def refresh_competitors(business_id: str, payload: RefreshPayload):
    """Accept pre-scraped search results and update all competitor snapshots.
    The frontend (or a manual curl) passes the TinyFish/search snippets here."""
    data = [r.model_dump() for r in payload.results]
    summary = comp_service.refresh_all_from_search_results(business_id, data)
    return {"status": "ok", **summary}


@router.post("/competitors/refresh-auto")
def refresh_competitors_auto(business_id: str):
    """Auto-refresh: fetch each competitor's page via requests, parse OG description,
    and store new snapshots. No external API key needed — uses public Facebook
    page metadata."""
    import requests as req

    competitors = comp_service.list_competitors(business_id)
    if not competitors:
        raise HTTPException(status_code=404, detail="no competitors to refresh")

    results = []
    for c in competitors:
        url = c["page_url"]
        try:
            resp = req.get(
                url, allow_redirects=True, timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MarketingAgent/1.0)"},
            )
            text = resp.text
            import re
            og_match = re.search(r'<meta\s+(?:property|name)="og:description"\s+content="([^"]*)"', text)
            desc_match = re.search(r'<meta\s+(?:property|name)="description"\s+content="([^"]*)"', text)
            og_desc = og_match.group(1) if og_match else ""
            meta_desc = desc_match.group(1) if desc_match else ""
            snippet = og_desc or meta_desc
            data = comp_service.refresh_competitor_from_snippet(c["id"], snippet, meta_desc)
            results.append({"competitor": c["name"], "status": "updated" if data else "no_data", "data": data})
        except Exception as e:
            log.warning("refresh failed for %s: %s", c["name"], e)
            results.append({"competitor": c["name"], "status": "error", "error": str(e)})

    updated = sum(1 for r in results if r["status"] == "updated")
    return {"status": "ok", "updated": updated, "total": len(competitors), "details": results}


@router.get("/competitive-position")
def competitive_position(business_id: str):
    result = comp_service.competitive_position(business_id)
    if not result:
        return {"status": "no_competitors", "detail": "add competitors first"}
    return {"status": "ok", **result}
