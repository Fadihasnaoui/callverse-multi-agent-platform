"""Meta Graph API client.

Covers:
- Token exchange: short-lived user token -> long-lived user token -> never-expiring page token
- Facebook: page posts, post insights, publishing (text/photo/video)
- Instagram: media listing, media insights
- Token health check
"""
import logging
import requests

from app import config

log = logging.getLogger(__name__)


class GraphAPIError(RuntimeError):
    def __init__(self, message, outcome_unknown=False):
        super().__init__(message)
        self.outcome_unknown=outcome_unknown


def _json(response) -> dict:
    try:
        return response.json()
    except ValueError:
        raise GraphAPIError(f"Meta returned a non-JSON reply (HTTP {response.status_code}): {response.text[:200]!r}",outcome_unknown=True)


def _get(path: str, access_token: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["access_token"] = access_token
    response = requests.get(f"{config.META_GRAPH_BASE_URL}/{path}", params=params, timeout=30)
    data = _json(response)
    if "error" in data:
        raise GraphAPIError(data["error"].get("message", str(data["error"])))
    return data


def _post(path: str, access_token: str, data: dict | None = None) -> dict:
    payload = dict(data or {})
    payload["access_token"] = access_token
    response = requests.post(f"{config.META_GRAPH_BASE_URL}/{path}", data=payload, timeout=60)
    body = _json(response)
    if "error" in body:
        raise GraphAPIError(body["error"].get("message", str(body["error"])))
    return body


# ---------------------------------------------------------------------------
# Token exchange — the goal is a never-expiring Page access token
# ---------------------------------------------------------------------------

def exchange_for_long_lived_token(short_lived_token: str, app_id: str, app_secret: str) -> str:
    """Exchange a short-lived user token (~1h) for a long-lived one (~60 days)."""
    response = requests.get(
        f"{config.META_GRAPH_BASE_URL}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": short_lived_token,
        },
        timeout=30,
    )
    data = _json(response)
    if "error" in data:
        raise GraphAPIError(f"Token exchange failed: {data['error'].get('message', str(data['error']))}")
    return data["access_token"]


def get_permanent_page_token(long_lived_user_token: str, page_id: str) -> str:
    """Get a never-expiring Page access token from a long-lived user token.
    Tries me/accounts first; falls back to direct page access if the app
    doesn't have pages_show_list but the user IS a page admin."""
    # Method 1: me/accounts (standard, works when pages_show_list is properly granted)
    data = _get("me/accounts", long_lived_user_token, {"fields": "id,name,access_token"})
    for page in data.get("data", []):
        if page["id"] == page_id:
            return page["access_token"]

    # Method 2: direct page access — some app configurations return an empty
    # me/accounts even when the user is a page admin. The page endpoint with
    # ?fields=access_token still works in that case.
    try:
        page_data = _get(page_id, long_lived_user_token, {"fields": "id,name,access_token"})
        if page_data.get("access_token"):
            log.info("Got page token via direct access (me/accounts was empty)")
            return page_data["access_token"]
    except GraphAPIError:
        pass

    available = [p["id"] for p in data.get("data", [])]
    raise GraphAPIError(
        f"Page {page_id} not found in user's managed pages. "
        f"Available page IDs: {available}. Make sure the user is a Page admin."
    )


def exchange_to_permanent_page_token(
    user_token: str, page_id: str, app_id: str = "", app_secret: str = ""
) -> dict:
    """Full pipeline: user token (any lifetime) -> permanent page token.
    Returns {access_token, token_type, page_name} or raises."""
    app_id = app_id or config.META_APP_ID
    app_secret = app_secret or config.META_APP_SECRET

    if not app_id or not app_secret:
        log.warning("META_APP_ID/META_APP_SECRET not set — storing token as-is (may expire)")
        return {"access_token": user_token, "token_type": "unknown", "exchanged": False}

    # Step 1: try exchanging to long-lived (works on short-lived tokens;
    # on an already-long-lived token Meta returns the same or a refreshed one)
    try:
        long_lived = exchange_for_long_lived_token(user_token, app_id, app_secret)
    except GraphAPIError:
        long_lived = user_token

    # Step 2: get the permanent page token
    page_token = get_permanent_page_token(long_lived, page_id)

    # Verify the page token
    debug = debug_token(page_token, app_id, app_secret)

    return {
        "access_token": page_token,
        "token_type": "permanent_page_token" if not debug.get("expires_at") else "long_lived_page_token",
        "exchanged": True,
        "expires_at": debug.get("expires_at"),
        "scopes": debug.get("scopes", []),
    }


def debug_token(token: str, app_id: str = "", app_secret: str = "") -> dict:
    """Inspect a token's validity, expiration, and scopes."""
    app_id = app_id or config.META_APP_ID
    app_secret = app_secret or config.META_APP_SECRET
    if not app_id or not app_secret:
        return {"valid": None, "reason": "META_APP_ID/META_APP_SECRET not set"}
    response = requests.get(
        f"{config.META_GRAPH_BASE_URL}/debug_token",
        params={"input_token": token, "access_token": f"{app_id}|{app_secret}"},
        timeout=15,
    )
    data = _json(response).get("data", {})
    return {
        "valid": data.get("is_valid"),
        "app_id": data.get("app_id"),
        "type": data.get("type"),
        "expires_at": data.get("expires_at") or None,  # 0 means never
        "scopes": data.get("scopes", []),
        "error": data.get("error"),
    }


# ---------------------------------------------------------------------------
# Facebook — posts and insights
# ---------------------------------------------------------------------------

def fetch_page_posts(page_id: str, access_token: str, limit: int = 100) -> list[dict]:
    """A Page's own post history with content, type, timestamp, and basic metrics."""
    posts: list[dict] = []
    # Start with basic fields that don't require pages_read_user_content;
    # try expanded fields (reactions/comments summaries) if available.
    basic_fields = "id,message,created_time,permalink_url,attachments{media_type},shares"
    full_fields = basic_fields + ",comments.summary(true),reactions.summary(true)"
    # Meta availability differs by app review, token type and Graph version.
    # ``feed`` remains a useful fallback where the separate ``posts`` edge is
    # unavailable. Keep field fallback independent of edge fallback.
    data = None
    last_error = None
    for edge in ("posts", "feed"):
        for fields in (full_fields, basic_fields):
            try:
                data = _get(f"{page_id}/{edge}", access_token, {"fields": fields, "limit": min(limit, 100)})
                break
            except GraphAPIError as exc:
                last_error = exc
        if data is not None:
            break
    if data is None:
        raise last_error or GraphAPIError("Facebook Page post feed is unavailable")
    posts.extend(data.get("data", []))
    next_url = (data.get("paging") or {}).get("next")

    while next_url and len(posts) < limit:
        response = requests.get(next_url, timeout=30)
        data = _json(response)
        if "error" in data:
            break
        posts.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")

    return posts[:limit]


def fetch_post_insights(post_id: str, access_token: str) -> dict:
    """Per-post insights. Returns {} when insights aren't available.
    Uses page-token-safe metrics that don't require pages_read_user_content."""
    full_metrics = "post_impressions,post_impressions_unique,post_clicks,post_reactions_by_type_total,post_video_views"
    safe_metrics = "post_impressions,post_impressions_unique,post_clicks"
    for metrics in (full_metrics, safe_metrics):
        try:
            data = _get(f"{post_id}/insights", access_token, {"metric": metrics, "period": "lifetime"})
            break
        except GraphAPIError:
            if metrics == safe_metrics:
                return {}
    else:
        return {}
    result = {}
    for item in data.get("data", []):
        values = item.get("values") or []
        value = values[0].get("value") if values else None
        if item["name"] == "post_reactions_by_type_total" and isinstance(value, dict):
            value = sum(value.values())
        name = item["name"]
        if name == "post_impressions_unique":
            result["reach"] = value
        elif name == "post_impressions":
            result["impressions"] = value
        else:
            result[name] = value
    return result


def fetch_page_info(page_id: str, access_token: str) -> dict:
    """Page-level metrics from the page object directly.
    Graph API v21.0 deprecated most /insights metrics for pages that haven't
    passed App Review, but basic page fields still work with a page token."""
    # Several historic Page fields are no longer returned by newer Graph API
    # versions. Probe modern/basic groups progressively rather than rejecting
    # all useful metadata because one legacy field is unavailable.
    data = None
    last_error = None
    for fields in (
        "id,name,followers_count,talking_about_count,category,link",
        "id,name,followers_count,category,link",
        "id,name,category,link",
    ):
        try:
            data = _get(page_id, access_token, {"fields": fields})
            break
        except GraphAPIError as exc:
            last_error = exc
    if data is None:
        log.warning("Page info unavailable: %s", last_error)
        return {}
    return {
        "followers_count": data.get("followers_count") or data.get("fan_count"),
        "fan_count": data.get("fan_count"),
        "talking_about_count": data.get("talking_about_count"),
        "new_like_count": data.get("new_like_count"),
        "were_here_count": data.get("were_here_count"),
        "category": data.get("category"),
        "page_name": data.get("name"),
    }


def fetch_page_insights(page_id: str, access_token: str, period: str = "days_28") -> dict:
    """Page-level aggregate metrics. Tries the insights endpoint first,
    falls back to page object fields if insights are unavailable (common
    for apps without App Review on Graph API v21.0+)."""
    metrics = (
        "page_impressions,page_impressions_unique,page_engaged_users,"
        "page_post_engagements,page_fans,page_fan_adds,page_views_total"
    )
    try:
        data = _get(f"{page_id}/insights", access_token, {"metric": metrics, "period": period})
        result = {}
        for item in data.get("data", []):
            values = item.get("values") or []
            if values:
                result[item["name"]] = values[-1].get("value")
        if result:
            return result
    except GraphAPIError:
        pass
    return fetch_page_info(page_id, access_token)


# ---------------------------------------------------------------------------
# Instagram — media and insights
# ---------------------------------------------------------------------------

def fetch_ig_media(ig_user_id: str, access_token: str, limit: int = 100) -> list[dict]:
    """Fetch Instagram media (posts, reels, stories) for a professional/business account."""
    media: list[dict] = []
    data = _get(
        f"{ig_user_id}/media",
        access_token,
        {
            "fields": "id,caption,media_type,media_url,thumbnail_url,permalink,timestamp,"
                       "like_count,comments_count",
            "limit": min(limit, 100),
        },
    )
    media.extend(data.get("data", []))
    next_url = (data.get("paging") or {}).get("next")

    while next_url and len(media) < limit:
        response = requests.get(next_url, timeout=30)
        data = _json(response)
        if "error" in data:
            break
        media.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")

    return media[:limit]


def fetch_ig_media_insights(media_id: str, access_token: str, media_type: str = "IMAGE") -> dict:
    """Per-media insights for Instagram. Metric names differ by media type."""
    if media_type in ("VIDEO", "REELS"):
        metrics = "reach,impressions,plays,saved,shares,comments,likes"
    elif media_type == "CAROUSEL_ALBUM":
        metrics = "reach,impressions,saved,shares,comments,likes"
    else:
        metrics = "reach,impressions,saved,shares,comments,likes"
    try:
        data = _get(f"{media_id}/insights", access_token, {"metric": metrics})
    except GraphAPIError:
        return {}
    result = {}
    for item in data.get("data", []):
        values = item.get("values") or []
        result[item["name"]] = values[0].get("value") if values else item.get("total_value", {}).get("value")
    return result


def fetch_ig_account_insights(ig_user_id: str, access_token: str) -> dict:
    """Account-level Instagram metrics (followers, reach, profile views)."""
    try:
        data = _get(
            f"{ig_user_id}/insights",
            access_token,
            {"metric": "reach,impressions,profile_views,accounts_engaged", "period": "day", "metric_type": "total_value"},
        )
    except GraphAPIError as e:
        log.warning("IG account insights unavailable: %s", e)
        return {}
    result = {}
    for item in data.get("data", []):
        result[item["name"]] = item.get("total_value", {}).get("value")
    return result


def fetch_ig_profile(ig_user_id: str, access_token: str) -> dict:
    """Basic IG profile info: followers, following, media count, bio."""
    try:
        return _get(ig_user_id, access_token, {
            "fields": "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url"
        })
    except GraphAPIError:
        return {}


# ---------------------------------------------------------------------------
# Discovery — find pages and linked Instagram accounts from a user token
# ---------------------------------------------------------------------------

def discover_pages(access_token: str) -> list[dict]:
    """List all Facebook Pages the user token has access to, with linked IG accounts.
    Used in setup: user gives a token, we show what pages and IG accounts they manage."""
    data = _get("me/accounts", access_token, {
        "fields": "id,name,category,access_token,"
                  "instagram_business_account{id,username,name,followers_count,media_count,profile_picture_url}"
    })
    pages = []
    for page in data.get("data", []):
        entry = {
            "page_id": page["id"],
            "page_name": page.get("name"),
            "category": page.get("category"),
        }
        ig = page.get("instagram_business_account")
        if ig:
            entry["instagram"] = {
                "ig_user_id": ig["id"],
                "username": ig.get("username"),
                "name": ig.get("name"),
                "followers_count": ig.get("followers_count"),
                "media_count": ig.get("media_count"),
            }
        pages.append(entry)
    return pages


def discover_instagram_account(page_id: str, access_token: str) -> dict | None:
    """Find the Instagram Business/Creator account linked to a specific Facebook Page."""
    try:
        data = _get(page_id, access_token, {
            "fields": "instagram_business_account{id,username,name,profile_picture_url,followers_count,media_count}"
        })
    except GraphAPIError:
        return None
    ig = data.get("instagram_business_account")
    if not ig:
        return None
    return {
        "ig_user_id": ig["id"],
        "username": ig.get("username"),
        "name": ig.get("name"),
        "profile_picture_url": ig.get("profile_picture_url"),
        "followers_count": ig.get("followers_count"),
        "media_count": ig.get("media_count"),
    }


# ---------------------------------------------------------------------------
# Comments — fetch comment text from our own posts
# ---------------------------------------------------------------------------

def fetch_post_comments(post_id: str, access_token: str, limit: int = 100) -> list[dict]:
    """Fetch comments on a Facebook post. Returns [{id, message, created_time, from}]."""
    comments: list[dict] = []
    try:
        data = _get(f"{post_id}/comments", access_token, {
            "fields": "id,message,created_time,from{id,name}",
            "limit": min(limit, 100),
        })
    except GraphAPIError:
        return []
    comments.extend(data.get("data", []))
    next_url = (data.get("paging") or {}).get("next")
    while next_url and len(comments) < limit:
        response = requests.get(next_url, timeout=30)
        data = _json(response)
        if "error" in data:
            break
        comments.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")
    return comments[:limit]


def fetch_ig_media_comments(media_id: str, access_token: str, limit: int = 100) -> list[dict]:
    """Fetch comments on an Instagram media item. Returns [{id, text, timestamp, username}]."""
    comments: list[dict] = []
    try:
        data = _get(f"{media_id}/comments", access_token, {
            "fields": "id,text,timestamp,username",
            "limit": min(limit, 100),
        })
    except GraphAPIError:
        return []
    comments.extend(data.get("data", []))
    next_url = (data.get("paging") or {}).get("next")
    while next_url and len(comments) < limit:
        response = requests.get(next_url, timeout=30)
        data = _json(response)
        if "error" in data:
            break
        comments.extend(data.get("data", []))
        next_url = (data.get("paging") or {}).get("next")
    return comments[:limit]


# ---------------------------------------------------------------------------
# Publishing (unchanged from before)
# ---------------------------------------------------------------------------

def publish_facebook_text(page_id: str, access_token: str, message: str) -> str:
    return _post(f"{page_id}/feed", access_token, {"message": message})["id"]


def publish_facebook_photo(page_id: str, access_token: str, photo_url: str, message: str) -> str:
    body = _post(f"{page_id}/photos", access_token, {"url": photo_url, "caption": message})
    return body.get("post_id") or body["id"]


def publish_facebook_video(page_id: str, access_token: str, video_path: str, message: str) -> str:
    with open(video_path, "rb") as f:
        response = requests.post(
            f"https://graph-video.facebook.com/{config.META_GRAPH_API_VERSION}/{page_id}/videos",
            data={"description": message, "access_token": access_token},
            files={"source": f},
            timeout=600,
        )
    body = _json(response)
    if "error" in body:
        raise GraphAPIError(body["error"].get("message", str(body["error"])))
    return body["id"]


def publish_instagram(
    ig_user_id: str, access_token: str, caption: str, image_url: str | None = None, video_url: str | None = None, on_container=None
) -> str:
    import time

    params = {"caption": caption}
    if video_url:
        params.update({"media_type": "REELS", "video_url": video_url})
    elif image_url:
        params["image_url"] = image_url
    else:
        raise GraphAPIError("Instagram posts need an image or video")
    container_id = _post(f"{ig_user_id}/media", access_token, params)["id"]
    if on_container:
        on_container(container_id)

    for _ in range(40):
        status = _get(container_id, access_token, {"fields": "status_code"}).get("status_code")
        if status == "FINISHED":
            break
        if status in {"ERROR", "EXPIRED"}:
            raise GraphAPIError(f"Instagram media container {status}")
        time.sleep(5)
    else:
        raise GraphAPIError("Instagram media container timed out")
    return _post(f"{ig_user_id}/media_publish", access_token, {"creation_id": container_id})["id"]
