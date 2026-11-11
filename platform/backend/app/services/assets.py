"""Phase 2 - visual asset sourcing. Google Places (New) Text Search as the
primary photo source - one real hotel's own Google Business listing, not the
agency's own site - with an Open Graph image read from the place's own
website as a fallback when Places has nothing usable. Every distinct
hotel/destination lookup is cached in asset_cache so repeat content
generation for the same place never re-pays the Places API cost."""
import re

import requests

from app import db, config

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_PHOTO_URL = "https://places.googleapis.com/v1/{photo_name}/media"


def _query_key(hotel_name: str, destination: str) -> str:
    return f"{hotel_name.strip().lower()}|{destination.strip().lower()}"


def _cached(conn, query_key: str) -> dict | None:
    row = conn.execute(
        "SELECT place_id, website_uri, photo_urls, source FROM asset_cache WHERE query_key = %s",
        (query_key,),
    ).fetchone()
    if not row:
        return None
    return {"place_id": row[0], "website_uri": row[1], "photo_urls": row[2] or [], "source": row[3]}


def _store_cache(conn, query_key: str, place_id: str | None, website_uri: str | None, photo_urls: list[str], source: str):
    conn.execute(
        "INSERT INTO asset_cache (query_key, place_id, website_uri, photo_urls, source) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (query_key) DO UPDATE SET place_id = EXCLUDED.place_id, "
        "website_uri = EXCLUDED.website_uri, photo_urls = EXCLUDED.photo_urls, source = EXCLUDED.source, "
        "fetched_at = now()",
        (query_key, place_id, website_uri, photo_urls, source),
    )
    conn.commit()


def _search_place(hotel_name: str, destination: str) -> dict | None:
    if not config.GOOGLE_PLACES_API_KEY:
        raise RuntimeError(
            "GOOGLE_PLACES_API_KEY is not set and no cached/manual photos exist for this hotel - "
            "either set the key, or POST /assets/manual with real photo URLs for this hotel first."
        )
    response = requests.post(
        PLACES_SEARCH_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": config.GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.websiteUri,places.photos",
        },
        json={"textQuery": f"{hotel_name} {destination} hotel"},
        timeout=20,
    )
    response.raise_for_status()
    places = response.json().get("places", [])
    return places[0] if places else None


def _photo_url(photo_name: str, max_width_px: int = 1200) -> str:
    return (
        PLACES_PHOTO_URL.format(photo_name=photo_name)
        + f"?maxWidthPx={max_width_px}&key={config.GOOGLE_PLACES_API_KEY}"
    )


def _og_image(website_uri: str) -> str | None:
    """A single lightweight metadata read - not a scrape - of the place's own
    official site. Deliberately not following links/parsing a gallery; just
    the one image the site itself nominated as its representative photo."""
    try:
        response = requests.get(website_uri, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException:
        return None
    match = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', response.text, re.IGNORECASE
    )
    if not match:
        match = re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', response.text, re.IGNORECASE
        )
    return match.group(1) if match else None


def register_manual_photos(
    hotel_name: str, destination: str, photo_urls: list[str], website_uri: str | None = None
) -> dict:
    """Hand-picked real photos, pulled by a human from the hotel's own
    website, written into the same cache Google Places would use. This is
    the primary photo source for now - no GOOGLE_PLACES_API_KEY required at
    all for any hotel registered this way."""
    query_key = _query_key(hotel_name, destination)
    with db.get_conn() as conn:
        _store_cache(conn, query_key, None, website_uri, photo_urls, "manual")
    return {"place_id": None, "website_uri": website_uri, "photo_urls": photo_urls, "source": "manual"}


def source_photos(hotel_name: str, destination: str, max_photos: int = 5) -> dict:
    """Returns {place_id, website_uri, photo_urls, source}. Checks
    asset_cache first; only calls out to Google/the hotel's site on a real
    cache miss."""
    query_key = _query_key(hotel_name, destination)

    with db.get_conn() as conn:
        cached = _cached(conn, query_key)
        if cached is not None:
            return cached

        place = _search_place(hotel_name, destination)
        if place is None:
            result = {"place_id": None, "website_uri": None, "photo_urls": [], "source": "none"}
            _store_cache(conn, query_key, None, None, [], "none")
            return result

        place_id = place.get("id")
        website_uri = place.get("websiteUri")
        photos = place.get("photos") or []

        if photos:
            photo_urls = [_photo_url(p["name"]) for p in photos[:max_photos]]
            source = "google_places"
        elif website_uri:
            og_url = _og_image(website_uri)
            photo_urls = [og_url] if og_url else []
            source = "og_fallback" if og_url else "none"
        else:
            photo_urls = []
            source = "none"

        _store_cache(conn, query_key, place_id, website_uri, photo_urls, source)
        return {"place_id": place_id, "website_uri": website_uri, "photo_urls": photo_urls, "source": source}
