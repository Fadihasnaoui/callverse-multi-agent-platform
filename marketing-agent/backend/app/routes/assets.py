import json
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app import config, db
from app.services import assets as assets_service

router = APIRouter(prefix="/assets", tags=["assets"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class RegisterPhotos(BaseModel):
    hotel_name: str
    destination: str
    photo_urls: list[str]
    website_uri: str | None = None


@router.get("/search")
def search(hotel_name: str, destination: str, max_photos: int = 5):
    return assets_service.source_photos(hotel_name, destination, max_photos=max_photos)


@router.post("/manual")
def register_manual(payload: RegisterPhotos):
    return assets_service.register_manual_photos(
        payload.hotel_name, payload.destination, payload.photo_urls, payload.website_uri
    )


@router.get("/catalogue")
def list_catalogue():
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT query_key, photo_urls, source, fetched_at "
            "FROM asset_cache ORDER BY fetched_at DESC"
        ).fetchall()
    items = []
    for r in rows:
        parts = r[0].split("|", 1)
        items.append({
            "hotel_name": parts[0] if parts else r[0],
            "destination": parts[1] if len(parts) > 1 else "",
            "photo_urls": r[1] or [],
            "source": r[2],
            "updated_at": r[3].isoformat() if r[3] else None,
        })
    return items


@router.post("/upload")
async def upload_photos(
    hotel_name: str = Form(...),
    destination: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="max 20 files per upload")

    os.makedirs(config.RENDERS_DIR, exist_ok=True)
    base = config.PUBLIC_BASE_URL.rstrip("/") if config.PUBLIC_BASE_URL else "http://127.0.0.1:8010"
    saved_urls: list[str] = []

    for f in files:
        if f.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"unsupported type: {f.content_type} — use JPEG, PNG or WebP")
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"{f.filename} exceeds 10 MB limit")
        ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[f.content_type]
        filename = f"catalogue-{uuid.uuid4().hex[:12]}.{ext}"
        path = os.path.join(config.RENDERS_DIR, filename)
        with open(path, "wb") as out:
            out.write(data)
        saved_urls.append(f"{base}/renders/{filename}")

    query_key = assets_service._query_key(hotel_name, destination)
    with db.get_conn() as conn:
        existing = conn.execute(
            "SELECT photo_urls FROM asset_cache WHERE query_key = %s", (query_key,)
        ).fetchone()
        merged = list(existing[0] or []) if existing else []
        merged.extend(saved_urls)
        conn.execute(
            "INSERT INTO asset_cache (query_key, photo_urls, source) "
            "VALUES (%s, %s, 'upload') "
            "ON CONFLICT (query_key) DO UPDATE SET photo_urls = %s, source = 'upload', fetched_at = now()",
            (query_key, merged, merged),
        )
        conn.commit()

    return {
        "hotel_name": hotel_name,
        "destination": destination,
        "uploaded": len(saved_urls),
        "photo_urls": merged,
    }


@router.delete("/catalogue")
def delete_catalogue_entry(hotel_name: str, destination: str):
    query_key = assets_service._query_key(hotel_name, destination)
    with db.get_conn() as conn:
        deleted = conn.execute(
            "DELETE FROM asset_cache WHERE query_key = %s RETURNING id", (query_key,)
        ).fetchone()
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="entry not found")
    return {"deleted": query_key}
