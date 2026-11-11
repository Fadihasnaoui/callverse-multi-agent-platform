import asyncio
import os
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config, db
from app.ai import AIError
from app import auth
from app.routes import ai_stats, assets, audit, businesses, competitors, content, glossary, memory, notifications, outcomes, webhooks
from app.routes import auth as auth_routes
from app.routes import workforce_proxy
from app.routes import digital_twin
from app.services import learning, metrics, publisher
from app.services import memory as memory_service
from app.services import jobs
from psycopg.errors import CheckViolation
from app.services.costs import BudgetExceeded

app = FastAPI(title="Marketing Agent")


@app.exception_handler(jobs.IdempotencyConflict)
async def duplicate_operation(request,error):
    return JSONResponse(status_code=409,content={'detail':str(error)})


@app.exception_handler(BudgetExceeded)
async def budget_exceeded(request,error):
    return JSONResponse(status_code=402,content={'detail':str(error)})


@app.exception_handler(CheckViolation)
async def state_conflict(request, error):
    return JSONResponse(status_code=409,content={'detail':'Content changed or is locked; reload and review the current version.'})


@app.middleware("http")
async def require_login(request: Request, call_next):
    """Everything under /businesses, /assets and /ai needs a staff session. Webhooks, health, /auth and
    /renders stay public (Meta and Instagram must reach them)."""
    # SPA entry pages must remain reachable so React can render the staff
    # login form.  Their JSON APIs are still protected by the prefixes below.
    # Without this exception a direct browser visit to /digital-twin returns a
    # raw {"detail":"login required"} response instead of the login UI.
    frontend_entries = {
        "/marketing", "/workforce", "/digital-twin", "/customer-advisor",
        "/client-simulator", "/quality-analyst", "/workforce-lab",
    }
    if request.url.path not in frontend_entries and auth.is_protected(request.url.path) and not config.ALLOW_NO_AUTH:
        if not config.AUTH_PASSWORD:
            return JSONResponse(status_code=503, content={"detail": "server login is not configured: set AUTH_PASSWORD"})
        if not auth.valid_token(request.cookies.get(auth.COOKIE)):
            return JSONResponse(status_code=401, content={"detail": "login required"})
    return await call_next(request)


@app.exception_handler(AIError)
async def ai_error_handler(request: Request, error: AIError):
    # The model could not produce a valid answer even after repair/fallback. Nothing bad was saved.
    return JSONResponse(status_code=502, content={"detail": f"The AI could not produce a valid answer, please try again. ({error})"})


app.include_router(auth_routes.router)
app.include_router(businesses.router)
app.include_router(audit.router)
app.include_router(content.router)
app.include_router(memory.router)
app.include_router(glossary.router)
app.include_router(ai_stats.router)
app.include_router(assets.router)
app.include_router(outcomes.router)
app.include_router(competitors.router)
app.include_router(notifications.router)
app.include_router(webhooks.router)
app.include_router(workforce_proxy.router)
app.include_router(digital_twin.router)

os.makedirs(config.RENDERS_DIR, exist_ok=True)
app.mount("/renders", StaticFiles(directory=config.RENDERS_DIR), name="renders")

_web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
_web_assets = _web_dist / "app-assets"
if _web_assets.exists():
    app.mount("/app-assets", StaticFiles(directory=_web_assets), name="app-assets")

_scheduler = BackgroundScheduler()


@app.on_event("startup")
async def on_startup():
    db.apply_schema()
    # Calls of a run cut off by this restart: close them cleanly in the background.
    from app.routes.digital_twin import recover_interrupted_calls
    asyncio.create_task(recover_interrupted_calls())
    _scheduler.add_job(jobs.run_pending,'interval',seconds=2,kwargs={'only_safe':True},max_instances=1,coalesce=True,misfire_grace_time=30)
    _scheduler.add_job(jobs.run_pending,'interval',seconds=2,kwargs={'only_safe':False},max_instances=1,coalesce=True,misfire_grace_time=30)
    # max_instances=1 + coalesce: a slow publish (video upload) never stacks ticks.
    _scheduler.add_job(
        publisher.publish_due,
        "interval",
        seconds=config.PUBLISH_INTERVAL_SECONDS,
        max_instances=1,
        coalesce=True,
        # Default grace is 1s; under Docker Desktop clock lag every tick was
        # being skipped as "missed" and nothing ever published.
        misfire_grace_time=30,
    )
    # Phase 8: learning + sentiment refresh every 3 days at 03:00 Tunis time.
    _scheduler.add_job(
        learning.run_learning_all, "interval", days=3,
        misfire_grace_time=3600, coalesce=True, max_instances=1,
    )
    _scheduler.add_job(
        metrics.sync_all, "interval", hours=6, max_instances=1, coalesce=True,
        misfire_grace_time=900,
    )
    # Keep long-lived memory small: fold old entries of busy scopes into summaries.
    _scheduler.add_job(
        memory_service.compact_all, "cron", day_of_week="mon", hour=3, minute=30,
        timezone="Africa/Tunis", misfire_grace_time=3600, coalesce=True,
    )
    _scheduler.start()


@app.on_event("shutdown")
def on_shutdown():
    _scheduler.shutdown(wait=False)


def _frontend():
    """Serve the built React application while API routes remain handled above."""
    index = _web_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        status_code=503,
        content={"detail": "interface is not built; run npm run build in web"},
    )


@app.get("/")
@app.get('/marketing')
@app.get('/workforce')
@app.get('/digital-twin')
@app.get('/customer-advisor')
@app.get('/client-simulator')
@app.get('/quality-analyst')
@app.get('/workforce-lab')
def frontend_root():
    return _frontend()


@app.get("/dashboard")
def dashboard():
    return _frontend()


@app.get("/calendar")
def calendar():
    return _frontend()


@app.get("/leads")
def leads_page():
    return _frontend()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ai-stats")
def ai_stats_page():
    return _frontend()


@app.get("/c/{frontend_path:path}")
def customer_app(frontend_path: str):
    return _frontend()
