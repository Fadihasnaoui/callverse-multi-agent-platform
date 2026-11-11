import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import SimulatorException, SimulationNotFoundException, ProfileNotFoundException
from app.database.session import init_db
from app.api.v1.router import api_v1_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Initialize DB tables on startup
    init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.exception_handler(SimulationNotFoundException)
@app.exception_handler(ProfileNotFoundException)
async def not_found_handler(request: Request, exc: SimulatorException):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Mount Static Files & Dashboard UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# HTML pages change with each release: make browsers revalidate instead of showing a stale copy
NO_CACHE = {"Cache-Control": "no-cache"}

@app.get("/", include_in_schema=False)
def serve_dashboard():
    dashboard_path = os.path.join(static_dir, "index.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, headers=NO_CACHE)
    return {"message": "Digital Twin Client Simulator API is running."}

@app.get("/admin", include_in_schema=False)
def serve_admin_dashboard():
    # The page itself is public; its data comes from /api/v1/admin/* which requires an admin token
    return FileResponse(os.path.join(static_dir, "admin.html"), headers=NO_CACHE)

@app.get("/questionnaire", include_in_schema=False)
def serve_questionnaire():
    # Public page for real clients: creates their "jumeau numérique" through POST /api/v1/twins
    return FileResponse(os.path.join(static_dir, "questionnaire.html"), headers=NO_CACHE)

@app.get("/espace", include_in_schema=False)
def serve_client_space():
    # Client space: data comes from /api/v1/me/* which requires a client token
    return FileResponse(os.path.join(static_dir, "client.html"), headers=NO_CACHE)

@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/x-icon")
    from fastapi import Response
    return Response(status_code=204)

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

