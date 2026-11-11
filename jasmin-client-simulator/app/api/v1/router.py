from fastapi import APIRouter
from app.api.v1.admin import router as admin_router
from app.api.v1.archetypes import router as archetypes_router
from app.api.v1.auth import router as auth_router
from app.api.v1.client_space import router as client_space_router
from app.api.v1.profiles import router as profiles_router
from app.api.v1.simulations import router as simulations_router
from app.api.v1.twins import router as twins_router

api_v1_router = APIRouter()
api_v1_router.include_router(admin_router)
api_v1_router.include_router(archetypes_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(client_space_router)
api_v1_router.include_router(profiles_router)
api_v1_router.include_router(simulations_router)
api_v1_router.include_router(twins_router)
