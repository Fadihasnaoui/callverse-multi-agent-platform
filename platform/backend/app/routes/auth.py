from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app import auth, config

router = APIRouter(prefix="/auth", tags=["auth"])


class Login(BaseModel):
    password: str


@router.post("/login")
def login(payload: Login, request: Request, response: Response):
    if not config.AUTH_PASSWORD:
        raise HTTPException(status_code=503, detail="AUTH_PASSWORD is not set on the server")
    client = request.client.host if request.client else "unknown"
    if auth.too_many_failures(client):
        raise HTTPException(status_code=429, detail="too many attempts, wait a few minutes")
    if not auth.check_password(payload.password):
        auth.record_failure(client)
        raise HTTPException(status_code=401, detail="wrong password")
    token = auth.make_token()
    response.set_cookie(auth.COOKIE, token, max_age=3600 * config.AUTH_SESSION_HOURS, httponly=True,
                        samesite="lax", secure=config.AUTH_COOKIE_SECURE)
    return {"ok": True}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    if config.ALLOW_NO_AUTH or auth.valid_token(request.cookies.get(auth.COOKIE)):
        return {"authenticated": True, "auth_required": not config.ALLOW_NO_AUTH}
    raise HTTPException(status_code=401, detail="not logged in")
