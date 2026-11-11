import secrets
from fastapi import APIRouter, Depends, HTTPException, status, Header
from typing import Optional
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import UserModel, UserSessionModel
from app.customers.schemas import LoginRequest, LoginResponse, UserResponse
from app.core.security import verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])

def current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> UserModel:
    token = (authorization or "").removeprefix("Bearer ").strip()
    session = db.get(UserSessionModel, token) if token else None
    user = db.get(UserModel, session.user_id) if session else None
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session invalide ou expirée.")
    return user


def require_admin(user: UserModel = Depends(current_user)) -> UserModel:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès réservé aux administrateurs.")
    return user


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email_clean = payload.email.strip().lower()
    user = db.query(UserModel).filter(UserModel.email == email_clean).first()
    
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides (email ou mot de passe incorrect)."
        )

    # Generate a bearer token session
    token = f"sim_token_{secrets.token_hex(24)}"
    db.add(UserSessionModel(token=token, user_id=user.id))
    db.commit()

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    db.query(UserSessionModel).filter(UserSessionModel.token == token).delete()
    db.commit()


@router.get("/me", response_model=UserResponse)
def get_current_user(user: UserModel = Depends(current_user)):
    return UserResponse.model_validate(user)
