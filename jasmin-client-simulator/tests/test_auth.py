from app.core.security import DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD, DEFAULT_USER_ROLE


def test_admin_credentials_are_not_served_publicly(client):
    assert client.get("/api/v1/auth/default-credentials").status_code == 404


def test_auth_login_successful(client):
    response = client.post("/api/v1/auth/login", json={
        "email": DEFAULT_USER_EMAIL,
        "password": DEFAULT_USER_PASSWORD
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == DEFAULT_USER_EMAIL
    assert data["user"]["role"] == DEFAULT_USER_ROLE


def test_auth_login_invalid_password(client):
    response = client.post("/api/v1/auth/login", json={
        "email": DEFAULT_USER_EMAIL,
        "password": "WrongPassword999!"
    })
    assert response.status_code == 401
    assert "Identifiants invalides" in response.json()["detail"]


def test_auth_login_unknown_user(client):
    response = client.post("/api/v1/auth/login", json={
        "email": "nonexistent@simulator.com",
        "password": "AnyPassword123!"
    })
    assert response.status_code == 401
