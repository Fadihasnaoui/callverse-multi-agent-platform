from app.core.security import DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD


def _auth_headers(client, email, password):
    token = client.post("/api/v1/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_admin_stats_requires_token(client):
    assert client.get("/api/v1/admin/overview").status_code == 401
    assert client.get("/api/v1/admin/overview", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_admin_stats_forbidden_for_sales_rep(client):
    headers = _auth_headers(client, "rep@simulator.com", "RepPass123!")
    assert client.get("/api/v1/admin/overview", headers=headers).status_code == 403


def test_admin_overview_stats_per_archetype(client):
    headers = _auth_headers(client, DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD)
    client.post("/api/v1/profiles/generate?archetype=unsatisfied")
    profiles = client.get("/api/v1/profiles").json()
    carlos = next(p for p in profiles if p["name"] == "Carlos Mendez")

    sim_id = client.post("/api/v1/simulations", json={"profile_id": carlos["id"]}).json()["id"]
    client.post(f"/api/v1/simulations/{sim_id}/messages", json={"message": "Bonjour Carlos !"})
    client.post(f"/api/v1/simulations/{sim_id}/messages", json={"message": "Tout inclus, sans frais caches, garantie."})

    data = client.get("/api/v1/admin/overview", headers=headers).json()
    assert data["totals"]["clients"] == 9  # 8 default personas + 1 generated
    assert data["totals"]["simulations"] == 1
    by_arch = {a["archetype"]: a for a in data["archetypes"]}
    assert list(by_arch) == ["satisfied", "unsatisfied", "hesitant", "anxious", "demanding", "rushed", "family", "vip"]
    assert by_arch["unsatisfied"]["clients"] == 2
    assert all(by_arch[k]["clients"] == 1 for k in by_arch if k != "unsatisfied")
    assert by_arch["family"]["label"] == "Famille avec enfants" and by_arch["family"]["color"]
    assert by_arch["unsatisfied"]["simulations"] == 1
    assert by_arch["unsatisfied"]["quality"]["excellent"] == 1

    # One curve point per customer message: greeting + 2 replies, trust rising after the good answer
    trust = data["curves"]["trust_level"]["unsatisfied"]
    assert len(trust) == 3 and trust[2] > trust[1]
    assert data["curves"]["trust_level"]["satisfied"] == []
    assert data["curve_counts"]["unsatisfied"] == [1, 1, 1]
    assert sum(data["daily"]["unsatisfied"]) == 1 and len(data["daily"]["dates"]) == 30
    assert next(c for c in data["clients"] if c["id"] == carlos["id"])["simulations"] == 1


def test_me_returns_token_owner(client):
    headers = _auth_headers(client, "rep@simulator.com", "RepPass123!")
    assert client.get("/api/v1/auth/me", headers=headers).json()["email"] == "rep@simulator.com"
    assert client.get("/api/v1/auth/me").status_code == 401


def test_session_is_persisted_and_logout_revokes_it(client, db_session):
    from app.database.models import UserSessionModel
    headers = _auth_headers(client, DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD)
    token = headers["Authorization"].removeprefix("Bearer ")
    # Stored in the DB, not process memory, so a server restart keeps the admin logged in
    assert db_session.get(UserSessionModel, token) is not None
    assert client.get("/api/v1/admin/overview", headers=headers).status_code == 200

    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/v1/admin/overview", headers=headers).status_code == 401


def test_pages_link_to_dashboard_and_are_not_cached(client):
    home = client.get("/")
    assert 'href="/admin"' in home.text
    assert home.headers["cache-control"] == "no-cache"
    admin_page = client.get("/admin")
    assert admin_page.status_code == 200 and "/api/v1/admin/overview" in admin_page.text
    assert admin_page.headers["cache-control"] == "no-cache"
