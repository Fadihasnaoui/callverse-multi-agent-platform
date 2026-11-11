from app.core.security import DEFAULT_CLIENT_PASSWORD, DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD
from app.customers.accounts import default_client_email
from tests.test_twins import BASE


def login(client, email, password):
    res = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}, res.json()["user"]


def profile_id(client, name):
    return next(p for p in client.get("/api/v1/profiles").json() if p["name"] == name)["id"]


def test_every_default_persona_can_log_in(client):
    headers, user = login(client, "leila.ben.ammar@client.jasmin.tn", DEFAULT_CLIENT_PASSWORD)  # accents dropped
    assert user["role"] == "client" and user["profile_id"] == profile_id(client, "Leïla Ben Ammar")
    assert client.get("/api/v1/me/dashboard", headers=headers).json()["profile"]["archetype"] == "vip"


def test_client_sees_only_its_own_simulations(client):
    sofia = client.post("/api/v1/simulations", json={"profile_id": profile_id(client, "Sofia Rossi")}).json()["id"]
    client.post(f"/api/v1/simulations/{sofia}/messages", json={"message": "Bonjour Sofia, je vous écoute."})
    carlos = client.post("/api/v1/simulations", json={"profile_id": profile_id(client, "Carlos Mendez")}).json()["id"]

    headers, _ = login(client, "sofia.rossi@client.jasmin.tn", DEFAULT_CLIENT_PASSWORD)
    data = client.get("/api/v1/me/dashboard", headers=headers).json()
    assert [s["id"] for s in data["simulations"]] == [sofia]
    assert data["stats"]["simulations"] == 1 and data["simulations"][0]["messages"] == 3

    transcript = client.get(f"/api/v1/me/simulations/{sofia}", headers=headers).json()["transcript"]
    assert [m["sender"] for m in transcript] == ["customer", "sales_rep", "customer"]
    assert transcript[1]["content"] == "Bonjour Sofia, je vous écoute."
    assert client.get(f"/api/v1/me/simulations/{carlos}", headers=headers).status_code == 404  # someone else's


def test_client_space_is_for_clients_only(client):
    assert client.get("/api/v1/me/dashboard").status_code == 401
    headers, _ = login(client, DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD)
    assert client.get("/api/v1/me/dashboard", headers=headers).status_code == 403
    sofia_headers, _ = login(client, "sofia.rossi@client.jasmin.tn", DEFAULT_CLIENT_PASSWORD)
    assert client.get("/api/v1/admin/overview", headers=sofia_headers).status_code == 403
    assert client.get("/espace").status_code == 200


def test_twin_logs_in_with_the_password_it_chose(client):
    assert client.post("/api/v1/twins", json=BASE).json()["email"] == "amira@example.com"
    headers, user = login(client, "AMIRA@example.com ", BASE["password"])
    assert client.get("/api/v1/me/dashboard", headers=headers).json()["profile"]["name"] == "Amira"
    assert client.post("/api/v1/twins", json=BASE).status_code == 409  # one account per email
    assert client.post("/api/v1/twins", json={**BASE, "email": "x@y.z", "password": "court"}).status_code == 422


def test_generated_persona_gets_a_login_and_admin_sees_it(client, db_session):
    generated = client.post("/api/v1/profiles/generate?archetype=rushed").json()
    headers, _ = login(client, DEFAULT_USER_EMAIL, DEFAULT_USER_PASSWORD)
    row = next(c for c in client.get("/api/v1/admin/overview", headers=headers).json()["clients"] if c["id"] == generated["id"])
    assert row["login_email"].endswith("@client.jasmin.tn")
    assert default_client_email(db_session, "Sofia Rossi") == "sofia.rossi-2@client.jasmin.tn"  # taken: suffixed
