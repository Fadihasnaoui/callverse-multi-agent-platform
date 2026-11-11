import pytest
from datetime import date, timedelta
from pydantic import ValidationError
from app.customers.archetypes import ARCHETYPES
from app.customers.twin import TRIP_GOALS, TwinQuestionnaire, derive_archetype
from app.database.models import CustomerProfileModel

BASE = {
    "first_name": "Amira", "travelers": 2, "budget": 6200, "currency": "TND",
    "destination": "Djerba", "trip_goal": "detente", "notes": None, "consent": True,
    "email": "amira@example.com", "password": "Voyage2026!",
    "start_date": (date.today() + timedelta(days=30)).isoformat(),
    "end_date": (date.today() + timedelta(days=37)).isoformat(),
}


def answers(**changes):
    return TwinQuestionnaire(**{**BASE, **changes})


def test_trip_goal_decides_the_traveler_type():
    expected = {"detente": "satisfied", "sans_surprise": "unsatisfied", "rassurer": "anxious", "luxe": "vip",
                "garanties": "demanding", "rapide": "rushed", "famille": "family", "conseils": "hesitant"}
    assert {goal: derive_archetype(answers(trip_goal=goal)) for goal in TRIP_GOALS} == expected
    assert set(expected.values()) == set(ARCHETYPES)  # every traveler type can be reached from the form


def test_form_validation():
    with pytest.raises(ValidationError):
        answers(consent=False)
    with pytest.raises(ValidationError):
        answers(destination="Paris")
    with pytest.raises(ValidationError):
        answers(trip_goal="inconnu")
    with pytest.raises(ValidationError):
        answers(travelers=0)
    with pytest.raises(ValidationError):
        answers(start_date=date.today().isoformat())  # must be in the future
    with pytest.raises(ValidationError):
        answers(end_date=BASE["start_date"])  # at least one night
    with pytest.raises(ValidationError):
        answers(end_date=(date.today() + timedelta(days=70)).isoformat())  # 30 nights max
    assert answers(currency="TND", budget=3100).budget_tnd == 3100
    assert answers(currency="USD", budget=1000).budget_tnd == 3100  # typed in dollars, stored in dinars


def test_create_persona_endpoint(client, db_session):
    res = client.post("/api/v1/twins", json={**BASE, "trip_goal": "luxe", "budget": 30000, "notes": "Anniversaire de mariage"})
    assert res.status_code == 201
    data = res.json()
    assert data["archetype"] == "vip" and data["archetype_label"] == "VIP" and data["archetype_color"]
    assert "suggested_hotels" not in data  # hotels are proposed by the advisor, not picked for the client

    persona = db_session.get(CustomerProfileModel, data["profile_id"])
    assert persona.scenario_id.startswith("twin_")
    assert persona.budget_total == 30000  # TND as typed
    assert persona.number_of_travelers == 2
    assert persona.destination_preferences == ["Djerba"]
    assert "Anniversaire de mariage" in persona.persona_description
    assert "La suite est-elle la meilleure de l'hôtel ?" in persona.objections

    # The persona plays the client in a training simulation right away, asking for its region
    sim = client.post("/api/v1/simulations", json={"profile_id": persona.id})
    assert sim.status_code == 201
    assert "Amira" in sim.json()["initial_message"] and "la région de Djerba" in sim.json()["initial_message"]


def test_persona_requires_consent(client):
    assert client.post("/api/v1/twins", json={**BASE, "consent": False}).status_code == 422


def test_form_page_and_admin_count(client):
    page = client.get("/questionnaire")
    assert page.status_code == 200 and "/api/v1/twins" in page.text and "trip_goal" in page.text
    assert 'href="/questionnaire"' in client.get("/").text

    client.post("/api/v1/twins", json=BASE)
    token = client.post("/api/v1/auth/login", json={"email": "admin@simulator.com", "password": "Password123!"}).json()["access_token"]
    overview = client.get("/api/v1/admin/overview", headers={"Authorization": f"Bearer {token}"}).json()
    assert overview["totals"]["twins"] == 1
    assert sum(c["is_twin"] for c in overview["clients"]) == 1
