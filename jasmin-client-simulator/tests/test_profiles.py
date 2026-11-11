def test_list_seeded_profiles(client):
    response = client.get("/api/v1/profiles")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    persona_names = [p["name"] for p in data]
    assert "Camille Dupont" in persona_names
    assert "Carlos Mendez" in persona_names
    assert "Sofia Rossi" in persona_names



def test_create_custom_profile(client):
    payload = {
        "scenario_id": "test_scenario",
        "name": "Alice Smith",
        "age_range": "25-30",
        "persona_description": "Budget solo traveler.",
        "destination_preferences": ["Hammamet", "Radisson Blu Palace Resort Djerba"],
        "travel_dates": {"start_date": "2026-11-01", "end_date": "2026-11-10", "flexible": True},
        "number_of_travelers": 1,
        "budget_total": 2000.0,
        "price_sensitivity": "medium",
        "preferred_accommodation": "3-star hotel",
        "preferred_transportation": "Economy flight",
        "motivations": ["Culture"],
        "constraints": ["Max $2000"],
        "personality_traits": ["curious"],
        "communication_style": "Polite",
        "objections": ["Expensive flights"],
        "decision_criteria": ["Price"],
        "initial_trust": 5.0,
        "initial_interest": 5.0,
        "initial_booking_intention": 3.0,
        "desired_outcome": "Réserver un séjour à Hammamet dans le budget."
    }
    response = client.post("/api/v1/profiles", json=payload)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["name"] == "Alice Smith"
    assert res_data["budget_total"] == 2000.0
    assert "id" in res_data


def test_profiles_outside_tunisia_are_refused(client):
    payload = client.get("/api/v1/profiles").json()[0]
    payload = {k: v for k, v in payload.items() if k not in ("id", "created_at")}
    res = client.post("/api/v1/profiles", json={**payload, "destination_preferences": ["Cancun", "Hammamet"]})
    assert res.status_code == 422 and "Cancun" in res.json()["detail"]
    assert client.post("/api/v1/profiles/generate?archetype=satisfied&destination=Bali").status_code == 422
    assert client.post("/api/v1/profiles/generate?archetype=satisfied&destination=Mahdia").status_code == 201


def test_every_stored_profile_is_in_tunisia(client, db_session):
    from app.hotels.catalog import is_tunisian_place
    for p in client.get("/api/v1/profiles").json():
        assert all(is_tunisian_place(d, db_session) for d in p["destination_preferences"]), p["name"]
