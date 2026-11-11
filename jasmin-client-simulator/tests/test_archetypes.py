import pytest
from app.customers.archetypes import ARCHETYPES
from app.customers.default_personas import DEFAULT_PERSONAS
from app.customers.generator import generate_customer_persona
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot
from app.conversation.engine import ConversationEngine
from app.conversation.prompts import build_system_prompt


def test_archetypes_endpoint_lists_every_type(client):
    data = client.get("/api/v1/archetypes").json()
    assert [a["key"] for a in data] == list(ARCHETYPES)
    assert {"anxious", "demanding", "rushed", "family"} <= {a["key"] for a in data}
    assert all(a["label"] and a["color"].startswith("#") for a in data)


def test_every_archetype_has_a_default_persona():
    assert {p.archetype for p in DEFAULT_PERSONAS} == set(ARCHETYPES)


@pytest.mark.parametrize("key", list(ARCHETYPES))
def test_greeting_and_prompt_follow_archetype(key, db_session):
    persona = generate_customer_persona(archetype=key, destination="Hammamet", budget=2000)
    assert persona.archetype == key
    profile = CustomerProfileResponse(**persona.model_dump(), id="p", created_at="2026-10-18T00:00:00Z")

    greeting = ConversationEngine(db_session).generate_initial_greeting(profile)
    assert profile.name in greeting and "2000" in greeting

    state = CustomerStateSnapshot(
        trust_level=5, interest_level=5, booking_intention=3, current_stage="greetings",
        unresolved_objections=[], known_requirements={}, questions_answered=[], provided_info={}
    )
    prompt = build_system_prompt(profile, state, "")
    assert ARCHETYPES[key]["behavior"] in prompt
    assert ARCHETYPES[key]["label"] in prompt


def test_unknown_archetype_falls_back_to_default():
    assert generate_customer_persona(archetype="inconnu").archetype == "satisfied"
