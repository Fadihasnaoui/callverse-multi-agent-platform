import io
import pytest
from app.conversation.prompts import build_system_prompt
from app.conversation.voice_service import voice_service
from app.customers.archetypes import ARCHETYPES, get_greeting
from app.customers.default_personas import DEFAULT_PERSONAS
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot


def profile_id(client, name="Sofia Rossi"):
    return next(p for p in client.get("/api/v1/profiles").json() if p["name"] == name)["id"]


@pytest.fixture
def tts_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(voice_service, "synthesize_speech_base64", lambda text, lang="fr": calls.append(lang) or "")
    return calls


def test_english_simulation_greets_and_speaks_in_english(client, tts_calls):
    sim = client.post("/api/v1/simulations", json={"profile_id": profile_id(client), "language": "en"}).json()
    assert sim["language"] == "en"
    assert sim["initial_message"].startswith("Hello") and "Sofia Rossi" in sim["initial_message"]
    assert "Bonjour" not in sim["initial_message"]
    assert tts_calls == ["en"]  # voice synthesized in English

    turn = client.post(f"/api/v1/simulations/{sim['id']}/messages", json={"message": "Hello, how can I help you?"}).json()
    assert "Hello" in turn["customer_response"] and "Bonjour" not in turn["customer_response"]  # demo AI answers in English too
    assert tts_calls == ["en", "en"]


def test_french_stays_the_default(client, tts_calls):
    sim = client.post("/api/v1/simulations", json={"profile_id": profile_id(client)}).json()
    assert sim["language"] == "fr" and sim["initial_message"].startswith("Bonjour")
    assert tts_calls == ["fr"]


def test_unknown_language_is_refused(client):
    assert client.post("/api/v1/simulations", json={"profile_id": profile_id(client), "language": "de"}).status_code == 422


def test_voice_message_is_transcribed_in_the_simulation_language(client, tts_calls, monkeypatch):
    heard = []
    monkeypatch.setattr(voice_service, "transcribe_audio",
                        lambda audio, filename="x", language="fr": heard.append(language) or "Hello, the hotel is available.")
    sim_id = client.post("/api/v1/simulations", json={"profile_id": profile_id(client), "language": "en"}).json()["id"]
    res = client.post(f"/api/v1/simulations/{sim_id}/voice_message",
                      files={"file": ("recording.webm", io.BytesIO(b"fake-audio"), "audio/webm")})
    assert res.status_code == 200 and heard == ["en"]
    assert res.json()["transcription"] == "Hello, the hotel is available."


@pytest.mark.parametrize("key", list(ARCHETYPES))
def test_prompt_and_greeting_follow_the_language(key):
    persona = next(p for p in DEFAULT_PERSONAS if p.archetype == key)
    profile = CustomerProfileResponse(**persona.model_dump(), id="p", created_at="2026-10-23T00:00:00Z")
    state = CustomerStateSnapshot(trust_level=5, interest_level=5, booking_intention=3, current_stage="greetings",
                                  unresolved_objections=[], known_requirements={}, questions_answered=[], provided_info={},
                                  language="en")
    prompt = build_system_prompt(profile, state, "")
    assert "STRICTLY in natural, fluent, lively ENGLISH" in prompt and "écrite en anglais" in prompt
    assert "STRICTEMENT en FRANÇAIS" not in prompt
    assert get_greeting(key, "en") != get_greeting(key, "fr")


def test_advisor_may_speak_tunisian_dialect():
    from app.conversation.guardrails import check_rep_message
    for msg in ["fama zghar maakom?", "9addech t7eb tkhalles ?", "n7ajzelkom tawa ?", "فما صغار معاكم؟"]:
        assert check_rep_message(msg) is None  # derja is a normal advisor message, never blocked
    persona = next(p for p in DEFAULT_PERSONAS if p.archetype == "hesitant")
    profile = CustomerProfileResponse(**persona.model_dump(), id="p", created_at="2026-10-23T00:00:00Z")
    state = CustomerStateSnapshot(trust_level=5, interest_level=5, booking_intention=3, current_stage="greetings",
                                  unresolved_objections=[], known_requirements={}, questions_answered=[], provided_info={})
    prompt = build_system_prompt(profile, state, "")
    assert "zghar / sghar" in prompt and "« fama zghar m3akom ? » = « Vous avez des enfants avec vous ? »" in prompt
