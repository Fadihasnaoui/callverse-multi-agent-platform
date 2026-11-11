import json
import pytest
from types import SimpleNamespace
from app.conversation.llm_provider import GroqLLMProvider
from app.core.exceptions import LLMProviderException
from app.customers.default_personas import DEFAULT_PERSONAS
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot

FINAL_JSON = {
    "customer_response": "Le Royal Azur m'intéresse, il est dans mon budget ?",
    "current_stage": "recommendation", "detected_rep_quality": "good",
    "signals": {"advisor_tone": "chaleureux"}, "addressed_objections": [], "new_unresolved_objections": [],
    "reasoning": "Bonne proposition.", "hotel_vise": "Royal Azur",
}


class FakeGroqClient:
    """Plays a scripted sequence of model messages (or errors) and records every request."""
    def __init__(self, messages):
        self.queue = list(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=item)])


def provider_with(messages):
    provider = GroqLLMProvider(api_key="test-key", model="test-model")
    provider.client = FakeGroqClient(messages)
    return provider


def answer(content):
    return SimpleNamespace(content=content, tool_calls=None)


def profile_and_state():
    persona = next(p for p in DEFAULT_PERSONAS if p.archetype == "hesitant")
    profile = CustomerProfileResponse(**persona.model_dump(), id="p", created_at="2026-10-21T00:00:00Z")
    state = CustomerStateSnapshot(trust_level=5, interest_level=5, booking_intention=3, current_stage="recommendation",
                                  unresolved_objections=[], known_requirements={}, questions_answered=[], provided_info={})
    return profile, state


def test_one_json_call_per_turn_and_no_tools():
    provider = provider_with([answer("```json\n" + json.dumps(FINAL_JSON) + "\n```")])
    profile, state = profile_and_state()
    turn = provider.generate_response(profile, state, "", "Je vous propose le Royal Azur à Hammamet.")
    (request,) = provider.client.requests
    assert request["response_format"] == {"type": "json_object"} and "tools" not in request
    assert "Ozone" not in request["messages"][0]["content"] and "chercher_hotels" not in request["messages"][0]["content"]
    assert turn.hotel_vise == "Royal Azur" and turn.signals.advisor_tone == "chaleureux"


def test_unusable_answer_raises_so_the_engine_can_fall_back():
    provider = provider_with([answer("je réfléchis...")])
    profile, state = profile_and_state()
    with pytest.raises(LLMProviderException):
        provider.generate_response(profile, state, "", "Bonjour")


class GroqRateLimited(Exception):
    """Same shape as groq.RateLimitError for the per-minute limit."""
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                         "Limit 8000, Used 3938, Requested 4527. Please try again in 0.01s.")


class GroqDailyQuota(Exception):
    """The 429 Groq sends when a model's tokens per day are used up (as seen in the logs)."""
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                         "Limit 200000, Used 197099, Requested 4647. Please try again in 12m34.272s.")


def test_rate_limit_waits_and_retries_instead_of_failing(monkeypatch):
    import app.conversation.llm_provider as llm
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    provider = provider_with([GroqRateLimited(), answer(json.dumps(FINAL_JSON))])
    profile, state = profile_and_state()
    assert provider.generate_response(profile, state, "", "Bonjour").customer_response == FINAL_JSON["customer_response"]
    assert waits == [0.51]  # Groq's "try again in 0.01s" + margin


def test_rate_limit_gives_up_after_retries(monkeypatch):
    import app.conversation.llm_provider as llm
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    provider = provider_with([GroqRateLimited() for _ in range(GroqLLMProvider.RATE_LIMIT_RETRIES + 1)])
    profile, state = profile_and_state()
    with pytest.raises(LLMProviderException):
        provider.generate_response(profile, state, "", "Bonjour")


def test_daily_quota_switches_to_the_next_model_instead_of_the_demo_client(monkeypatch):
    import app.conversation.llm_provider as llm
    monkeypatch.setattr(llm, "_EXHAUSTED", {})
    provider = provider_with([GroqDailyQuota(), answer(json.dumps(FINAL_JSON))])
    profile, state = profile_and_state()
    assert provider.generate_response(profile, state, "", "Bonjour").customer_response == FINAL_JSON["customer_response"]
    first, second = provider.client.requests
    assert first["model"] == "test-model" and second["model"] == "openai/gpt-oss-20b"
    assert llm._EXHAUSTED["test-model"] > llm.time.time() + 700  # out for ~12 minutes
    assert GroqLLMProvider(api_key="k", model="test-model").model == "openai/gpt-oss-20b"  # next requests skip it
