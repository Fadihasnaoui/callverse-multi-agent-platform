import pytest
from unittest.mock import MagicMock, patch
from app.conversation.llm_provider import GroqLLMProvider, get_llm_provider, MockLLMProvider
from app.core.exceptions import LLMProviderException
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot, TravelDatesSchema

@pytest.fixture
def sample_profile():
    return CustomerProfileResponse(
        id="test-profile-id",
        scenario_id="test_scenario",
        name="Carlos Mendez",
        age_range="28-35",
        persona_description="Budget traveler.",
        destination_preferences=["Sousse"],
        travel_dates=TravelDatesSchema(start_date="2026-11-10", end_date="2026-11-17", flexible=True),
        number_of_travelers=1,
        budget_total=950.0,
        price_sensitivity="high",
        preferred_accommodation="3-star hostel",
        preferred_transportation="Economy flight",
        motivations=["Beach"],
        constraints=["Max $950 USD"],
        personality_traits=["cautious"],
        communication_style="Direct",
        objections=["Too expensive"],
        decision_criteria=["Low price"],
        initial_trust=5.0,
        initial_interest=5.0,
        initial_booking_intention=3.0,
        desired_outcome="Book vacation under $950",
        created_at="2026-10-09T00:00:00Z"
    )

@pytest.fixture
def sample_state():
    return CustomerStateSnapshot(
        trust_level=5.0,
        interest_level=5.0,
        booking_intention=3.0,
        current_stage="greetings",
        unresolved_objections=["Too expensive"],
        known_requirements={"budget": 950.0},
        questions_answered=[],
        provided_info={}
    )

def test_groq_missing_api_key_raises_exception():
    with pytest.raises(LLMProviderException) as exc_info:
        GroqLLMProvider(api_key="")
    assert "Groq API key is missing" in str(exc_info.value)

@patch("groq.Groq")
def test_groq_successful_response(mock_groq_cls, sample_profile, sample_state):
    # Mock Groq Client
    mock_client = MagicMock()
    mock_groq_cls.return_value = mock_client
    
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="""{
            "customer_response": "Bonjour, je cherche un séjour à Sousse sous 950 $.",
            "current_stage": "needs_discovery",
            "trust_level": 5.5,
            "interest_level": 6.0,
            "booking_intention": 3.5,
            "detected_rep_quality": "good",
            "addressed_objections": [],
            "new_unresolved_objections": [],
            "reasoning": "Representative greeted politely."
        }"""))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    
    provider = GroqLLMProvider(api_key="gsk_dummy_test_key", model="llama-3.3-70b-versatile")
    turn_response = provider.generate_response(
        profile=sample_profile,
        state=sample_state,
        history_formatted="",
        latest_rep_message="Hello Carlos!"
    )
    
    assert turn_response.customer_response == "Bonjour, je cherche un séjour à Sousse sous 950 $."
    assert turn_response.current_stage == "needs_discovery"
    assert turn_response.detected_rep_quality == "good"
    assert turn_response.trust_level == 5.5

def test_get_llm_provider_factory():
    provider_mock = get_llm_provider("mock")
    assert isinstance(provider_mock, MockLLMProvider)
