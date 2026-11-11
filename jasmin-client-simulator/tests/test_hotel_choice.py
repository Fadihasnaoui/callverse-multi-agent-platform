from app.conversation.engine import ConversationEngine
from app.conversation.llm_provider import BaseLLMProvider
from app.conversation.prompts import build_system_prompt
from app.customers.default_personas import DEFAULT_PERSONAS
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot, CustomerTurnResponse
from app.database.models import CustomerProfileModel, SimulationModel
from app.hotels.catalog import is_tunisian_place, profile_region, trip_nights


class ScriptedProvider(BaseLLMProvider):
    """Stands in for the LLM: returns a fixed hotel choice and records the hotel context it received."""
    def __init__(self, hotel_vise):
        self.hotel_vise = hotel_vise
        self.seen_context = None

    def generate_response(self, profile, state, history_formatted, latest_rep_message, hotel_catalog=""):
        self.seen_context = hotel_catalog
        return CustomerTurnResponse(
            customer_response="D'accord, parlons de cet hôtel.", current_stage="recommendation",
            detected_rep_quality="good", addressed_objections=[], new_unresolved_objections=[], reasoning="test",
            hotel_vise=self.hotel_vise,
        )


def start(client, name="Sofia Rossi"):
    profile = next(p for p in client.get("/api/v1/profiles").json() if p["name"] == name)
    return client.post("/api/v1/simulations", json={"profile_id": profile["id"]}).json()


def test_client_starts_with_a_tourist_region_and_no_hotel(client):
    sim = start(client)  # Sofia (hesitant) wants Djerba
    assert sim["known_requirements"]["hotel_vise"] is None and sim["known_requirements"]["region"] == "Djerba"
    assert "la région de Djerba" in sim["initial_message"] and "Radisson" not in sim["initial_message"]


def test_client_only_keeps_hotels_the_advisor_proposed(client, db_session):
    sim_id = start(client)["id"]
    turn = ConversationEngine(db_session, llm_provider=ScriptedProvider("Royal Azur")).process_rep_message(
        sim_id, "Je vous propose le Royal Azur à Hammamet, 4 étoiles en demi-pension.")
    assert turn.hotel_vise == "Royal Azur" and turn.hotel_vise_details["name"] == "Royal Azur"
    db_session.expire_all()
    assert db_session.get(SimulationModel, sim_id).known_requirements["hotel_vise"]["name"] == "Royal Azur"

    # A hotel the advisor never named is refused: the previous choice stays, and is given back to the client
    provider = ScriptedProvider("Hôtel Imaginaire Paradise")
    turn = ConversationEngine(db_session, llm_provider=provider).process_rep_message(sim_id, "Et pour la chambre ?")
    assert turn.hotel_vise == "Royal Azur" and "Royal Azur" in provider.seen_context


def test_greeting_names_the_region_and_no_hotel(client, db_session):
    family = db_session.query(CustomerProfileModel).filter(CustomerProfileModel.scenario_id == "family_7").first()
    greeting = client.post("/api/v1/simulations", json={"profile_id": family.id}).json()["initial_message"]
    assert "la région de Djerba" in greeting
    assert not any(w.split(" (")[0] in greeting for w in family.destination_preferences)


def test_prompt_knows_the_region_but_no_hotel_list():
    persona = next(p for p in DEFAULT_PERSONAS if p.scenario_id == "family_7")
    profile = CustomerProfileResponse(**persona.model_dump(), id="p", created_at="2026-10-18T00:00:00Z")
    state = CustomerStateSnapshot(trust_level=5, interest_level=5, booking_intention=3, current_stage="greetings",
                                  unresolved_objections=[], known_requirements={}, questions_answered=[], provided_info={})
    prompt = build_system_prompt(profile, state, "")
    assert profile_region(profile) == "Djerba" and "Houmt Souk" in prompt  # what a traveller knows of Djerba
    assert "Ozone" not in prompt and "Djerba Explore Park (Lella Hadhria)" not in prompt


def test_places_helpers():
    assert is_tunisian_place("Yasmine Hammamet") and not is_tunisian_place("Cancun")
    assert trip_nights({"start_date": "2026-11-10", "end_date": "2026-11-17"}) == 7


class RepeatingProvider(ScriptedProvider):
    """First answer repeats the price question; the rewrite (with the correction note) moves on."""
    def __init__(self):
        super().__init__(None)
        self.notes = []

    def generate_response(self, profile, state, history_formatted, latest_rep_message, hotel_catalog="", retry_note=""):
        self.notes.append(retry_note)
        text = ("Bon, j'aurais préféré un chiffre, mais d'accord. Et le transfert depuis l'aéroport, il est compris ?" if retry_note
                else "Ok, mais je voudrais connaître le tarif exact pour m'assurer que ça rentre dans le budget.")
        return CustomerTurnResponse(customer_response=text, current_stage="negotiation", detected_rep_quality="neutral",
                                    addressed_objections=[], new_unresolved_objections=[], reasoning="test")


def test_client_does_not_ask_again_what_the_advisor_answered(client, db_session):
    from app.conversation.decision import remember_questions
    sim_id = start(client)["id"]
    sim = db_session.get(SimulationModel, sim_id)
    ds = dict(sim.decision_state)
    remember_questions(ds, "J'aimerais connaître le tarif exact, s'il vous plaît, pour m'assurer que ça rentre dans mon budget.")
    sim.decision_state = ds
    db_session.commit()

    provider = RepeatingProvider()
    turn = ConversationEngine(db_session, llm_provider=provider).process_rep_message(
        sim_id, "Je n'ai pas le prix exact, mais je vous assure que ça rentre dans votre budget.")
    assert len(provider.notes) == 2 and "tarif exact" in provider.notes[1]  # one rewrite, told what not to ask
    assert turn.customer_response.startswith("Bon, j'aurais préféré un chiffre")
    db_session.expire_all()
    asked = [a["q"] for a in db_session.get(SimulationModel, sim_id).decision_state["asked"]]
    assert any("transfert" in q for q in asked)  # the new question is remembered for next turns
