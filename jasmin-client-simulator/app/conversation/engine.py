import copy
import logging
import re
from sqlalchemy.orm import Session
from app.customers.schemas import CustomerProfileResponse, CustomerTurnResponse
from app.conversation.llm_provider import BaseLLMProvider, get_llm_provider, MockLLMProvider
from app.memory.state_manager import StateManager
from app.core.exceptions import LLMProviderException
from app.customers.archetypes import get_archetype, get_greeting
from app.conversation.guardrails import check_customer_reply, check_rep_message, redirect
from app.conversation import decision, scenarios
from app.conversation.prompts import decision_state_for
from app.customers.human import identity_card, load_human, voice_guide
from app.hotels.catalog import profile_region

logger = logging.getLogger("simulator")

# Typographic characters some models emit (non-breaking hyphens, narrow spaces) that look like typos in the chat
_TYPO = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2012": "-", "\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u200b": ""})


def tidy(text: str) -> str:
    return " ".join((text or "").translate(_TYPO).split())


def one_question(text: str) -> str:
    """Keep the first question of a reply and drop the following ones (statements are kept): a real client
    asks one thing at a time."""
    sentences = re.findall(r"[^.!?…]+[.!?…]*", text or "")
    kept, asked = [], False
    for sentence in sentences:
        if sentence.strip().endswith("?"):
            if asked:
                continue
            asked = True
        kept.append(sentence.strip())
    return " ".join(kept) or text


def profile_region_of(sim) -> str:
    return (sim.known_requirements or {}).get("region") or ""


class ConversationEngine:
    def __init__(self, db: Session, llm_provider: BaseLLMProvider = None):
        self.db = db
        self.state_manager = StateManager(db)
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            try:
                self.llm_provider = get_llm_provider()
            except Exception as e:
                logger.warning(f"Could not initialize configured LLM provider ({str(e)}). Falling back to MockLLMProvider.")
                self.llm_provider = MockLLMProvider()

    def process_rep_message(self, simulation_id: str, rep_message: str) -> CustomerTurnResponse:
        # 1. Fetch simulation and customer profile
        sim, profile_model = self.state_manager.get_simulation_with_profile(simulation_id)

        # 2. Build profile schema response object
        profile_schema = CustomerProfileResponse.model_validate(profile_model)

        # 3. Snapshot current state & format history
        current_state = self.state_manager.build_state_snapshot(sim)
        history_formatted = self.state_manager.format_history(sim.messages)
        language = current_state.language

        # Guardrail: off-topic or injection messages never reach the model; the client redirects to its trip
        blocked = check_rep_message(rep_message)
        if blocked:
            logger.info(f"Advisor message blocked ({blocked}) in simulation {simulation_id}")
            turn_response = self.blocked_turn(current_state, blocked)
            self.apply_decision(sim, profile_schema, current_state, turn_response)
            self.resolve_hotel(sim, turn_response, history_formatted, rep_message)
            self.state_manager.record_turn(sim, rep_message, turn_response)
            self.finish_if_ended(sim, profile_schema, turn_response)
            return turn_response

        hotel_catalog = self.build_hotel_context(sim)

        # 4. Generate structured customer response via LLM provider (with fallback safety)
        try:
            turn_response = self.llm_provider.generate_response(
                profile=profile_schema,
                state=current_state,
                history_formatted=history_formatted,
                latest_rep_message=rep_message,
                hotel_catalog=hotel_catalog,
            )
        except LLMProviderException as exc:
            if not isinstance(self.llm_provider, MockLLMProvider):
                logger.warning(f"LLM Provider failed: {str(exc)}. Falling back to MockLLMProvider.")
                fallback_provider = MockLLMProvider()
                turn_response = fallback_provider.generate_response(
                    profile=profile_schema,
                    state=current_state,
                    history_formatted=history_formatted,
                    latest_rep_message=rep_message,
                    hotel_catalog=hotel_catalog
                )
            else:
                raise exc

        turn_response.customer_response = one_question(tidy(turn_response.customer_response))
        turn_response = self.avoid_repeats(current_state, profile_schema, history_formatted, rep_message, hotel_catalog, turn_response)
        if not check_customer_reply(turn_response.customer_response):
            logger.warning(f"Customer reply leaked instructions, replaced (simulation {simulation_id})")
            turn_response.customer_response = redirect(language)

        # 5. The decision engine turns the client's signals into trust, interest, intention and maybe an outcome
        self.apply_decision(sim, profile_schema, current_state, turn_response)

        # 6. Only keep a hotel the advisor really proposed; an invented name keeps the previous choice
        self.resolve_hotel(sim, turn_response, history_formatted, rep_message)

        # 7. Persist turn state and messages in DB, then close the simulation if the client decided
        self.state_manager.record_turn(sim, rep_message, turn_response)
        self.finish_if_ended(sim, profile_schema, turn_response)
        return turn_response

    def avoid_repeats(self, state, profile, history, rep_message, hotel_catalog, turn):
        """The model sometimes asks again what the advisor just answered: one rewrite, told exactly what not to ask."""
        repeated = decision.repeated_questions(decision_state_for(profile, state), turn.customer_response)
        if not repeated or isinstance(self.llm_provider, MockLLMProvider):
            return turn
        note = ("Ta réponse repose une question déjà posée, à laquelle l'agent a répondu : " + " / ".join(f"« {q} »" for q in repeated)
                + ". Réécris ta réplique SANS reposer cette question ni la reformuler : accepte sa réponse (au besoin en disant "
                  "une fois qu'elle ne te satisfait pas complètement), puis parle d'autre chose ou avance vers ta décision. "
                  "Réponds avec le même format JSON.")
        try:
            retry = self.llm_provider.generate_response(profile=profile, state=state, history_formatted=history,
                                                        latest_rep_message=rep_message, hotel_catalog=hotel_catalog, retry_note=note)
        except LLMProviderException:
            return turn
        retry.customer_response = one_question(tidy(retry.customer_response))
        logger.info(f"Client repeated a question ({repeated[0]!r}); reply rewritten")
        return retry

    def apply_decision(self, sim, profile, state, turn: CustomerTurnResponse) -> None:
        ds = copy.deepcopy(decision_state_for(profile, state))
        turn.event = ds.get("events", {}).get(str(ds.get("turn", 0) + 1))
        human = load_human(profile)
        scores = {"trust": state.trust_level, "interest": state.interest_level, "intent": state.booking_intention}
        r = decision.apply(ds, turn.signals, scores, profile.archetype, [h.fact for h in human.hidden])
        decision.remember_questions(ds, turn.customer_response)
        turn.trust_level, turn.interest_level, turn.booking_intention = r["trust"], r["interest"], r["intent"]
        turn.intent_reason, turn.patience_left = r["reason"], r["patience_left"]
        turn.outcome, turn.outcome_reason = r["outcome"], r["outcome_reason"]
        if turn.outcome:
            turn.current_stage = "closing"
            # the engine ended the talk (patience, turns) but the reply did not say goodbye: close it in the client's voice
            if turn.signals.proposed_outcome != turn.outcome and not turn.signals.wants_to_leave and not turn.signals.buying_signal:
                closing = decision.closing_line(turn.outcome, state.language)
                if closing:
                    turn.customer_response = f"{turn.customer_response} {closing}".strip()
            sim.outcome, sim.outcome_reason, sim.status = turn.outcome, turn.outcome_reason, "completed"
        sim.decision_state = ds  # reassign so SQLAlchemy sees the JSON change

    def finish_if_ended(self, sim, profile, turn: CustomerTurnResponse) -> None:
        if turn.outcome:
            from app.conversation.debrief import build_debrief
            turn.debrief = build_debrief(self.db, sim, profile, self.llm_provider)

    @staticmethod
    def blocked_turn(state, reason: str) -> CustomerTurnResponse:
        """The client's answer to an off-topic message: in-character redirect; the decision engine
        counts it as an evasive turn (less trust, less patience)."""
        return CustomerTurnResponse(
            customer_response=redirect(state.language),
            current_stage=state.current_stage,
            trust_level=state.trust_level, interest_level=state.interest_level,
            booking_intention=state.booking_intention,
            detected_rep_quality="poor", addressed_objections=[], new_unresolved_objections=[],
            reasoning=f"Message de l'agent hors sujet, bloqué par le filtre de sécurité : {reason}.",
            signals={"advisor_tone": "evasif", "red_flags": ["message hors sujet"]},
        )

    @staticmethod
    def build_hotel_context(sim) -> str:
        """The hotel the client is considering, among those the advisor proposed (the proposals are in the history)."""
        current = (sim.known_requirements or {}).get("hotel_vise")
        if not current:
            return ""
        return f"Hôtel qui t'intéresse le plus parmi les propositions de l'agent : {current['name']}."

    @staticmethod
    def trip_facts(profile, ds: dict = None) -> list:
        """What the client knows of its own trip when it walks in (region, dates, party)."""
        check_in, check_out = scenarios.trip_dates(ds, profile)
        return [f"Mes dates : du {check_in:%d/%m/%Y} au {check_out:%d/%m/%Y}, {profile.number_of_travelers} voyageur(s)",
                f"La région où je veux aller : {profile_region(profile)}"]

    @staticmethod
    def resolve_hotel(sim, turn_response: CustomerTurnResponse, history: str = "", rep_message: str = "") -> None:
        """Keep the client's preferred hotel only if the advisor named it at some point; else keep the previous one."""
        current = (sim.known_requirements or {}).get("hotel_vise")
        name = (turn_response.hotel_vise or "").strip().strip('"«» ')
        advisor_text = " ".join(line for line in (history or "").splitlines() if line.startswith("Sales Rep:")) + " " + (rep_message or "")
        short = name.lower().removeprefix("l'hôtel ").removeprefix("hôtel ").removeprefix("le ").strip()
        if name and len(short) >= 3 and short in advisor_text.lower():
            details = {"name": name, "city": profile_region_of(sim)}
        else:
            if name:
                logger.info(f"Ignored a hotel the advisor never proposed: {name!r}")
            details = current
        turn_response.hotel_vise_details = details
        turn_response.hotel_vise = details["name"] if details else None

    def generate_opening(self, profile, hotel: dict = None, language: str = "fr", ds: dict = None) -> str:
        """First words of the client, generated locally so a provider outage never delays call startup."""
        ds = ds or {}
        scenario = ds.get("scenario", scenarios.DEFAULT_SCENARIO)
        if scenario == scenarios.DEFAULT_SCENARIO:
            return one_question(self.generate_initial_greeting(profile, hotel, language))
        if scenario == "reclamation":  # the complaint alone: the new trip comes once the client has been heard
            return scenarios.SCENARIOS[scenario]["opening_en" if language == "en" else "opening_fr"]
        opener = scenarios.SCENARIOS[scenario]["opening_en" if language == "en" else "opening_fr"]
        name = load_human(profile).identity.name or profile.name
        region = profile_region(profile)
        intro = (f"I'm {name}, it's for a stay in the {region} area." if language == "en"
                 else f"Je suis {name}, c'est pour un séjour dans la région de {region}.")
        return f"{opener} {intro}"

    def opening_prompt(self, profile, hotel, language: str, ds: dict) -> str:
        human = load_human(profile)
        archetype = get_archetype(profile.archetype)
        hotel_line = (f"Tu veux séjourner dans la région de {profile_region(profile)}. Tu n'as AUCUN hôtel en tête : "
                      "c'est au conseiller de t'en proposer. Ne cite aucun nom d'hôtel ni aucun prix d'hôtel.")
        lang = "anglais" if language == "en" else "français"
        return (
            f"Tu es {human.identity.name or profile.name}, un vrai client (pas une IA).\n{identity_card(profile, human)}\n"
            f"Ton type : {archetype['label']}. {archetype['behavior']}\n"
            f"Ta façon de parler :\n{voice_guide(human)}\n"
            f"Exemples du ton (ne pas recopier) : " + " / ".join(archetype["human"]["examples"][:2]) + "\n"
            f"Situation : {scenarios.SCENARIOS[ds.get('scenario', scenarios.DEFAULT_SCENARIO)]['situation']}\n"
            f"Ce que tu veux obtenir dans cette conversation : {scenarios.SCENARIOS[ds.get('scenario', scenarios.DEFAULT_SCENARIO)]['goal']}.\n"
            f"Humeur : {ds.get('mood_text', '')}\n{hotel_line}\n"
            "Voyage : du {:%d/%m/%Y} au {:%d/%m/%Y}, ".format(*scenarios.trip_dates(ds, profile))
            + f"en {ds.get('season_text', '').split(' :')[0] or 'saison normale'}, "
            + f"{profile.number_of_travelers} voyageur(s), budget {profile.budget_total:.0f} TND (parle toujours en dinars ; "
            "cite tes dates avec l'année et n'en invente jamais d'autres).\n\n"
            f"Écris ton tout premier message au conseiller, en {lang} parlé : 1 ou 2 phrases, une salutation et UN SEUL SUJET : "
            f"{scenarios.SCENARIOS[ds.get('scenario', scenarios.DEFAULT_SCENARIO)]['first_topic']}. "
            "Rien d'autre : le reste (budget, nombre de voyageurs, envies, autres demandes) viendra dans tes messages suivants, "
            "un sujet à la fois, ou quand l'agent te posera la question. Au plus une question. Le conseiller ne t'a encore RIEN "
            "proposé : ne demande pas de confirmer un prix, une formule ou une disponibilité. Ne fais jamais référence à une "
            "information que tu n'as pas donnée (pas de « ces dates » sans les avoir dites). Pas de formule toute faite "
            "(« Je souhaiterais », « Pourriez-vous »). Écris dans un français correct (orthographe, accords, ponctuation). "
            "Réponds uniquement par ta réplique, sans guillemets."
        )

    def generate_initial_greeting(self, profile, hotel: dict = None, language: str = "fr") -> str:
        """Template opening built around the client's tourist region. `language` is the simulation language (fr | en)."""
        en = language == "en"
        region = profile_region(profile)
        if region != "Tunisie":
            dest_str = f"the {region} area" if en else f"la région de {region}"
        else:
            dest_str = "an exceptional stay in Tunisia" if en else "un séjour d'exception en Tunisie"
        greeting = get_greeting(profile.archetype, language).format(
            name=profile.name,
            dest_str=dest_str,
            budget=int(profile.budget_total) if profile.budget_total else 1500,
            travelers=profile.number_of_travelers,
        )
        return greeting
