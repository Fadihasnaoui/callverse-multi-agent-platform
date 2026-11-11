import re
import unicodedata
from typing import List, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database.models import SimulationModel, CustomerProfileModel, ConversationMessageModel
from app.customers.schemas import CustomerStateSnapshot, CustomerTurnResponse
from app.core.exceptions import SimulationNotFoundException, ProfileNotFoundException

def utcnow():
    return datetime.now(timezone.utc)


MAX_ASKS = 2  # customer messages mentioning the same point before it is closed


def _keywords(text: str) -> set:
    plain = unicodedata.normalize("NFKD", (text or "").lower()).encode("ascii", "ignore").decode()
    # crude stem (first 5 letters): "annuler" and "annulation" are the same point
    return {w[:5] for w in re.findall(r"[a-z]+", plain) if len(w) >= 4}


def same_topic(a: str, b: str) -> bool:
    """Two objections are the same point if half the words of the shorter one appear in the other."""
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return (a or "").strip().lower() == (b or "").strip().lower()
    return len(ka & kb) / min(len(ka), len(kb)) >= 0.5

class StateManager:
    def __init__(self, db: Session):
        self.db = db

    def get_simulation_with_profile(self, simulation_id: str) -> Tuple[SimulationModel, CustomerProfileModel]:
        sim = self.db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
        if not sim:
            raise SimulationNotFoundException(f"Simulation session {simulation_id} not found.")

        profile = self.db.query(CustomerProfileModel).filter(CustomerProfileModel.id == sim.profile_id).first()
        if not profile:
            raise ProfileNotFoundException(f"Customer profile {sim.profile_id} not found.")

        return sim, profile

    def build_state_snapshot(self, sim: SimulationModel) -> CustomerStateSnapshot:
        return CustomerStateSnapshot(
            trust_level=sim.trust_level,
            interest_level=sim.interest_level,
            booking_intention=sim.booking_intention,
            current_stage=sim.current_stage,
            unresolved_objections=sim.unresolved_objections or [],
            known_requirements=sim.known_requirements or {},
            questions_answered=sim.questions_answered or [],
            provided_info=sim.provided_info or {},
            language=getattr(sim, "language", None) or "fr",
            decision_state=getattr(sim, "decision_state", None) or {}
        )

    MAX_HISTORY_MESSAGES = 16  # ponytail: older middle turns are dropped (session memory keeps their facts); summarise if talks get much longer

    def format_history(self, messages: List[ConversationMessageModel]) -> str:
        formatted = []
        messages = list(messages)
        if len(messages) > self.MAX_HISTORY_MESSAGES:
            formatted.append(f"Customer: {messages[0].content}")
            formatted.append(f"(... {len(messages) - self.MAX_HISTORY_MESSAGES + 1} messages plus anciens, voir la mémoire de la session ...)")
            messages = messages[-(self.MAX_HISTORY_MESSAGES - 1):]
        for msg in messages:
            role = "Sales Rep" if msg.sender == "sales_rep" else "Customer"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def record_turn(
        self,
        sim: SimulationModel,
        rep_message: str,
        turn_response: CustomerTurnResponse
    ) -> Tuple[ConversationMessageModel, ConversationMessageModel]:
        # Record Rep Message
        rep_msg = ConversationMessageModel(
            simulation_id=sim.id,
            sender="sales_rep",
            content=rep_message,
            stage_at_message=sim.current_stage,
            state_snapshot={
                "trust_level": sim.trust_level,
                "interest_level": sim.interest_level,
                "booking_intention": sim.booking_intention
            },
            timestamp=utcnow()
        )
        self.db.add(rep_msg)

        # Update Simulation State
        sim.current_stage = turn_response.current_stage
        sim.trust_level = turn_response.trust_level
        sim.interest_level = turn_response.interest_level
        sim.booking_intention = turn_response.booking_intention

        # Objections management: the LLM rewords objections, so match them by topic, not by exact text
        current_unresolved = list(sim.unresolved_objections or [])
        resolved = list(sim.questions_answered or [])
        for obj in turn_response.addressed_objections:
            matches = [o for o in current_unresolved if same_topic(o, obj)]
            for o in matches:
                current_unresolved.remove(o)
            for o in matches or [obj]:
                if not any(same_topic(o, r) for r in resolved):
                    resolved.append(o)
        for obj in turn_response.new_unresolved_objections:
            # never reopen a point the agent already settled, never duplicate an open one
            if not any(same_topic(obj, o) for o in current_unresolved + resolved):
                current_unresolved.append(obj)
        # A point the client already raised in MAX_ASKS of its messages is closed, answered or not:
        # the model does not reliably follow "never ask twice", so convergence is enforced here
        said = [m.content for m in getattr(sim, "messages", None) or [] if m.sender == "customer"]
        said.append(turn_response.customer_response)
        for obj in list(current_unresolved):
            if sum(same_topic(obj, text) for text in said) >= MAX_ASKS:
                current_unresolved.remove(obj)
                resolved.append(f"{obj} (sujet clos, ne plus le reposer)")
        sim.unresolved_objections = current_unresolved
        sim.questions_answered = resolved
        if turn_response.hotel_vise_details:
            # reassign (not mutate) so SQLAlchemy sees the JSON column change
            sim.known_requirements = {**(sim.known_requirements or {}), "hotel_vise": turn_response.hotel_vise_details}
        sim.updated_at = utcnow()

        # Record Customer Response Message
        cust_msg = ConversationMessageModel(
            simulation_id=sim.id,
            sender="customer",
            content=turn_response.customer_response,
            stage_at_message=turn_response.current_stage,
            state_snapshot={
                "trust_level": sim.trust_level,
                "interest_level": sim.interest_level,
                "booking_intention": sim.booking_intention,
                "detected_quality": turn_response.detected_rep_quality,
                "hotel_vise": turn_response.hotel_vise,
                "intent_reason": turn_response.intent_reason,
                "event": turn_response.event,
                "outcome": turn_response.outcome
            },
            timestamp=utcnow()
        )
        self.db.add(cust_msg)

        self.db.commit()
        self.db.refresh(sim)
        self.db.refresh(rep_msg)
        self.db.refresh(cust_msg)

        return rep_msg, cust_msg
