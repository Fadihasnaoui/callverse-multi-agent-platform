"""Contact models.

The split between these classes is the core anti-leakage boundary:

* ``GroundTruthContact`` — hidden simulator truth. Only the Digital Twin and the
  reward calculation may read it.
* ``ObservedContactFeatures`` — what the intake/analysis layer *predicts* about
  the contact. This is the only per-contact input to the routing observation.
* ``ContactRuntimeState`` — mutable simulator bookkeeping for a live contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from callverse_workforce.domain.enums import (
    CallbackReason,
    Channel,
    ContactState,
    LanguageDialect,
    Outcome,
    RequestType,
    RoutingAction,
    Sensitivity,
    Urgency,
)


@dataclass(frozen=True)
class GroundTruthContact:
    """Hidden truth about one contact. Never given to a policy."""

    contact_id: str
    index: int
    arrival_time: float  # simulated minutes since day start
    request_type: RequestType
    urgency: Urgency
    sensitivity: Sensitivity
    channel: Channel
    language_dialect: LanguageDialect
    request_value: float  # normalised in [0, 1]
    deadline_offset_minutes: float  # true deadline = arrival_time + offset
    patience_minutes: float  # cumulative online waiting tolerated before abandoning
    human_service_minutes: float
    ai_service_minutes: float
    ai_correct_prob: float  # P(AI resolves correctly)
    ai_false_resolution_prob: float  # P(AI appears to resolve but is wrong)
    false_discovery_delay_minutes: float  # delay until a false AI resolution surfaces
    callback_allowed: bool
    human_requested: bool
    # Lineage. Exogenous contacts are their own root (depth 0). A recontact created after a
    # discovered false AI resolution has depth 1 and shares the root's ``index`` (RNG key).
    root_contact_id: str | None = None
    parent_contact_id: str | None = None
    recontact_depth: int = 0

    def __post_init__(self) -> None:
        if self.root_contact_id is None:
            object.__setattr__(self, "root_contact_id", self.contact_id)
        if (self.recontact_depth == 0) != (self.parent_contact_id is None):
            raise ValueError("parent_contact_id must be set exactly when recontact_depth > 0")

    @property
    def rng_key(self) -> tuple[int, int]:
        """(root index, depth): identifies this instance's per-contact RNG streams."""
        return (self.index, self.recontact_depth)

    @property
    def deadline_at(self) -> float:
        return self.arrival_time + self.deadline_offset_minutes

    @property
    def value_weight(self) -> float:
        """``V_i = 1 + request_value_i`` used by the reward."""
        return 1.0 + self.request_value


@dataclass(frozen=True)
class ObservedContactFeatures:
    """Predicted/observable per-contact features produced by ``analyze_request``."""

    contact_id: str
    observed_at: float
    request_type: RequestType  # predicted
    urgency_pred: Urgency
    sensitivity_pred: Sensitivity
    channel: Channel  # directly observable
    language_dialect: LanguageDialect  # directly observable
    request_value: float  # observable from CRM/booking data (V1 assumption)
    predicted_deadline_at: float  # absolute simulated minute
    callback_allowed: bool  # observable business fact
    human_requested: bool  # observable customer request
    ai_resolution_score: float  # predicted AI resolvability in [0, 1]


@dataclass
class ContactRuntimeState:
    """Mutable simulator bookkeeping for one contact."""

    truth: GroundTruthContact
    observed: ObservedContactFeatures
    state: ContactState = ContactState.PENDING_DECISION
    state_entered_at: float = 0.0
    online_wait_accumulated: float = 0.0
    routing_attempt_count: int = 0  # number of routing decisions already taken
    wait_count: int = 0
    actions: list[RoutingAction] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    escalated: bool = False
    callback_started_at: float | None = None  # callback creation time
    callback_attempts: int = 0
    callback_attempt_times: list[float] = field(default_factory=list)
    callback_completed_at: float | None = None  # reached (success) or terminal time
    callback_reason: CallbackReason | None = None
    callback_pending_minutes: float = 0.0
    human_queue_minutes: float = 0.0
    human_direct_minutes: float = 0.0  # advisor minutes in HUMAN_SERVICE (direct or escalated)
    callback_human_minutes: float = 0.0  # advisor minutes dialling and serving callbacks
    human_touched: bool = False  # HIR definition, see tracing.summary.human_intervention_rates
    effective_patience: float | None = None  # V1.1 frustration: overrides truth patience once escalated
    handoff_minutes: float = 0.0  # V1.1 handoff overhead added to this contact's human service
    outcome: Outcome | None = None
    terminal_time: float | None = None

    @property
    def contact_id(self) -> str:
        return self.truth.contact_id

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not None
