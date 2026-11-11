"""Enumerations shared across the Workforce Manager domain."""

from __future__ import annotations

from enum import Enum


class RoutingAction(str, Enum):
    """The four routing actions available to the Workforce Manager.

    The declaration order is the canonical action index order (used by masks
    and, later, by the RL action space).
    """

    AI = "AI"
    HUMAN = "HUMAN"
    WAIT = "WAIT"
    CALLBACK = "CALLBACK"

    @property
    def index(self) -> int:
        return ACTION_ORDER.index(self)


ACTION_ORDER: tuple[RoutingAction, ...] = tuple(RoutingAction)


class RequestType(str, Enum):
    FAQ = "FAQ"
    BOOKING = "BOOKING"
    CHANGE_CANCEL = "CHANGE_CANCEL"
    COMPLAINT_INCIDENT = "COMPLAINT_INCIDENT"


class Urgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Sensitivity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Channel(str, Enum):
    PHONE = "PHONE"
    CHAT = "CHAT"
    WHATSAPP = "WHATSAPP"
    EMAIL = "EMAIL"


class LanguageDialect(str, Enum):
    FR = "FR"
    EN = "EN"
    AR_MSA = "AR_MSA"
    AR_DIALECT = "AR_DIALECT"


class DayType(str, Enum):
    WEEKDAY = "WEEKDAY"
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"


class ContactState(str, Enum):
    """Simulator-side lifecycle state of a contact (never observed directly)."""

    PENDING_DECISION = "PENDING_DECISION"
    WAITING = "WAITING"  # deliberately deferred by a WAIT action
    HUMAN_QUEUE = "HUMAN_QUEUE"
    HUMAN_SERVICE = "HUMAN_SERVICE"
    AI_QUEUE = "AI_QUEUE"
    AI_SERVICE = "AI_SERVICE"
    CALLBACK_QUEUE = "CALLBACK_QUEUE"
    CALLBACK_ATTEMPT = "CALLBACK_ATTEMPT"
    CALLBACK_SERVICE = "CALLBACK_SERVICE"
    AWAITING_FALSE_DISCOVERY = "AWAITING_FALSE_DISCOVERY"  # looks resolved, is not
    TERMINAL = "TERMINAL"


# States in which the customer is actively waiting online (patience runs, waiting cost accrues).
ONLINE_WAITING_STATES = frozenset({ContactState.WAITING, ContactState.HUMAN_QUEUE, ContactState.AI_QUEUE})
# States in which a callback is pending (callback-pending cost accrues).
CALLBACK_PENDING_STATES = frozenset({ContactState.CALLBACK_QUEUE, ContactState.CALLBACK_ATTEMPT})
# States in which a human advisor is busy with the contact.
HUMAN_BUSY_STATES = frozenset(
    {ContactState.HUMAN_SERVICE, ContactState.CALLBACK_ATTEMPT, ContactState.CALLBACK_SERVICE}
)
# States in which an AI session is busy with the contact.
AI_BUSY_STATES = frozenset({ContactState.AI_SERVICE})
# Waiting states in which a passing deadline becomes a terminal missed deadline.
DEADLINE_SENSITIVE_STATES = ONLINE_WAITING_STATES | frozenset({ContactState.CALLBACK_QUEUE})


class Outcome(str, Enum):
    """Terminal outcome of a contact. Assigned exactly once."""

    RESOLVED_AI = "RESOLVED_AI"
    RESOLVED_HUMAN = "RESOLVED_HUMAN"
    RESOLVED_CALLBACK = "RESOLVED_CALLBACK"
    FALSE_AI_DISCOVERED = "FALSE_AI_DISCOVERED"
    FALSE_AI_UNDISCOVERED = "FALSE_AI_UNDISCOVERED"  # only when settle_at_horizon is false
    ABANDONED = "ABANDONED"
    MISSED_DEADLINE = "MISSED_DEADLINE"
    CALLBACK_FAILED = "CALLBACK_FAILED"  # callback reason MAX_ATTEMPTS
    CALLBACK_WINDOW_EXPIRED = "CALLBACK_WINDOW_EXPIRED"  # callback reason WINDOW_EXPIRED (B1.5)
    CALLBACK_DROPOUT = "CALLBACK_DROPOUT"  # callback reason CUSTOMER_DROPOUT (B1.5): a lost customer
    UNFINISHED = "UNFINISHED"

    @property
    def is_correct_resolution(self) -> bool:
        return self in (Outcome.RESOLVED_AI, Outcome.RESOLVED_HUMAN, Outcome.RESOLVED_CALLBACK)


class CallbackReason(str, Enum):
    """Terminal reason of a contact that entered the callback path (one per callback contact)."""

    CALLBACK_SUCCESS = "CALLBACK_SUCCESS"  # customer reached and served (resolved)
    CUSTOMER_DROPOUT = "CUSTOMER_DROPOUT"  # gave up while pending
    MAX_ATTEMPTS = "MAX_ATTEMPTS"  # all dial attempts unanswered
    WINDOW_EXPIRED = "WINDOW_EXPIRED"  # completion window ended during operating time
    MISSED_DEADLINE = "MISSED_DEADLINE"  # request deadline passed while pending
    UNFINISHED = "UNFINISHED"  # still pending (or in service) at the end of the day


class WakeCause(str, Enum):
    """Why a WAIT contact was brought back to the routing decision point."""

    TIMER = "TIMER"
    HUMAN_AVAILABLE = "HUMAN_AVAILABLE"


class EventType(str, Enum):
    ARRIVAL = "ARRIVAL"
    INTAKE_PREDICTION = "INTAKE_PREDICTION"
    ROUTING_DECISION = "ROUTING_DECISION"
    WAIT_STARTED = "WAIT_STARTED"
    WAIT_REEVALUATION = "WAIT_REEVALUATION"
    HUMAN_QUEUE_ENTERED = "HUMAN_QUEUE_ENTERED"
    HUMAN_SERVICE_STARTED = "HUMAN_SERVICE_STARTED"
    HUMAN_SERVICE_COMPLETED = "HUMAN_SERVICE_COMPLETED"
    AI_QUEUE_ENTERED = "AI_QUEUE_ENTERED"
    AI_ATTEMPT_STARTED = "AI_ATTEMPT_STARTED"
    AI_RESOLVED = "AI_RESOLVED"
    AI_VISIBLE_FAILURE = "AI_VISIBLE_FAILURE"
    AI_ESCALATION = "AI_ESCALATION"
    AI_FALSE_RESOLUTION = "AI_FALSE_RESOLUTION"  # simulator truth: looks resolved, is wrong
    FALSE_AI_DISCOVERED = "FALSE_AI_DISCOVERED"
    FALSE_AI_SETTLED_AT_HORIZON = "FALSE_AI_SETTLED_AT_HORIZON"
    RECONTACT_SCHEDULED = "RECONTACT_SCHEDULED"
    RECONTACT_NOT_CREATED = "RECONTACT_NOT_CREATED"  # customer does not return, or returns after the horizon
    CALLBACK_SCHEDULED = "CALLBACK_SCHEDULED"
    CALLBACK_ATTEMPT_STARTED = "CALLBACK_ATTEMPT_STARTED"
    CALLBACK_ATTEMPT_FAILED = "CALLBACK_ATTEMPT_FAILED"
    CALLBACK_REACHED = "CALLBACK_REACHED"
    CALLBACK_FAILED = "CALLBACK_FAILED"  # MAX_ATTEMPTS
    CALLBACK_WINDOW_EXPIRED = "CALLBACK_WINDOW_EXPIRED"
    CALLBACK_DROPOUT = "CALLBACK_DROPOUT"
    ABANDONMENT = "ABANDONMENT"
    DEADLINE_MISSED = "DEADLINE_MISSED"
    CONTACT_RESOLVED = "CONTACT_RESOLVED"
    UNFINISHED_AT_HORIZON = "UNFINISHED_AT_HORIZON"
    REWARD = "REWARD"
