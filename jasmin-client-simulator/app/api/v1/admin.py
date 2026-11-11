from collections import defaultdict
from datetime import date, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, selectinload
from app.database.session import get_db
from app.database.models import CustomerProfileModel, SimulationModel, UserModel
from app.api.v1.auth import require_admin
from app.customers.archetypes import ARCHETYPES, normalize_archetype
from app.conversation.decision import OUTCOMES
from app.conversation.scenarios import SCENARIOS

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])

METRICS = ["trust_level", "interest_level", "booking_intention"]
QUALITIES = ["poor", "neutral", "good", "excellent"]
CONVERSION_THRESHOLD = 7.0  # final booking intention counted as a "won" simulation
MAX_TURNS = 15
DAILY_WINDOW_DAYS = 30


def _avg(values):
    return round(sum(values) / len(values), 2) if values else None


def is_won(s) -> bool:
    """Booked or option held; simulations from before the decision engine fall back to the intention threshold."""
    if s.outcome:
        return s.outcome in ("reserve", "option")
    return s.booking_intention >= CONVERSION_THRESHOLD


def outcome_counts(sims) -> dict:
    counts = dict.fromkeys(OUTCOMES, 0)
    for s in sims:
        if s.outcome in counts:
            counts[s.outcome] += 1
    return counts


@router.get("/overview")
def get_overview(db: Session = Depends(get_db)):
    # ponytail: aggregates in Python over every row; move to SQL GROUP BYs if simulations reach the tens of thousands
    profiles = db.query(CustomerProfileModel).all()
    sims = db.query(SimulationModel).options(selectinload(SimulationModel.messages)).all()
    archetype_of = {p.id: normalize_archetype(p.archetype) for p in profiles}
    login_of = dict(db.query(UserModel.profile_id, UserModel.email).filter(UserModel.profile_id.isnot(None)).all())

    sims_by_arch = defaultdict(list)
    sims_by_profile = defaultdict(list)
    for s in sims:
        sims_by_arch[normalize_archetype(archetype_of.get(s.profile_id))].append(s)
        sims_by_profile[s.profile_id].append(s)

    archetypes = []
    curves = {m: {} for m in METRICS}
    curve_counts = {}
    for arch in ARCHETYPES:
        arch_sims = sims_by_arch[arch]
        quality = dict.fromkeys(QUALITIES, 0)
        per_turn = defaultdict(lambda: defaultdict(list))  # metric -> turn -> values
        for s in arch_sims:
            customer_msgs = [m for m in s.messages if m.sender == "customer"][:MAX_TURNS + 1]
            for turn, msg in enumerate(customer_msgs):
                snap = msg.state_snapshot or {}
                for metric in METRICS:
                    if metric in snap:
                        per_turn[metric][turn].append(snap[metric])
                if snap.get("detected_quality") in quality:
                    quality[snap["detected_quality"]] += 1

        for metric in METRICS:
            turns = per_turn[metric]
            curves[metric][arch] = [_avg(turns[t]) for t in range(max(turns, default=-1) + 1)]
        trust_turns = per_turn["trust_level"]
        curve_counts[arch] = [len(trust_turns[t]) for t in range(max(trust_turns, default=-1) + 1)]

        archetypes.append({
            "archetype": arch,
            "label": ARCHETYPES[arch]["label"],
            "color": ARCHETYPES[arch]["color"],
            "difficulty": ARCHETYPES[arch]["difficulty"],
            "clients": sum(1 for a in archetype_of.values() if a == arch),
            "simulations": len(arch_sims),
            "completed": sum(1 for s in arch_sims if s.status == "completed"),
            "avg_trust": _avg([s.trust_level for s in arch_sims]),
            "avg_interest": _avg([s.interest_level for s in arch_sims]),
            "avg_intent": _avg([s.booking_intention for s in arch_sims]),
            "conversion_rate": _avg([1.0 if is_won(s) else 0.0 for s in arch_sims]),
            "outcomes": outcome_counts(arch_sims),
            "quality": quality,
        })

    # Simulations started per day, last DAILY_WINDOW_DAYS days
    today = date.today()
    days = [today - timedelta(days=i) for i in range(DAILY_WINDOW_DAYS - 1, -1, -1)]
    daily = {arch: dict.fromkeys(days, 0) for arch in ARCHETYPES}
    for s in sims:
        d = s.created_at.date() if s.created_at else None
        arch = normalize_archetype(archetype_of.get(s.profile_id))
        if d in daily[arch]:
            daily[arch][d] += 1

    return {
        "totals": {
            "clients": len(profiles),
            "twins": sum(1 for p in profiles if (p.scenario_id or "").startswith("twin_")),
            "simulations": len(sims),
            "avg_trust": _avg([s.trust_level for s in sims]),
            "conversion_rate": _avg([1.0 if is_won(s) else 0.0 for s in sims]),
            "outcomes": outcome_counts(sims),
        },
        "outcome_labels": OUTCOMES,
        "scenarios": [{"key": k, "label": v["label"], "simulations": sum(1 for s in sims if s.scenario == k),
                       "conversion_rate": _avg([1.0 if is_won(s) else 0.0 for s in sims if s.scenario == k])}
                      for k, v in SCENARIOS.items()],
        "conversion_threshold": CONVERSION_THRESHOLD,
        "archetypes": archetypes,
        "curves": curves,
        "curve_counts": curve_counts,
        "daily": {"dates": [d.isoformat() for d in days], **{a: list(daily[a].values()) for a in ARCHETYPES}},
        "clients": [
            {
                "id": p.id,
                "name": p.name,
                "archetype": archetype_of[p.id],
                "is_twin": (p.scenario_id or "").startswith("twin_"),
                "login_email": login_of.get(p.id),
                "budget_total": p.budget_total,
                "simulations": len(sims_by_profile[p.id]),
                "avg_final_trust": _avg([s.trust_level for s in sims_by_profile[p.id]]),
                "avg_final_intent": _avg([s.booking_intention for s in sims_by_profile[p.id]]),
            }
            for p in profiles
        ],
    }
