"""
Booking-intention engine: the model reports WHAT HAPPENED in a turn (signals), this module decides
how the client's trust, interest and intention move, when patience runs out, and how the talk ends.

Why in code: the model is a good judge of a single reply but a poor accountant over a whole conversation.
Here every change has a written reason, depends on the persona (needs, weights, patience, threshold)
and can be tested without any model.
"""
import re
from app.customers.archetypes import get_archetype
from app.memory.state_manager import same_topic

OUTCOMES = {
    "reserve": "Réservé",
    "option": "Option posée",
    "rappel": "À rappeler",
    "devis_email": "Devis par e-mail",
    "perdu": "Perdu",
}
MAX_TURNS = 20
MIN_PATIENCE = 3  # even a rushed, angry client gives the advisor a few chances
DECIDE_FROM_TURN = 4  # from this exchange on, the client is told to make up its mind
# A conversation that goes on in good faith wears resistance down: from turn 4 the bar to book lowers a little each turn
SOFTEN_PER_TURN = 0.04
MIN_THRESHOLD = 0.45
MAX_PROMISES = 12
MAX_CLIENT_FACTS = 15
MAX_ASKED = 20
# A request phrased without "?" is still a question ("Je voudrais connaître le prix exact.")
_REQUEST = re.compile(r"\b(j.aimerais|je voudrais|je veux|pouvez-vous|pourriez-vous|dites-moi|savoir|conna[iî]tre|combien|quel(le)?s?)\b",
                      re.IGNORECASE)

TONE_TRUST = {"chaleureux": 0.8, "neutre": 0.2, "pressant": -0.8, "evasif": -1.0}
PRICE_SCORE = {"confort": 1.0, "limite": 0.55, "trop_cher": 0.1, "inconnu": 0.5}

# Closing line added when the ENGINE ends the talk (patience, turns) and the model's reply did not say goodbye
CLOSINGS = {
    "perdu": {"fr": "Bon, je crois que je vais regarder ailleurs. Merci quand même.",
              "en": "Well, I think I'll look elsewhere. Thanks anyway."},
    "rappel": {"fr": "Écoutez, je vais réfléchir de mon côté et je vous rappelle.",
               "en": "Listen, I'll think it over and call you back."},
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def init_state(human, setup: dict) -> dict:
    """Decision state of a new simulation, from the persona's portrait and the drawn scenario/mood/events."""
    d = human.decision
    patience = max(MIN_PATIENCE, d.patience + setup["patience_delta"])
    companions = [c.name for c in human.identity.companions if c.decides]
    return {
        **setup,
        "must_haves": list(d.must_haves), "met": [],
        "deal_breakers": list(d.deal_breakers), "delighters": list(d.delighters),
        "threshold": _clamp(d.threshold + setup["threshold_delta"], 0.3, 0.95),
        "patience": patience, "patience_left": patience,
        "decides_alone": d.decides_alone and setup["decides_alone"] and not companions,
        "co_deciders": companions, "payment": d.payment,
        "revealed": [], "promises": [], "emotion": 0.5, "price_feeling": "inconnu",
        "turn": 0, "log": [], "outcome": None,
    }


def apply(state: dict, signals, scores: dict, archetype: str, hidden_facts: list = ()) -> dict:
    """Update `state` with one turn's signals. scores = {trust, interest, intent} before the turn.
    Returns the new scores, the reason of the change, and the outcome when the talk ends."""
    w = get_archetype(archetype)["human"]["weights"]
    state["turn"] += 1
    reasons = []

    new_met = [m for m in state["must_haves"] if m not in state["met"]
               and any(same_topic(m, s) for s in signals.must_haves_met)]
    state["met"] += new_met
    revealed = [f for f in hidden_facts if f not in state["revealed"] and any(same_topic(f, s) for s in signals.hidden_revealed)]
    state["revealed"] += revealed
    for p in signals.advisor_promises:
        if p and not any(same_topic(p, q) for q in state["promises"]):
            state["promises"] = (state["promises"] + [p])[-MAX_PROMISES:]
    remember_client(state, signals.client_facts)
    if signals.price_feeling != "inconnu":
        state["price_feeling"] = signals.price_feeling
    deal_breaker = bool(signals.deal_breaker_hit)
    if deal_breaker:
        state["broken"] = True  # a deal breaker hit stops the softening
    tone = signals.advisor_tone

    trust_d = (TONE_TRUST[tone] + 0.7 * len(new_met) - 1.2 * len(signals.red_flags)
               - (2.5 if deal_breaker else 0) + (0.4 if signals.delighter else 0) + (0.3 if revealed else 0))
    interest_d = (0.6 * len(new_met) + (1.0 if signals.delighter else 0) - 0.5 * len(signals.red_flags)
                  - (0.8 if signals.price_feeling == "trop_cher" else 0) + (0.3 if signals.buying_signal else 0))
    state["emotion"] = round(_clamp(state["emotion"] + (0.25 if signals.delighter else 0) + (0.05 if tone == "chaleureux" else 0)
                                    - (0.2 if tone in ("pressant", "evasif") else 0) - (0.3 if deal_breaker else 0)
                                    - 0.1 * len(signals.red_flags), 0.0, 1.0), 2)
    trust = round(_clamp(scores["trust"] + _clamp(trust_d, -2.5, 2.5), 0, 10), 1)
    interest = round(_clamp(scores["interest"] + _clamp(interest_d, -2, 2), 0, 10), 1)

    fit = len(state["met"]) / len(state["must_haves"]) if state["must_haves"] else 0.5
    target = 10 * (w["fit"] * fit + w["trust"] * trust / 10 + w["price"] * PRICE_SCORE[state["price_feeling"]]
                   + w["emotion"] * state["emotion"])
    if deal_breaker:
        target *= 0.6
    negative = bool(deal_breaker or signals.red_flags or tone in ("pressant", "evasif") or signals.price_feeling == "trop_cher")
    move = _clamp(target - scores["intent"], -2, 2)
    if move < 0 and not negative:
        move = 0  # nothing went wrong: the client's wish to book does not fade just because the talk is young
    intent = round(_clamp(scores["intent"] + move, 0, 10), 1)

    if new_met:
        reasons.append("besoin couvert : " + ", ".join(new_met))
    if tone != "neutre":
        reasons.append(f"ton {tone}")
    if signals.delighter:
        reasons.append(f"coup de cœur : {signals.delighter}")
    if signals.red_flags:
        reasons.append("alerte : " + ", ".join(signals.red_flags))
    if deal_breaker:
        reasons.append(f"rédhibitoire : {signals.deal_breaker_hit}")
    if signals.price_feeling != "inconnu":
        reasons.append(f"prix ressenti {signals.price_feeling.replace('_', ' ')}")
    if revealed:
        reasons.append("info cachée révélée : " + ", ".join(revealed))

    # Patience: a bad turn costs one point at once; turns without progress cost one point every two in a row
    progress = bool(new_met or revealed or signals.delighter or signals.buying_signal or signals.advisor_promises)
    state["stalls"] = 0 if progress else state.get("stalls", 0) + 1
    # A bad tone costs a point at once; an alert only if the turn brought nothing; two empty turns in a row cost one
    if tone in ("pressant", "evasif") or deal_breaker or (signals.red_flags and not progress) or state["stalls"] >= 2:
        state["patience_left"] = max(0, state["patience_left"] - 1)
        state["stalls"] = 0
        reasons.append(f"patience {state['patience_left']}/{state['patience']}")

    outcome, why = decide(state, signals, intent, fit)
    state["outcome"] = outcome
    reason = f"Intention {intent - scores['intent']:+.1f}" + (" : " + ", ".join(reasons) if reasons else "")
    state["log"].append({"turn": state["turn"], "intent": intent, "trust": trust, "reason": reason})
    return {"trust": trust, "interest": interest, "intent": intent, "reason": reason,
            "outcome": outcome, "outcome_reason": why, "patience_left": state["patience_left"]}


def questions_in(text: str) -> list:
    """The questions and requests in a client's message ("Et le transfert, c'est inclus ?",
    "J'aimerais connaître le tarif exact.")."""
    sentences = [x.strip() for x in re.findall(r"[^.!?]+[.!?]?", text or "")]
    return [x for x in sentences if len(x) > 8 and (x.endswith("?") or _REQUEST.search(x))]


def remember_questions(state: dict, text: str) -> None:
    """Count the questions the client asked, by topic, so it never loops on a point the advisor already answered."""
    asked = list(state.get("asked", []))
    for q in questions_in(text):
        for item in asked:
            if same_topic(item["q"], q):
                item["n"] += 1
                break
        else:
            asked.append({"q": q, "n": 1})
    state["asked"] = asked[-MAX_ASKED:]


def repeated_questions(state: dict, text: str) -> list:
    """Questions of `text` the client already asked earlier (the advisor has replied since)."""
    return [q for q in questions_in(text) if any(same_topic(item["q"], q) for item in state.get("asked", []))]


def remember_client(state: dict, facts) -> None:
    """Short-term memory of what the client said in this session; a newer statement on the same topic replaces the old one."""
    kept = list(state.get("client_facts", []))
    for f in facts or []:
        f = (f or "").strip()
        if f:
            kept = [k for k in kept if not same_topic(k, f)] + [f]
    state["client_facts"] = kept[-MAX_CLIENT_FACTS:]


def decide(state: dict, signals, intent: float, fit: float):
    """(outcome, reason) or (None, None) while the talk goes on."""
    turn = state["turn"]
    threshold = max(MIN_THRESHOLD, state["threshold"] - SOFTEN_PER_TURN * max(0, turn - 3))
    enough = (fit >= 1 or (fit >= 0.5 and (state["emotion"] >= 0.7 or turn >= 5))
              or (turn >= 7 and not state.get("broken")))
    ready = intent / 10 >= threshold and state["price_feeling"] != "trop_cher" and enough
    wants = signals.proposed_outcome
    # The client said yes in its own words: that decision stands, unless the price is over budget or a deal breaker was hit
    if wants in ("reserve", "option") and state["price_feeling"] != "trop_cher" and not signals.deal_breaker_hit:
        if state["decides_alone"]:
            return "reserve", ("Besoins couverts, confiance suffisante et prix acceptable : le client réserve." if ready
                               else "Le client a décidé de réserver.")
        who = ", ".join(state["co_deciders"]) or "les autres voyageurs"
        if wants == "option":
            return "option", f"Convaincu, mais doit valider avec {who} : il pose une option."
        return "rappel", f"Convaincu, mais doit d'abord en parler avec {who}."
    if signals.wants_to_leave or wants == "perdu":
        if wants == "perdu" or state["patience_left"] <= 1 or signals.deal_breaker_hit or state["price_feeling"] == "trop_cher":
            return "perdu", "Le client met fin à l'échange : " + (signals.deal_breaker_hit or "offre ou échange insatisfaisant") + "."
    if wants in ("devis_email", "rappel") and state["turn"] >= 2:
        return wants, "Le client préfère " + ("recevoir le devis par e-mail." if wants == "devis_email" else "réfléchir et rappeler.")
    # Patience never ends the talk by itself: at 0 the client is told it is at the end of its rope and leaves in its
    # own words (wants_to_leave / proposed_outcome). Only a runaway conversation is closed by the engine.
    if state["turn"] >= MAX_TURNS:
        if intent >= 5:
            return "rappel", "Conversation trop longue : l'offre l'intéresse, il rappellera peut-être."
        return "perdu", "Conversation trop longue sans conclusion."
    return None, None


def feelings(state: dict, trust: float, intent: float) -> str:
    """What the client feels, in words (the prompt never shows it numbers)."""
    t = ("tu ne lui fais pas confiance" if trust < 3 else "tu restes prudent(e) avec lui" if trust < 5
         else "tu commences à lui faire confiance" if trust < 7 else "tu lui fais confiance")
    i = ("tu n'es pas du tout décidé(e)" if intent < 3 else "tu hésites encore" if intent < 5
         else "l'offre t'intéresse vraiment" if intent < 7 else "tu as presque envie de dire oui")
    ratio = state["patience_left"] / max(1, state["patience"])
    p = ("tu as encore de la patience" if ratio > 0.6 else "tu commences à t'impatienter" if ratio > 0.3
         else "tu es à bout : si l'agent ne fait pas avancer les choses tout de suite, tu mets fin à l'échange")
    missing = [m for m in state["must_haves"] if m not in state["met"]]
    lines = [f"- Envers l'agent : {t}.", f"- Sur l'offre : {i}.", f"- Patience : {p}."]
    lines.append("- Il te manque encore pour te décider : " + (", ".join(missing) if missing else "rien, l'essentiel est couvert") + ".")
    turn = state.get("turn", 0) + 1  # the exchange being answered
    if state["price_feeling"] != "trop_cher" and (turn >= DECIDE_FROM_TURN or (not missing and intent >= 5)):
        lines.append(f"- C'est déjà l'échange n°{turn} : il est temps de te décider. Si le prix rentre dans ton budget, conclus "
                     "maintenant (tu réserves, ou tu poses une option) au lieu de chercher d'autres garanties.")
    if not state["decides_alone"]:
        who = ", ".join(state["co_deciders"]) or "les autres voyageurs"
        lines.append(f"- Tu ne peux pas réserver sans l'accord de {who} : si tout te convient, propose de poser une option "
                     "ou dis que tu en parles et que tu rappelles.")
    return "\n".join(lines)


def closing_line(outcome: str, language: str) -> str:
    return CLOSINGS.get(outcome, {}).get(language, "")


if __name__ == "__main__":
    from types import SimpleNamespace as S

    def sig(**k):
        base = dict(must_haves_met=[], deal_breaker_hit=None, delighter=None, red_flags=[], advisor_tone="neutre",
                    price_feeling="inconnu", hidden_revealed=[], advisor_promises=[], buying_signal=False,
                    wants_to_leave=False, proposed_outcome=None, client_facts=[])
        return S(**{**base, **k})

    setup = {"patience_delta": 0, "trust_delta": 0, "threshold_delta": 0, "decides_alone": True}
    human = S(decision=S(must_haves=["prix total tout compris", "taxe de séjour précisée"], deal_breakers=[], delighters=[],
                         threshold=0.8, patience=5, decides_alone=True, payment=""), identity=S(companions=[]))
    st = init_state(human, setup)
    scores = {"trust": 2.5, "interest": 4.5, "intent": 2.0}
    r = apply(st, sig(must_haves_met=["prix total tout compris"], advisor_tone="chaleureux", price_feeling="confort"), scores, "unsatisfied")
    assert r["trust"] > 2.5 and r["intent"] > 2.0 and r["outcome"] is None and "besoin couvert" in r["reason"]
    # buying too early: not ready (one need missing) -> talk goes on
    scores = {"trust": r["trust"], "interest": r["interest"], "intent": r["intent"]}
    assert apply(st, sig(buying_signal=True), scores, "unsatisfied")["outcome"] is None
    # vague advisor exhausts patience -> lost
    st2 = init_state(human, setup)
    s2 = {"trust": 2.5, "interest": 4.5, "intent": 2.0}
    for _ in range(5):
        r2 = apply(st2, sig(advisor_tone="evasif"), s2, "unsatisfied")
        s2 = {"trust": r2["trust"], "interest": r2["interest"], "intent": r2["intent"]}
    assert r2["outcome"] is None and r2["patience_left"] == 0 and r2["trust"] < 2.5  # at the end of its rope, it decides
    assert "tu es à bout" in feelings(st2, r2["trust"], r2["intent"])
    assert apply(st2, sig(advisor_tone="evasif", wants_to_leave=True), s2, "unsatisfied")["outcome"] == "perdu"
    print("decision ok")
