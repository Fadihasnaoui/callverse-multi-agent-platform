"""
End-of-simulation debrief for the advisor: outcome and its real reason, hidden facts found or missed,
key moments of the booking intention, three tips and a score out of 20.

The model writes it from the transcript and the decision log; without a model, a factual debrief
is built from the decision state alone so the advisor always gets feedback.
"""
import json
import logging
from app.conversation.decision import OUTCOMES
from app.customers.human import load_human

logger = logging.getLogger("simulator")

RUBRIC = ("découverte des besoins (questions posées), écoute et reformulation, réponses aux inquiétudes, "
          "honnêteté et précision (prix, inclusions), adaptation au type de client, conclusion")


def _key_moments(log: list, n: int = 3) -> list:
    """Turns where the intention moved the most."""
    moves, prev = [], None
    for entry in log:
        if prev is not None:
            moves.append((abs(entry["intent"] - prev), entry))
        prev = entry["intent"]
    return [f"Tour {e['turn']} : {e['reason']}" for _, e in sorted(moves, key=lambda m: -m[0])[:n]]


def fallback_debrief(sim, profile) -> dict:
    ds = sim.decision_state or {}
    human = load_human(profile)
    facts = [h.fact for h in human.hidden]
    found = [f for f in facts if f in ds.get("revealed", [])]
    missing = [m for m in ds.get("must_haves", []) if m not in ds.get("met", [])]
    tips = []
    if len(found) < len(facts):
        tips.append("Posez plus de questions ouvertes sur le contexte du voyage (occasion, accompagnants, besoins particuliers).")
    if missing:
        tips.append("Traitez explicitement ce qui compte pour le client : " + ", ".join(missing) + ".")
    if any("evasif" in e["reason"] or "pressant" in e["reason"] for e in ds.get("log", [])):
        tips.append("Évitez l'esquive et la pression : répondez précisément, puis proposez la suite.")
    tips.append("Proposez clairement l'étape suivante (réservation, option ou devis) dès que les besoins sont couverts.")
    score = 8 + 6 * (len(ds.get("met", [])) / max(1, len(ds.get("must_haves", [])))) + 3 * (len(found) / max(1, len(facts)))
    score += {"reserve": 3, "option": 2, "rappel": 1, "devis_email": 1}.get(sim.outcome, 0)
    return {
        "resultat": OUTCOMES.get(sim.outcome, "Terminée sans décision"),
        "raison": sim.outcome_reason or "Simulation terminée par le conseiller.",
        "infos_cachees_trouvees": found,
        "infos_cachees_ratees": [f for f in facts if f not in found],
        "besoins_non_couverts": missing,
        "moments_cles": _key_moments(ds.get("log", [])),
        "conseils": tips[:3],
        "score_sur_20": round(min(20, score)),
        "source": "règles",
    }


def build_debrief(db, sim, profile, provider) -> dict:
    """Write (once) and store the debrief of a finished simulation."""
    if sim.debrief:
        return sim.debrief
    base = fallback_debrief(sim, profile)
    transcript = "\n".join(f"{'Conseiller' if m.sender == 'sales_rep' else 'Client'} : {m.content}" for m in (sim.messages or []))
    prompt = (
        "Tu es formateur commercial dans une agence de voyages. Voici une simulation de vente entre un conseiller et un client simulé.\n"
        f"Client : {profile.name}, type {profile.archetype}. Résultat : {base['resultat']} ({base['raison']}).\n"
        f"Informations cachées du client trouvées : {base['infos_cachees_trouvees'] or 'aucune'} ; ratées : {base['infos_cachees_ratees'] or 'aucune'}.\n"
        f"Besoins non couverts : {base['besoins_non_couverts'] or 'aucun'}.\n"
        f"Évolution de l'intention d'achat : {base['moments_cles']}.\n\nTranscription :\n{transcript}\n\n"
        f"Évalue le conseiller selon : {RUBRIC}. Réponds UNIQUEMENT en JSON : "
        '{"points_forts": ["..."], "points_faibles": ["..."], "conseils": ["3 conseils concrets"], "score_sur_20": 12, '
        '"resume": "2 phrases"}'
    )
    text = provider.generate_text(prompt, temperature=0.3) if provider else None
    debrief = dict(base)
    if text:
        try:
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            debrief.update({k: data[k] for k in ("points_forts", "points_faibles", "conseils", "resume") if k in data})
            if isinstance(data.get("score_sur_20"), (int, float)):
                debrief["score_sur_20"] = max(0, min(20, round(data["score_sur_20"])))
            debrief["source"] = "ia"
        except (ValueError, TypeError):
            logger.warning("Debrief: the model's JSON was unusable, keeping the rule-based debrief")
    sim.debrief = debrief
    db.commit()
    return debrief
