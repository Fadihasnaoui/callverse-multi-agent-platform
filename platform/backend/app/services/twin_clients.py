"""Client sources for Digital Twin calls.

Every Twin call is driven by one simulated client.  Three sources exist so a
call-centre run does not replay the same few customers:

* ``jasmin_profile``   one of Jasmin's stored personas, with a random
                       situation (scenario) and difficulty for this call;
* ``jasmin_generated`` a brand-new Jasmin persona from a random archetype,
                       destination, budget, scenario and difficulty;
* ``scripted``         a Voice Lab rule-based persona (also covers Arabic and
                       Darija clients, which Jasmin does not speak).

The scripted engine is the Voice Lab's own ``MockClientSimulator``, loaded
read-only from its source file, so both labs test Amira with the same client.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from app import config

ARCHETYPES = ("satisfied", "unsatisfied", "hesitant", "anxious", "demanding", "rushed", "family", "vip")
SCENARIOS = ("premiere_demande", "rappel_devis", "derniere_minute", "modification", "reclamation", "client_fidele", "groupe")
DIFFICULTIES = ("facile", "normal", "difficile")
DESTINATIONS = ("Hammamet", "Sousse", "Djerba", "Monastir", "Tunis", "Mahdia", "Tabarka", "Tozeur")
# Voice Lab persona files spell Darija two ways; the Twin uses one code.
_LANGUAGE = {"darija": "aeb-Latn"}

_engine: Any = None
# Live scripted clients, keyed by Twin session ID.  They hold no customer data
# and are rebuilt per call; after an API restart such a call is closed as failed.
_scripted: dict[str, Any] = {}


def _mock_engine() -> Any:
    global _engine
    if _engine is None:
        path = Path(config.DIGITAL_TWIN_SCRIPTED_ENGINE)
        spec = importlib.util.spec_from_file_location("voice_lab_call_orchestrator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # stdlib-only module; it does not run main()
        _engine = module.MockClientSimulator
    return _engine


def scripted_personas() -> list[dict[str, Any]]:
    folder = Path(config.DIGITAL_TWIN_SCRIPTED_PERSONAS_DIR)
    if not folder.is_dir() or not Path(config.DIGITAL_TWIN_SCRIPTED_ENGINE).is_file():
        return []
    personas = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("opening"):
            data["id"] = path.stem
            data["language"] = _LANGUAGE.get(str(data.get("language") or "fr"), str(data.get("language") or "fr"))
            personas.append(data)
    return personas


def pick(rng: random.Random, mix: str, jasmin_profiles: list[str], language: str,
         used: set[str] | None = None) -> dict[str, Any]:
    """Choose one call's client.  ``language`` is fr/en/mixed for Jasmin calls.

    ``used`` holds the scripted personas already drawn in this run: a scripted
    persona always plays the same conversation, so drawing it twice produced
    two identical calls.  Repeats only resume once every persona was used.
    """
    scripted = scripted_personas() if mix == "random" else []
    roll = rng.random()
    if scripted and roll < 0.3:
        used = used if used is not None else set()
        fresh = [p for p in scripted if p["id"] not in used] or scripted
        persona = rng.choice(fresh)
        used.add(persona["id"])
        return {"source": "scripted", "persona": persona["id"], "name": persona.get("name") or persona["id"],
                "language": persona["language"]}
    lang = language if language in {"fr", "en"} else rng.choice(("fr", "en"))
    choice = {"scenario": rng.choice(SCENARIOS), "difficulty": rng.choice(DIFFICULTIES), "language": lang}
    if mix == "random" and roll < 0.6:
        return {"source": "jasmin_generated", "archetype": rng.choice(ARCHETYPES),
                "destination": rng.choice(DESTINATIONS), "budget": rng.choice((600, 900, 1200, 1800, 2500, 4000)), **choice}
    return {"source": "jasmin_profile", "profile_id": rng.choice(jasmin_profiles), **choice}


def start_scripted(session_id: str, persona_id: str) -> tuple[str, dict[str, Any]]:
    persona = next((p for p in scripted_personas() if p["id"] == persona_id), None)
    if persona is None:
        raise KeyError(persona_id)
    client = _mock_engine()(persona)
    _scripted[session_id] = {"client": client, "done": False}
    return client.start(), persona


def scripted_next(session_id: str, advisor: dict[str, Any]) -> tuple[str, bool]:
    state = _scripted.get(session_id)
    if state is None:
        raise KeyError(session_id)
    text, done = state["client"].next(advisor)
    state["done"] = bool(done)
    return text, bool(done)


def scripted_active(session_id: str) -> bool:
    state = _scripted.get(session_id)
    return bool(state) and not state["done"]


def end_scripted(session_id: str) -> None:
    _scripted.pop(session_id, None)
