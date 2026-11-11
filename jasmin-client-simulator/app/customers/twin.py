"""
"Jumeau numérique": a real client fills a short form, and we build the persona that
plays them in the advisors' training simulations.

The type of traveler comes straight from what the client seeks in this trip (TRIP_GOALS),
so the result is explainable: one answer -> one archetype.
"""
import uuid
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from app.core.config import settings
from app.customers.archetypes import get_archetype
from app.customers.schemas import CustomerProfileCreate, TravelDatesSchema
from app.hotels.catalog import DESTINATION_CITIES

DESTINATIONS = DESTINATION_CITIES
MAX_NIGHTS = 30

# What the client seeks in this trip -> (label shown in the form, archetype, objections the persona will raise)
TRIP_GOALS = {
    "detente": ("Me détendre et profiter, je fais confiance à l'agence", "satisfied",
                ["La chambre a-t-elle une belle vue et un vrai calme ?", "Que comprend exactement la formule ?"]),
    "sans_surprise": ("Un séjour sans mauvaise surprise : j'ai déjà été déçu(e) par une agence", "unsatisfied",
                      ["Le prix annoncé inclut-il vraiment toutes les taxes et les frais ?", "Qu'est-ce qui me garantit que l'hôtel sera conforme ?"]),
    "rassurer": ("Être rassuré(e) : je m'inquiète facilement (sécurité, santé, annulation)", "anxious",
                 ["La région est-elle sûre pour les touristes ?", "Puis-je annuler sans frais en cas d'imprévu ?"]),
    "luxe": ("Le luxe et un service d'exception, sans compromis", "vip",
             ["La suite est-elle la meilleure de l'hôtel ?", "Un transfert privé et une conciergerie sont-ils inclus ?"]),
    "garanties": ("Que chaque détail soit parfait, avec des garanties écrites", "demanding",
                  ["Pouvez-vous me garantir par écrit la catégorie et la vue de la chambre ?", "Quels soins exacts sont inclus ?"]),
    "rapide": ("Une offre claire et rapide, je n'ai pas beaucoup de temps", "rushed",
               ["Quel est le prix total, tout compris ?", "Pouvez-vous confirmer la réservation aujourd'hui ?"]),
    "famille": ("Des vacances réussies pour toute la famille, avec les enfants", "family",
                ["Avez-vous une chambre familiale pour nous tous ?", "Le club enfants est-il inclus ?"]),
    "conseils": ("Je ne sais pas encore, j'ai besoin de vos conseils", "hesitant",
                 ["Quelle région me conseillez-vous pour cette période ?", "Pouvez-vous me comparer deux options simples ?"]),
}


class TwinQuestionnaire(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=60)
    travelers: int = Field(..., ge=1, le=12)
    budget: float = Field(..., gt=0, le=1_000_000)
    currency: Literal["TND", "EUR", "USD"] = "TND"
    destination: Literal[tuple(DESTINATIONS)]
    start_date: date
    end_date: date
    trip_goal: Literal[tuple(TRIP_GOALS)]
    notes: Optional[str] = Field(None, max_length=500)
    consent: bool
    # Login of the client's space, where they read the simulations played with their persona
    # Optional "portrait humain" questions: make the persona sound like the real client
    companions: Optional[str] = Field(None, max_length=200)      # "ma femme Amel et nos deux enfants"
    occasion: Optional[str] = Field(None, max_length=200)        # "nos 10 ans de mariage"
    book_now_if: Optional[str] = Field(None, max_length=200)     # what would make them book on the spot
    walk_away_if: Optional[str] = Field(None, max_length=200)    # what would make them leave
    past_trip: Optional[str] = Field(None, max_length=300)       # a memorable trip, good or bad
    address: Literal["vouvoie", "tutoie"] = "vouvoie"
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("first_name", "notes", "companions", "occasion", "book_now_if", "walk_away_if", "past_trip")
    @classmethod
    def strip_text(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_consent_and_dates(self):
        if not self.consent:
            raise ValueError("Le consentement est obligatoire pour créer votre persona voyageur.")
        if self.start_date <= date.today():
            raise ValueError("La date d'arrivée doit être dans le futur.")
        if not 0 < (self.end_date - self.start_date).days <= MAX_NIGHTS:
            raise ValueError(f"La date de départ doit être après l'arrivée, pour {MAX_NIGHTS} nuits au maximum.")
        return self

    @property
    def budget_tnd(self) -> float:
        tnd_per_unit = {"TND": 1.0, "USD": settings.TND_PER_USD, "EUR": settings.TND_PER_USD / settings.EUR_PER_USD}[self.currency]
        return round(self.budget * tnd_per_unit)


def twin_human_profile(a: TwinQuestionnaire) -> dict:
    """Portrait built from the client's own words (optional answers); the type fills the rest."""
    decision = {"decides_alone": not a.companions, "delighters": [a.book_now_if] if a.book_now_if else [],
                "deal_breakers": [a.walk_away_if] if a.walk_away_if else []}
    return {
        # a real client filling in the form is making a first request (or is in a hurry)
        "scenario": "derniere_minute" if a.trip_goal == "rapide" else "premiere_demande",
        "identity": {"family": a.companions or "", "companions": [{"name": a.companions, "role": "compagnon(s) de voyage", "decides": True}]
                     if a.companions else []},
        "story": {"why_now": a.occasion or "", "past_trips": [a.past_trip] if a.past_trip else []},
        "voice": {"address": a.address},
        "decision": decision,
    }


def derive_archetype(a: TwinQuestionnaire) -> str:
    return TRIP_GOALS[a.trip_goal][1]


def build_twin_profile(a: TwinQuestionnaire) -> CustomerProfileCreate:
    """The persona built from the form answers (it asks for a region; the advisor will propose hotels)."""
    archetype = derive_archetype(a)
    preset = get_archetype(archetype)
    goal_label, _, objections = TRIP_GOALS[a.trip_goal]
    start, end = a.start_date, a.end_date
    budget_tnd = a.budget_tnd

    description = (f"Persona de {a.first_name}, {a.travelers} voyageur(s), séjour à {a.destination}. "
                   f"du {start:%d/%m/%Y} au {end:%d/%m/%Y}. Ce qu'il cherche : {goal_label.lower()}.")
    if a.notes:
        description += f" Précision du client : {a.notes}"

    profile = CustomerProfileCreate(
        scenario_id=f"twin_{uuid.uuid4().hex[:8]}",
        archetype=archetype,
        name=a.first_name,
        age_range="Non précisé",
        persona_description=description,
        destination_preferences=[a.destination],
        travel_dates=TravelDatesSchema(start_date=start.isoformat(), end_date=end.isoformat(), flexible=True),
        number_of_travelers=a.travelers,
        budget_total=budget_tnd,
        price_sensitivity=preset["price_sensitivity"],
        preferred_accommodation=f"Hôtel {preset['hotel_min_stars']} étoiles minimum à {a.destination}",
        preferred_transportation="Vol direct",
        motivations=[goal_label],
        constraints=[f"Budget maximum de {budget_tnd:.0f} TND tout compris", f"Séjour à {a.destination}"],
        personality_traits=[preset["label"].lower()],
        communication_style=preset["description"],
        objections=objections,
        decision_criteria=[goal_label],
        initial_trust=preset["initial_trust"],
        initial_interest=preset["initial_interest"],
        initial_booking_intention=preset["initial_booking_intention"],
        desired_outcome=f"Trouver un séjour à {a.destination} qui réponde à son attente : {goal_label.lower()}.",
        human_profile=twin_human_profile(a),
    )
    return profile
