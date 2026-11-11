import json
import logging
import random
from typing import Optional
from app.customers.schemas import CustomerProfileCreate, TravelDatesSchema
from app.core.config import settings
from app.customers.archetypes import ARCHETYPES, normalize_archetype

logger = logging.getLogger("simulator")

FALLBACK_NAMES = [
    "Yasmine Trabelsi", "Mehdi Jaziri", "Amel Ben Youssef", "Sami Gharbi", "Inès Karoui", "Walid Mansouri",
    "Rania Chaabane", "Hichem Ayari", "Claire Moreau", "Julien Lefèvre", "Sophie Martin", "Thomas Girard",
    "Emma Wilson", "James Carter", "Olivia Brown", "Lucas Schneider", "Giulia Bianchi", "Omar El Fassi",
]

def _valid_human(raw):
    """The generated portrait if it matches the schema, else None (the type's defaults apply)."""
    from app.customers.human import HumanProfile
    try:
        return HumanProfile.model_validate(raw).model_dump() if isinstance(raw, dict) else None
    except ValueError:
        logger.warning("Generated human profile rejected (schema mismatch)")
        return None


def generate_customer_persona(
    archetype: str,
    destination: Optional[str] = None,
    budget: Optional[float] = None,
    background: Optional[str] = None
) -> CustomerProfileCreate:
    archetype_key = normalize_archetype(archetype)
    preset = ARCHETYPES[archetype_key]
    dest = destination or preset["default_destination"]
    total_budget = budget or preset["default_budget"]
    bg_context = background.strip() if background else ""
    # The name is chosen here, not by the model: left to itself the LLM keeps
    # writing the same few names, so "new" clients looked like the stored ones.
    client_name = random.choice(FALLBACK_NAMES)

    # Try generating via Groq LLM if API key is configured
    if settings.GROQ_API_KEY and settings.LLM_PROVIDER.lower() == "groq":
        try:
            # Same provider chain as the conversations (Groq fallbacks, then
            # OpenRouter): calling Groq directly failed whenever its quota ran
            # out, and every "new" client became the archetype's fixed name.
            from app.conversation.llm_provider import GroqLLMProvider
            provider = GroqLLMProvider()
            
            prompt = f"""
            Génère un profil client réaliste au format JSON pour une agence de voyages, cherchant un séjour dans des HÔTELS EN TUNISIE (ex: Hammamet, Djerba, Sousse, Monastir, Tunis Gammarth, Mahdia, Tozeur).
            Type de client : {preset['label']} ({preset['description']})
            Comportement attendu : {preset['behavior']}
            Nom du client (à utiliser tel quel) : {client_name}
            Contexte personnel / Background : {bg_context if bg_context else 'Recherche un séjour hôtelier en Tunisie'}
            Préférence de destination / hôtel en Tunisie : {dest}
            Budget total : {total_budget} TND (dinars tunisiens)

            Le profil doit être exigeant et chercher des détails concrets (chambres vue mer, formule All Inclusive, transferts, thalasso, taxes).

            Renvoie STRICTEMENT un JSON valide respectant cette structure exacte :
            {{
                "name": "Prénom Nom",
                "age_range": "25-35",
                "persona_description": "Description détaillée de l'attitude, des attentes et exigences du client pour son séjour en Tunisie",
                "destination_preferences": ["{dest}"],
                "start_date": "2026-11-01",
                "end_date": "2026-11-08",
                "flexible": true,
                "number_of_travelers": 2,
                "budget_total": {total_budget},
                "price_sensitivity": "{preset['price_sensitivity']}",
                "preferred_accommodation": "Hôtel 4 ou 5 étoiles en Tunisie",
                "preferred_transportation": "Vol direct vers Tunis, Enfidha ou Djerba",
                "motivations": ["Détente", "Plage et soleil en Tunisie"],
                "constraints": ["Budget max strict", "Chambre au calme avec balcon"],
                "personality_traits": ["exigeant", "attentif aux détails"],
                "communication_style": "Style de communication en français : pose des questions précises sur les prestations et demande des détails",
                "objections": ["Objection 1 sur les taxes ou frais", "Objection 2 sur les prestations incluses"],
                "decision_criteria": ["Rapport qualité/prix", "Transparence totale"],
                "desired_outcome": "Obtenir les détails complets et garanties avant de réserver",
                "human_profile": {{
                    "identity": {{"age": 42, "job": "métier précis", "city": "ville de résidence", "family": "situation familiale",
                                 "companions": [{{"name": "Prénom", "role": "épouse / ami / fils de 8 ans", "decides": false}}]}},
                    "story": {{"why_now": "la raison concrète de CE voyage, maintenant", "past_trips": ["un souvenir de voyage marquant, bon ou mauvais"],
                              "tunisia_link": "son lien avec la Tunisie"}},
                    "big_five": {{"openness": 3, "conscientiousness": 4, "extraversion": 2, "agreeableness": 3, "neuroticism": 4}},
                    "voice": {{"address": "vouvoie", "sentence_length": "court | moyen | long", "tics": ["2 ou 3 petits mots à lui"],
                              "derja_words": [], "never_says": ["Cependant"]}},
                    "emotions": {{"reassured_by": ["..."], "annoyed_by": ["..."], "delighted_by": ["..."]}},
                    "hidden": [{{"fact": "une info qu'il garde pour lui (vrai budget, contrainte de santé, devis concurrent...)",
                                "reveal_if": "la question de l'agent qui la fait sortir"}}],
                    "decision": {{"must_haves": ["2 ou 3 besoins indispensables"], "deal_breakers": ["ce qui le fait renoncer"],
                                 "delighters": ["ce qui le ferait craquer"], "decides_alone": true, "payment": "carte | acompte | virement"}}
                }}
            }}
            Le portrait humain doit être cohérent avec le type de client (Big Five de 1 à 5) et crédible : une vraie personne,
            pas un cliché. Mots de tunisien (derja_words) seulement si la personne est tunisienne.
            """
            
            message = provider._complete([{"role": "user", "content": prompt}], json_mode=True, temperature=0.7)
            data = json.loads(message.content)
            
            return CustomerProfileCreate(
                scenario_id=f"{archetype_key}_generated",
                archetype=archetype_key,
                name=client_name,
                age_range=data.get("age_range", "30-40"),
                persona_description=data.get("persona_description", bg_context or preset["description"]),
                destination_preferences=data.get("destination_preferences", [dest]),
                travel_dates=TravelDatesSchema(
                    start_date=data.get("start_date", "2026-11-01"),
                    end_date=data.get("end_date", "2026-11-08"),
                    flexible=data.get("flexible", True)
                ),
                number_of_travelers=data.get("number_of_travelers", 2),
                budget_total=float(data.get("budget_total", total_budget)),
                price_sensitivity=data.get("price_sensitivity", preset["price_sensitivity"]),
                preferred_accommodation=data.get("preferred_accommodation", "Standard hotel"),
                preferred_transportation=data.get("preferred_transportation", "Flight"),
                motivations=data.get("motivations", ["Vacation"]),
                constraints=data.get("constraints", ["Budget"]),
                personality_traits=data.get("personality_traits", [archetype_key]),
                communication_style=data.get("communication_style", "Direct"),
                objections=data.get("objections", ["Price", "Availability"]),
                decision_criteria=data.get("decision_criteria", ["Value"]),
                initial_trust=preset["initial_trust"],
                initial_interest=preset["initial_interest"],
                initial_booking_intention=preset["initial_booking_intention"],
                desired_outcome=data.get("desired_outcome", "Book a trip"),
                human_profile=_valid_human(data.get("human_profile")),
            )
        except Exception as err:
            logger.warning(f"AI Persona Generator fallback due to error: {err}")

    # Default fallback generator
    return CustomerProfileCreate(
        scenario_id=f"{archetype_key}_fallback",
        archetype=archetype_key,
        # A varied name, not the archetype's fixed one shared with the stored personas.
        name=client_name,
        age_range="28-40",
        persona_description=bg_context or preset["description"],
        destination_preferences=[dest],
        travel_dates=TravelDatesSchema(start_date="2026-11-01", end_date="2026-11-08", flexible=True),
        number_of_travelers=4 if archetype_key == "family" else 2,
        budget_total=total_budget,
        price_sensitivity=preset["price_sensitivity"],
        preferred_accommodation="Hôtel 4 ou 5 étoiles en Tunisie",
        preferred_transportation="Vol direct",
        motivations=["Découverte", "Détente"],
        constraints=[f"Budget max de {total_budget:.0f} TND"],
        personality_traits=[preset["label"].lower()],
        communication_style=preset["behavior"],
        objections=["Le prix total inclut-il toutes les taxes et les transferts ?"],
        decision_criteria=["Rapport qualité/prix"],
        initial_trust=preset["initial_trust"],
        initial_interest=preset["initial_interest"],
        initial_booking_intention=preset["initial_booking_intention"],
        desired_outcome=f"Réserver un séjour à {dest}"
    )
