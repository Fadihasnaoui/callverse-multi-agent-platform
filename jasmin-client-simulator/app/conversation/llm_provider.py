import abc
import json
import logging
import re
import time
from app.customers.schemas import CustomerProfileResponse, CustomerStateSnapshot, CustomerTurnResponse
from app.conversation.prompts import build_system_prompt
from app.core.config import settings
from app.core.exceptions import LLMProviderException

logger = logging.getLogger("simulator")

# Amounts in dinars: "2 950 TND", "1200 DT", "800 dinars". Bare numbers (years, nights) are ignored.
_PRICE_RE = re.compile(r"(\d[\d\s,.]*\d|\d)\s*(?:tnd|dt|dinars?)\b", re.IGNORECASE)


def _mentioned_prices(text: str) -> list:
    prices = []
    for m in _PRICE_RE.finditer(text):
        try:
            prices.append(float(re.sub(r"[\s,]", "", m.group(1))))
        except ValueError:
            pass
    return prices

class BaseLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(
        self,
        profile: CustomerProfileResponse,
        state: CustomerStateSnapshot,
        history_formatted: str,
        latest_rep_message: str,
        hotel_catalog: str = "",
        retry_note: str = "",
    ) -> CustomerTurnResponse:
        """Generate structured customer response and updated state snapshot."""
        pass

    def generate_text(self, prompt: str, temperature: float = 0.9):
        """Free text (the client's opening line). None when the provider cannot write it: callers use a template."""
        return None


class MockLLMProvider(BaseLLMProvider):
    def generate_response(
        self,
        profile: CustomerProfileResponse,
        state: CustomerStateSnapshot,
        history_formatted: str,
        latest_rep_message: str,
        hotel_catalog: str = "",
        retry_note: str = "",
    ) -> CustomerTurnResponse:
        msg_lower = latest_rep_message.lower()

        # Current state values
        trust = state.trust_level
        interest = state.interest_level
        booking_intent = state.booking_intention
        current_stage = state.current_stage

        unresolved = list(state.unresolved_objections) if state.unresolved_objections else list(profile.objections[:2])
        addressed = []
        new_unresolved = []
        quality = "neutral"
        reasoning = ""

        # 0. The advisor asks to confirm: the demo client accepts (the decision engine still checks it is ready)
        closing_words = ["confirm", "réserv", "reserv", "valid", "on passe", "bloquer", "je vous inscris", "on y va", "book"]
        if current_stage != "greetings" and any(k in msg_lower for k in closing_words):
            trust = min(10.0, trust + 1.0)
            booking_intent = min(10.0, booking_intent + 2.0)
            current_stage = "closing"
            quality = "good"
            reasoning = "Le conseiller propose de conclure ; le client est prêt à confirmer."
            branch = "closing"
            customer_reply = "Très bien, ça me convient. On confirme la réservation : envoyez-moi la confirmation par écrit, s'il vous plaît."

        # 1. Greeting stage
        elif current_stage == "greetings":
            current_stage = "needs_discovery"
            trust = min(10.0, trust + 0.5)
            quality = "good"
            reasoning = "Le conseiller a débuté la conversation avec courtoisie et écoute."

            # Talk about ONE place: the hotel the client picked from the table, else its first wish
            # The client asks for a region; proposing hotels is the advisor's job
            region = (state.known_requirements or {}).get("region") or "Tunisie"
            place, place_en = f"la région de {region}", f"the {region} area"
            branch = "greeting"
            customer_reply = (
                f"Bonjour ! Je suis {profile.name}. Nous préparons activement un séjour en Tunisie, à {place}, "
                f"pour {profile.number_of_travelers} voyageur(s) autour du {profile.travel_dates.start_date}. "
                f"Mon budget maximum est strictement plafonné à {profile.budget_total:.0f} TND tout compris. "
                f"Quels hôtels me proposez-vous, que comprend exactement la formule All Inclusive et la taxe de séjour tunisienne est-elle bien incluse ?"
            )

        # 2. Price / Budget checks (detect if rep suggests amount over budget or mentions above/over budget)
        elif (
            any(k in msg_lower for k in [
                "over budget", "above your budget", "exceeds", "more expensive", "too high",
                "dépasse", "au-dessus du budget", "plus cher", "trop cher", "hors budget"
            ]) or
            (any(k in msg_lower for k in ["over", "above", "more than", "dépasse", "supérieur"]) and "budget" in msg_lower) or
            any(price > profile.budget_total for price in _mentioned_prices(msg_lower))
        ):
            # Rep suggested over budget
            trust = max(0.0, trust - 1.5)
            interest = max(0.0, interest - 1.0)
            quality = "poor"
            reasoning = "Le conseiller a proposé une offre qui dépasse le budget strict du client."
            branch = "over_budget"
            customer_reply = (
                f"Cette proposition me paraît inacceptable car mon budget maximum est strictement limité à {profile.budget_total:.0f} TND. "
                f"Je ne peux absolument pas aller au-delà de cette somme ni payer des suppléments cachés. "
                f"Avez-vous une alternative de qualité en Tunisie qui respecte scrupuleusement mon budget de {profile.budget_total:.0f} TND tout inclus, et pouvez-vous me détailler ce que comprend la formule sans mauvaise surprise ?"
            )
            if "Price is higher than expected" not in unresolved:
                new_unresolved.append("Price is higher than expected")

        # 3. Transparent breakdown / Discount / Value offer
        elif any(k in msg_lower for k in [
            "discount", "all-inclusive", "included", "no hidden", "breakdown", "guarantee", "perks",
            "tout inclus", "inclus", "sans frais", "détail", "garantie", "réduction", "transparence"
        ]):
            trust = min(10.0, trust + 1.5)
            interest = min(10.0, interest + 1.5)
            booking_intent = min(10.0, booking_intent + 1.0)
            current_stage = "negotiation" if booking_intent < 7.0 else "closing"
            quality = "excellent"
            reasoning = "Le conseiller a apporté des garanties de transparence et valorisé l'offre."
            branch = "value"

            # Address price or fee objections
            for obj in list(unresolved):
                if any(x in obj.lower() for x in ["price", "fee", "cost", "competitor", "prix", "taxe"]):
                    addressed.append(obj)
                    unresolved.remove(obj)

            customer_reply = (
                f"C'est une excellente nouvelle, j'apprécie votre transparence sur le tarif dans les limites de mon budget de {profile.budget_total:.0f} TND. "
                f"Avant de prendre ma décision, pouvez-vous me préciser les détails du vol et des transferts vers l'hôtel en Tunisie ? "
                f"La formule comprend-elle bien les boissons locales, l'accès au spa/thalasso et la taxe touristique ?"
            )

        # 4. Addressing questions / Discovery / Recommendations
        elif any(k in msg_lower for k in [
            "recommend", "option", "hotel", "resort", "flight", "package",
            "recommande", "conseille", "propose", "hôtel", "séjour", "chambre", "plage"
        ]):
            trust = min(10.0, trust + 1.0)
            interest = min(10.0, interest + 1.0)
            current_stage = "recommendation"
            quality = "good"
            reasoning = "Le conseiller a formulé une recommandation adaptée aux attentes du client."
            branch = "recommend"

            customer_reply = (
                f"Merci pour cette recommandation d'hôtel en Tunisie. "
                f"Pouvez-vous m'en dire plus sur les caractéristiques de la chambre (vue mer garantie, balcon, climatisation) ainsi que sur les activités de la plage privée ? "
                f"Le montant global pour {profile.number_of_travelers} personne(s) reste-t-il parfaitement sous mon plafond de {profile.budget_total:.0f} TND tout compris ?"
            )

        # 5. Generic / Follow-up
        else:
            quality = "neutral"
            reasoning = "Message général du conseiller commercial ; demande de précisions complémentaires."
            branch = "generic"
            customer_reply = (
                f"J'entends bien vos explications, mais j'aimerais avoir des réponses claires sur l'établissement. "
                f"Comment cet hôtel répond-il à mes exigences pour un séjour réussi en Tunisie tout en restant sous les {profile.budget_total:.0f} TND ? "
                f"Pouvez-vous me détailler les services inclus au niveau des repas, des boissons et des loisirs ?"
            )

        if getattr(state, "language", "fr") == "en":
            budget = f"{profile.budget_total:.0f} TND"
            customer_reply = {
                "greeting": (f"Hello! I'm {profile.name}. We're planning a stay in Tunisia, at {place_en if branch == 'greeting' else ''}, "
                             f"for {profile.number_of_travelers} traveller(s) around {profile.travel_dates.start_date}. "
                             f"My maximum budget is strictly {budget} all inclusive. Which hotels can you offer me, "
                             f"what exactly does its All Inclusive cover, and is the Tunisian tourist tax included?"),
                "over_budget": (f"That's not acceptable: my maximum budget is strictly {budget}. I can't go beyond it or pay hidden extras. "
                                f"Do you have a good alternative in Tunisia within {budget} all inclusive?"),
                "value": (f"That's great news, thank you for being transparent within my {budget} budget. Before I decide, "
                          f"could you tell me about the flight and the transfer to the hotel? Are local drinks, the spa and the tourist tax included?"),
                "recommend": (f"Thanks for the recommendation. Could you tell me more about the room (guaranteed sea view, balcony, air conditioning)? "
                              f"Does the total for {profile.number_of_travelers} person(s) stay under {budget} all inclusive?"),
                "generic": (f"I hear you, but I'd like clear answers about the hotel. How does it meet my needs while staying under {budget}? "
                            f"Which meals, drinks and activities are included?"),
                "closing": "All right, that works for me. Let's confirm the booking: please send me the confirmation in writing.",
            }[branch]

        # Never repeat a canned line already said in this conversation
        if customer_reply in (history_formatted or ""):
            en = getattr(state, "language", "fr") == "en"
            customer_reply = ("Fine. So, concretely, what is the total price for me, everything included?" if en
                              else "D'accord. Et concrètement, le prix total tout compris pour moi, c'est combien ?")
            if customer_reply in (history_formatted or ""):
                customer_reply = ("Okay. What would you suggest as the next step?" if en
                                  else "Bon. Qu'est-ce que vous me proposez pour la suite ?")

        # Signals for the decision engine (the demo client reads the advisor with simple keyword rules)
        ds = getattr(state, "decision_state", None) or {}
        missing = [m for m in ds.get("must_haves", []) if m not in ds.get("met", [])]
        signals = {
            "greeting": {"advisor_tone": "chaleureux"},
            "over_budget": {"price_feeling": "trop_cher", "red_flags": ["offre au-dessus du budget"]},
            "value": {"advisor_tone": "chaleureux", "price_feeling": "confort", "must_haves_met": missing[:1],
                      "buying_signal": booking_intent >= 6},
            "recommend": {"advisor_tone": "chaleureux", "must_haves_met": missing[:1]},
            "generic": {"advisor_tone": "neutre"},
            "closing": {"advisor_tone": "chaleureux", "price_feeling": "confort", "must_haves_met": missing,
                        "buying_signal": True, "proposed_outcome": "reserve"},
        }[branch]

        return CustomerTurnResponse(
            customer_response=customer_reply,
            signals=signals,
            current_stage=current_stage,
            trust_level=round(trust, 1),
            interest_level=round(interest, 1),
            booking_intention=round(booking_intent, 1),
            detected_rep_quality=quality,
            addressed_objections=addressed,
            new_unresolved_objections=new_unresolved,
            reasoning=reasoning
        )


# Model -> time (epoch) until which Groq refuses it (daily token quota reached). Shared by all requests of the process.
_EXHAUSTED: dict = {}


def _available_models(models: list) -> list:
    now = time.time()
    return [m for m in models if _EXHAUSTED.get(m, 0) <= now]


class GroqLLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = settings.GROQ_API_KEY if api_key is None else api_key
        preferred = model or settings.GROQ_MODEL
        self.models = [preferred] + [m for m in settings.GROQ_FALLBACK_MODELS if m != preferred]
        self.model = (_available_models(self.models) or [preferred])[0]
        if not self.api_key:
            raise LLMProviderException(
                "Groq API key is missing. Please set GROQ_API_KEY environment variable or in .env file."
            )
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key, timeout=settings.GROQ_TIMEOUT_SECONDS, max_retries=0)
        except ImportError:
            raise LLMProviderException("The 'groq' python package is not installed.")

    def generate_response(
        self,
        profile: CustomerProfileResponse,
        state: CustomerStateSnapshot,
        history_formatted: str,
        latest_rep_message: str,
        hotel_catalog: str = "",
        retry_note: str = "",
    ) -> CustomerTurnResponse:
        system_prompt = build_system_prompt(profile, state, history_formatted, hotel_catalog)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sales Representative: {latest_rep_message}"}
        ]
        if retry_note:
            messages.append({"role": "user", "content": retry_note})
        try:
            data = _parse_json(self._complete(messages, json_mode=True).content)
            if data is None:
                raise ValueError("la réponse du modèle n'est pas un JSON valide")
            return CustomerTurnResponse.model_validate(data)
        except Exception as e:
            logger.error(f"Groq LLM Generation error: {str(e)}")
            raise LLMProviderException(f"Failed to generate response using Groq: {str(e)}")

    RATE_LIMIT_RETRIES = 2
    # The simulator must return promptly. Longer rate limits move to a fallback model or deterministic client.
    MAX_RATE_LIMIT_WAIT = 8  # seconds

    def generate_text(self, prompt: str, temperature: float = 0.9):
        try:
            return (self._complete([{"role": "user", "content": prompt}], temperature=temperature).content or "").strip() or None
        except Exception as exc:
            logger.warning(f"Groq free-text generation failed, using the template: {exc}")
            return None

    TEMPERATURE = 0.85  # client replies: high enough to vary wording between simulations

    def _complete(self, messages, json_mode=False, temperature=None):
        temperature = self.TEMPERATURE if temperature is None else temperature
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        attempt = 0
        while True:
            try:
                kwargs["model"] = self.model
                return self.client.chat.completions.create(**kwargs).choices[0].message
            except Exception as exc:
                wait = _rate_limit_wait(exc)
                if wait is None and _model_unavailable(exc):
                    # A retired model will not come back; skip it for a day.
                    wait = 24 * 3600
                if wait is None:
                    raise
                if wait > self.MAX_RATE_LIMIT_WAIT:
                    # Long limit (tokens per day): this model is out for a while, switch to the next one instead of the demo client
                    _EXHAUSTED[self.model] = time.time() + wait
                    others = _available_models(self.models)
                    if not others:
                        if settings.OPENROUTER_API_KEY:
                            logger.warning(f"All Groq models exhausted; using OpenRouter {settings.OPENROUTER_MODEL}")
                            return _openrouter_complete(messages, json_mode, temperature)
                        raise
                    logger.warning(f"Groq quota reached for {self.model} ({wait / 60:.0f} min), switching to {others[0]}")
                    self.model = others[0]
                    continue
                # Short limit (tokens per minute): wait the delay Groq gives
                attempt += 1
                if attempt > self.RATE_LIMIT_RETRIES:
                    raise
                logger.info(f"Groq rate limit reached, retrying in {wait:.1f}s")
                time.sleep(wait)


def _parse_json(content):
    """Parse the model's JSON answer, tolerating ```json fences or text around the object. None if impossible."""
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _model_unavailable(exc):
    """True for Groq's "model does not exist / no access" error (a retired or unlisted model)."""
    return getattr(exc, "status_code", None) == 404 or "model_not_found" in str(exc)


class _Message:
    def __init__(self, content):
        self.content = content


def _openrouter_complete(messages, json_mode, temperature):
    """One chat completion through OpenRouter's OpenAI-compatible endpoint."""
    import httpx  # installed with the groq SDK
    body = {"model": settings.OPENROUTER_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions", json=body, timeout=settings.GROQ_TIMEOUT_SECONDS + 8,
        headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}", "X-Title": "Jasmin client simulator"},
    )
    response.raise_for_status()
    return _Message(response.json()["choices"][0]["message"].get("content") or "")


def _rate_limit_wait(exc):
    """Seconds to wait if exc is a Groq 429 rate limit ("Please try again in 3.48s"), else None."""
    if getattr(exc, "status_code", None) != 429 and "rate_limit" not in str(exc):
        return None
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc))
    return (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3)) + 0.5) if m else 5.0





def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    provider = (provider_name or settings.LLM_PROVIDER).lower()
    if provider == "groq":
        return GroqLLMProvider()
    elif provider == "mock":
        return MockLLMProvider()
    else:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to MockLLMProvider.")
        return MockLLMProvider()
