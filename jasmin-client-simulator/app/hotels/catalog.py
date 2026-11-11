"""
Tunisian tourist regions the simulated clients travel to.

Clients ask for a region, never a hotel: proposing hotels and prices is the advisor's job.
REGIONS also gives the client what a traveller usually knows about the place, so it can talk about it naturally.
"""
import re
from datetime import date, timedelta
from typing import Optional

DEFAULT_NIGHTS = 7

# Region -> what a traveller usually knows about it (told to the client, in French like the prompt)
REGIONS = {
    "Hammamet": "station balnéaire réputée : plages de sable fin, Yasmine Hammamet et sa marina, médina, thalasso, soirées animées",
    "Nabeul": "ville de la poterie et du marché du vendredi, plages calmes, tout près de Hammamet",
    "Sousse": "médina classée à l'UNESCO, Port El Kantaoui, plages animées, nombreux hôtels-clubs et vie nocturne",
    "Monastir": "ribat et mausolée de Bourguiba, plages de Skanes, aéroport tout proche, ambiance familiale",
    "Mahdia": "plages de sable blanc parmi les plus belles du pays, ville calme et authentique, thalasso",
    "Djerba": "île aux plages de sable blanc, Houmt Souk, Guellala, la Ghriba, parfaite en famille, parc Djerba Explore",
    "Zarzis": "plages tranquilles face à Djerba, oliveraies, séjours reposants",
    "Tunis": "capitale : médina classée à l'UNESCO, musée du Bardo, Carthage et Sidi Bou Saïd tout proches",
    "Gammarth": "banlieue chic au nord de Tunis, hôtels de luxe en bord de mer, à côté de La Marsa et Sidi Bou Saïd",
    "Tozeur": "porte du désert : oasis et palmeraie, chott el-Jérid, décors de Star Wars, hôtels de charme",
    "Douz": "« porte du Sahara » : dunes, balades à dos de dromadaire, nuits sous la tente, festival du Sahara",
    "Tabarka": "côte nord verdoyante : forêts de chênes-lièges, plongée sous-marine, golf, festival de jazz",
}
DESTINATION_CITIES = list(REGIONS)


def profile_city(profile) -> Optional[str]:
    """The client's destination city: first known region named in its wishes ("Hotel X (Djerba)", "Hammamet"...)."""
    text = " ".join(profile.destination_preferences or []).lower()
    hits = [(m.start(), city) for city in DESTINATION_CITIES
            for m in [re.search(rf"\b{re.escape(city.lower())}\b", text)] if m]
    return min(hits)[1] if hits else None


def profile_region(profile) -> str:
    """The tourist region the client asks for (Hammamet, Sousse...)."""
    return profile_city(profile) or "Tunisie"


def region_facts(region: str) -> str:
    return REGIONS.get(region, "la Tunisie : plages, médinas, désert et thalasso")


def profile_dates(profile) -> tuple:
    """(check_in, check_out) from the profile; a past or invalid trip is moved to a month from now, same length."""
    try:
        d = profile.travel_dates if isinstance(profile.travel_dates, dict) else profile.travel_dates.model_dump()
        check_in, check_out = date.fromisoformat(str(d["start_date"])), date.fromisoformat(str(d["end_date"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        check_in, check_out = date.today() + timedelta(days=30), date.today() + timedelta(days=30 + DEFAULT_NIGHTS)
    nights = min(30, max(1, (check_out - check_in).days))
    if check_in <= date.today():
        check_in = date.today() + timedelta(days=30)
    return check_in, check_in + timedelta(days=nights)


def trip_nights(travel_dates) -> int:
    try:
        d = travel_dates if isinstance(travel_dates, dict) else travel_dates.model_dump()
        n = (date.fromisoformat(str(d["end_date"])) - date.fromisoformat(str(d["start_date"]))).days
        return max(1, n)
    except (KeyError, ValueError, TypeError, AttributeError):
        return DEFAULT_NIGHTS


# Places a client can name (persona creation refuses anything else)
TUNISIAN_PLACES = [
    "tunisie", "tunisia", "tunis", "gammarth", "carthage", "sidi bou", "la marsa", "hammamet", "yasmine", "nabeul",
    "korba", "kelibia", "sousse", "kantaoui", "monastir", "skanes", "mahdia", "djerba", "jerba", "zarzis", "tozeur",
    "nefta", "douz", "tabarka", "bizerte", "kairouan", "el jem", "sfax", "kerkennah", "tataouine", "matmata", "enfidha",
]


def is_tunisian_place(text: str, db=None) -> bool:
    t = (text or "").lower().strip()
    return bool(t) and any(p in t for p in TUNISIAN_PLACES)
