"""
Scenarios, moods, surprise events and difficulty: what makes two simulations of the same persona differ.

Everything random is drawn once at the start from a seed stored on the simulation,
so a simulation can be replayed with exactly the same mood and events.
"""
import random
from datetime import date, timedelta

# situation: told to the client (French prompt). first_topic: the ONE thing its first message is about.
# opening_*: first words when the AI opening is unavailable.
SCENARIOS = {
    "premiere_demande": {
        "label": "Première demande",
        "first_topic": "ta demande : la région où tu veux aller et tes dates",
        "goal": "trouver le bon séjour et décider si tu réserves",
        "situation": "Tu contactes l'agence pour la première fois pour ce séjour.",
        "opening_fr": "", "opening_en": "",
        "patience": 0, "trust": 0.0,
    },
    "rappel_devis": {
        "label": "Rappel après devis",
        "first_topic": "le fait que tu rappelles au sujet du devis reçu (sans encore parler de l'offre concurrente)",
        "goal": "dire d'emblée que tu rappelles pour le devis, parler de l'offre concurrente moins chère et voir si l'agence s'aligne",
        "situation": ("L'agence t'a envoyé un devis il y a trois jours pour un séjour dans ta région. Entre-temps, tu as vu une offre "
                      "concurrente un peu moins chère. Tu rappelles pour décider : sois prêt(e) à comparer."),
        "opening_fr": "Bonjour, je vous rappelle pour le devis que vous m'avez envoyé.",
        "opening_en": "Hello, I'm calling back about the quote you sent me.",
        "patience": -1, "trust": 0.5,
    },
    "derniere_minute": {
        "label": "Dernière minute",
        "first_topic": "ton départ dans 48 heures pour la région où tu veux aller : est-ce encore possible ?",
        "goal": "dire tout de suite que tu pars dans 48 heures et savoir vite ce qui est encore disponible",
        "situation": ("Tu veux partir dans 48 heures. Tu dois décider vite et tu as peur qu'il ne reste plus de chambres. "
                      "Tu acceptes plus facilement un compromis si on te propose une solution tout de suite."),
        "opening_fr": "Bonjour, c'est un peu urgent : je voudrais partir après-demain.",
        "opening_en": "Hello, it's a bit urgent: I'd like to leave the day after tomorrow.",
        "patience": -2, "trust": 0.0, "threshold": -0.1,
    },
    "modification": {
        "label": "Modification de réservation",
        "first_topic": "le fait que tu as déjà réservé chez eux et que tu voudrais modifier ton séjour (sans encore dire quoi)",
        "goal": "expliquer que tu as déjà réservé et que tu veux décaler d'un jour et passer en demi-pension, puis savoir combien ça coûte",
        "situation": ("Tu as déjà réservé ce séjour avec l'agence. Tu appelles pour décaler l'arrivée d'un jour et passer à la "
                      "demi-pension, sans que ça te coûte trop cher. Si c'est trop compliqué, tu envisages d'annuler."),
        "opening_fr": "Bonjour, j'ai déjà réservé chez vous et j'aurais besoin de modifier mon séjour.",
        "opening_en": "Hello, I've already booked with you and I'd need to change my stay.",
        "patience": 0, "trust": 1.0,
    },
    "reclamation": {
        "label": "Réclamation puis nouveau séjour",
        "first_topic": "UNIQUEMENT ta réclamation sur le dernier séjour (ce qui s'est mal passé) ; ne parle PAS encore du nouveau séjour : ni région, ni dates, ni budget",
        "goal": "COMMENCER par raconter ce qui s'est mal passé lors du dernier séjour et obtenir des excuses sincères et un geste ; tu ne parles du nouveau séjour qu'une fois écouté(e)",
        "situation": ("Ton dernier séjour organisé par l'agence s'est mal passé (chambre bruyante, transfert en retard d'une heure). "
                      "Tu es encore fâché(e) : avant de leur refaire confiance pour ce nouveau séjour, tu veux être écouté(e) "
                      "et obtenir un geste commercial. Tu te calmes si l'agent reconnaît le problème sincèrement."),
        "opening_fr": "Bonjour. Je vous préviens, mon dernier séjour avec vous ne s'est pas bien passé du tout.",
        "opening_en": "Hello. I'll be upfront: my last trip with you didn't go well at all.",
        "patience": -1, "trust": -2.0,
    },
    "client_fidele": {
        "label": "Client fidèle",
        "first_topic": "le fait que tu es déjà client(e) et que tu reviens pour un nouveau séjour dans ta région préférée",
        "goal": "rappeler que tu es déjà client(e), être reconnu(e), et voir ce qu'on te propose de mieux que d'habitude",
        "situation": ("C'est ton troisième voyage avec l'agence et tu en es content(e). Tu es ouvert(e) à monter en gamme "
                      "(chambre supérieure, excursion) si on te le propose bien, mais tu attends d'être reconnu(e)."),
        "opening_fr": "Bonjour, c'est encore moi ! On a tellement aimé nos derniers séjours qu'on revient.",
        "opening_en": "Hello, it's me again! We loved our last trips so much that we're back.",
        "patience": 2, "trust": 1.5,
    },
    "groupe": {
        "label": "Groupe d'amis",
        "first_topic": "le fait que tu organises un séjour pour un groupe d'amis, et la région visée",
        "goal": "expliquer que tu organises pour un groupe aux envies différentes et trouver une offre qui mette tout le monde d'accord",
        "situation": ("Tu organises pour un groupe d'amis. Chacun a son avis : l'un veut de l'animation, l'autre le calme, "
                      "un troisième le prix le plus bas. Tu dois revenir vers eux avec une proposition qui fait consensus, "
                      "tu ne peux donc pas réserver seul(e) sans leur accord."),
        "opening_fr": "Bonjour, j'organise un séjour pour un groupe d'amis et ce n'est pas simple de mettre tout le monde d'accord.",
        "opening_en": "Hello, I'm organising a trip for a group of friends and it's not easy to get everyone to agree.",
        "patience": 1, "trust": 0.0, "decides_alone": False,
    },
}
DEFAULT_SCENARIO = "premiere_demande"

MOODS = [
    {"key": "bonne_humeur", "text": "Aujourd'hui tu es de bonne humeur et détendu(e).", "patience": 1, "trust": 0.5},
    {"key": "fatigue", "text": "Tu sors d'une longue journée de travail : tu es fatigué(e) et un peu à cran.", "patience": -1, "trust": 0.0},
    {"key": "presse", "text": "Tu as peu de temps aujourd'hui, tu veux avancer.", "patience": -2, "trust": 0.0},
    {"key": "enthousiaste", "text": "Tu rêves déjà de ce voyage, tu es enthousiaste.", "patience": 1, "trust": 0.5},
    {"key": "mefiant", "text": "Tu viens de lire des avis négatifs sur une agence de voyages : tu es sur tes gardes.", "patience": 0, "trust": -1.0},
    {"key": "neutre", "text": "Tu es dans ton état normal.", "patience": 0, "trust": 0.0},
]

# Surprises the client brings up at a given turn (scenarios=None: any scenario)
EVENTS = [
    {"text": "Tu viens de voir le même hôtel environ 10 % moins cher sur un comparateur en ligne : parles-en à l'agent.", "scenarios": None},
    {"text": "Ton/ta conjoint(e) (ou ton compagnon de voyage) t'envoie un message : il/elle préférerait la demi-pension. Demande ce que ça change.", "scenarios": None},
    {"text": "Tu dois finalement raccourcir le séjour d'une nuit : demande ce que ça change au prix.", "scenarios": None},
    {"text": "Une personne de plus voudrait se joindre au voyage : demande si c'est possible et à quel prix.", "scenarios": None},
    {"text": "Ton téléphone sonne : tu vas devoir raccrocher bientôt, demande à l'agent d'aller à l'essentiel.", "scenarios": None},
    {"text": "Un ami t'a dit que cet hôtel était bruyant le soir : demande ce qu'il en est.", "scenarios": None},
    {"text": "Tu demandes un geste commercial pour compenser le dernier séjour raté.", "scenarios": ["reclamation"]},
]

# Season -> months, and what it changes for the traveller (told to the client)
SEASONS = {
    "hiver": ([12, 1, 2], "hiver : basse saison, prix plus doux, mer fraîche ; idéal pour le désert et le sud (Tozeur, Douz, Djerba)"),
    "printemps": ([3, 4, 5], "printemps : temps doux et ensoleillé, nature en fleurs, idéal pour visiter et se balader"),
    "ete": ([6, 7, 8], "été : haute saison, forte chaleur, plages très animées, prix plus élevés, il faut réserver tôt"),
    "automne": ([9, 10, 11], "automne : mer encore chaude, moins de monde, arrière-saison aux prix intéressants"),
}
MIN_DAYS_AHEAD, MAX_DAYS_AHEAD = 10, 240  # trips start between 10 days and ~8 months from today
# Families travel during school holidays
SEASONS_BY_ARCHETYPE = {"family": ["printemps", "ete"]}


def draw_trip(rng: random.Random, archetype: str, nights: int, today: date = None) -> dict:
    """A season and real dates in it, in the coming months (never past, never a year away), keeping the trip length."""
    today = today or date.today()
    window = [today + timedelta(days=d) for d in range(MIN_DAYS_AHEAD, MAX_DAYS_AHEAD + 1)]
    seasons = [s for s in SEASONS_BY_ARCHETYPE.get(archetype, list(SEASONS))
               if any(d.month in SEASONS[s][0] for d in window)]  # seasons reachable in the window
    season = rng.choice(seasons)
    check_in = rng.choice([d for d in window if d.month in SEASONS[season][0]])
    nights = max(1, min(30, nights))
    return {"season": season, "season_text": SEASONS[season][1], "check_in": check_in.isoformat(),
            "check_out": (check_in + timedelta(days=nights)).isoformat()}


def trip_dates(ds: dict, profile=None) -> tuple:
    """The simulation's dates (drawn at the start), else the persona's own dates."""
    if ds and ds.get("check_in"):
        return date.fromisoformat(ds["check_in"]), date.fromisoformat(ds["check_out"])
    from app.hotels.catalog import profile_dates
    return profile_dates(profile)


DIFFICULTIES = {
    "facile": {"label": "Facile", "patience": 3, "threshold": -0.1, "trust": 1.0, "events": 0},
    "normal": {"label": "Normal", "patience": 0, "threshold": 0.0, "trust": 0.0, "events": 1},
    "difficile": {"label": "Difficile", "patience": -2, "threshold": 0.1, "trust": -1.0, "events": 2},
}


def setup(scenario: str, difficulty: str, seed: int, archetype: str = "satisfied", nights: int = 7) -> dict:
    """Draw mood, season, dates and events for a new simulation. Returns what the decision engine starts from."""
    rng = random.Random(seed)
    trip = draw_trip(random.Random(f"trip-{seed}"), archetype, nights)
    if scenario == "derniere_minute":  # "je pars dans 48 heures"
        start = date.today() + timedelta(days=2)
        season = next(s for s, (months, _) in SEASONS.items() if start.month in months)
        trip = {"season": season, "season_text": SEASONS[season][1], "check_in": start.isoformat(),
                "check_out": (start + timedelta(days=max(1, min(30, nights)))).isoformat()}
    sc, diff = SCENARIOS[scenario], DIFFICULTIES[difficulty]
    mood = rng.choice(MOODS)
    pool = [e["text"] for e in EVENTS if e["scenarios"] is None or scenario in e["scenarios"]]
    texts = rng.sample(pool, k=min(diff["events"], len(pool)))
    turns = sorted(rng.sample(range(2, 7), k=len(texts)))
    return {
        "scenario": scenario, "difficulty": difficulty, "seed": seed, "mood": mood["key"], "mood_text": mood["text"], **trip,
        "events": {str(t): text for t, text in zip(turns, texts)},  # advisor turn number -> event
        "patience_delta": sc["patience"] + mood["patience"] + diff["patience"],
        "trust_delta": sc["trust"] + mood["trust"] + diff["trust"],
        "threshold_delta": sc.get("threshold", 0.0) + diff["threshold"],
        "decides_alone": sc.get("decides_alone", True),
    }


if __name__ == "__main__":
    a, b = setup("reclamation", "difficile", 42), setup("reclamation", "difficile", 42)
    assert a == b and len(a["events"]) == 2  # same seed, same simulation
    assert setup("premiere_demande", "facile", 1)["events"] == {}
    assert not setup("groupe", "normal", 3)["decides_alone"]
    seasons = {draw_trip(random.Random(i), "satisfied", 7)["season"] for i in range(40)}
    assert seasons == set(SEASONS)  # every season comes up
    assert {draw_trip(random.Random(i), "family", 7)["season"] for i in range(40)} <= {"printemps", "ete"}
    for i in range(200):
        t = draw_trip(random.Random(i), "family" if i % 2 else "satisfied", 5, today=date(2026, 10, 30))
        start = date.fromisoformat(t["check_in"])
        assert date(2026, 11, 9) <= start <= date(2027, 6, 27), start  # never past, never a year away
        assert (date.fromisoformat(t["check_out"]) - start).days == 5
    print("scenarios ok")
