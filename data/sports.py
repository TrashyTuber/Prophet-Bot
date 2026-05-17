"""Fetch sports data from The Odds API and free stats APIs.

Provides consensus bookmaker odds, recent team stats, and power ratings
for sports prediction markets.

Requires THE_ODDS_API_KEY in environment (free at https://the-odds-api.com).
"""

import logging
import math
import os
import time
from datetime import datetime, timezone

import httpx
from tavily import TavilyClient

log = logging.getLogger(__name__)

ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
_tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_CACHE_TTL = 4 * 3600  # 4 hours — championship odds barely move

OUTRIGHT_SPORT_KEYS = {
    "basketball_nba": "basketball_nba_championship_winner",
    "icehockey_nhl": "icehockey_nhl_championship_winner",
    "baseball_mlb": "baseball_mlb_world_series_winner",
    "americanfootball_nfl": "americanfootball_nfl_super_bowl_winner",
    "soccer_epl": "soccer_epl_winner",
    "soccer_spain_la_liga": "soccer_spain_la_liga_winner",
    "soccer_germany_bundesliga": "soccer_germany_bundesliga_winner",
    "soccer_italy_serie_a": "soccer_italy_serie_a_winner",
    "soccer_uefa_champs_league": "soccer_uefa_champs_league_winner",
    "soccer_fifa_world_cup": "soccer_fifa_world_cup_winner",
}

CHAMPIONSHIP_KEYWORDS = [
    "win the", "champion", "title", "finals", "world series",
    "stanley cup", "super bowl", "conference", "premier league",
    "la liga", "serie a", "bundesliga", "world cup",
]

_odds_cache: dict[str, tuple[float, list]] = {}  # sport_key -> (timestamp, events)
_ratings_cache: dict[str, tuple[float, dict]] = {}  # "nba"/"nhl"/"mlb" -> (timestamp, ratings)
RATINGS_TTL = 6 * 3600

SPORT_KEYWORDS = {
    "nba": {
        "odds_key": "basketball_nba",
        "label": "NBA",
    },
    "basketball": {
        "odds_key": "basketball_nba",
        "label": "NBA",
    },
    "nfl": {
        "odds_key": "americanfootball_nfl",
        "label": "NFL",
    },
    "football": {
        "odds_key": "americanfootball_nfl",
        "label": "NFL",
    },
    "mlb": {
        "odds_key": "baseball_mlb",
        "label": "MLB",
    },
    "baseball": {
        "odds_key": "baseball_mlb",
        "label": "MLB",
    },
    "nhl": {
        "odds_key": "icehockey_nhl",
        "label": "NHL",
    },
    "hockey": {
        "odds_key": "icehockey_nhl",
        "label": "NHL",
    },
    "soccer": {
        "odds_key": "soccer_usa_mls",
        "label": "MLS",
    },
    "mls": {
        "odds_key": "soccer_usa_mls",
        "label": "MLS",
    },
    "premier league": {
        "odds_key": "soccer_epl",
        "label": "English Premier League",
    },
    "epl": {
        "odds_key": "soccer_epl",
        "label": "English Premier League",
    },
    "champions league": {
        "odds_key": "soccer_uefa_champs_league",
        "label": "UEFA Champions League",
    },
    "ucl": {
        "odds_key": "soccer_uefa_champs_league",
        "label": "UEFA Champions League",
    },
    "la liga": {
        "odds_key": "soccer_spain_la_liga",
        "label": "La Liga",
    },
    "serie a": {
        "odds_key": "soccer_italy_serie_a",
        "label": "Serie A",
    },
    "bundesliga": {
        "odds_key": "soccer_germany_bundesliga",
        "label": "Bundesliga",
    },
    "ufc": {
        "odds_key": "mma_mixed_martial_arts",
        "label": "UFC/MMA",
    },
    "mma": {
        "odds_key": "mma_mixed_martial_arts",
        "label": "UFC/MMA",
    },
    "tennis": {
        "odds_key": "tennis_atp_french_open",
        "odds_keys_fallback": [
            "tennis_atp_wimbledon",
            "tennis_atp_us_open",
            "tennis_atp_australian_open",
            "tennis_wta_french_open",
            "tennis_wta_wimbledon",
            "tennis_wta_us_open",
            "tennis_wta_australian_open",
        ],
        "label": "Tennis",
    },
    "atp": {
        "odds_key": "tennis_atp_french_open",
        "odds_keys_fallback": [
            "tennis_atp_wimbledon",
            "tennis_atp_us_open",
            "tennis_atp_australian_open",
        ],
        "label": "ATP Tennis",
    },
    "wta": {
        "odds_key": "tennis_wta_french_open",
        "odds_keys_fallback": [
            "tennis_wta_wimbledon",
            "tennis_wta_us_open",
            "tennis_wta_australian_open",
        ],
        "label": "WTA Tennis",
    },
    "wimbledon": {
        "odds_key": "tennis_atp_wimbledon",
        "label": "Wimbledon",
    },
    "french open": {
        "odds_key": "tennis_atp_french_open",
        "label": "French Open",
    },
    "us open": {
        "odds_key": "tennis_atp_us_open",
        "label": "US Open",
    },
    "australian open": {
        "odds_key": "tennis_atp_australian_open",
        "label": "Australian Open",
    },
    "ncaa": {
        "odds_key": "basketball_ncaab",
        "label": "NCAA Basketball",
    },
    "college basketball": {
        "odds_key": "basketball_ncaab",
        "label": "NCAA Basketball",
    },
    "college football": {
        "odds_key": "americanfootball_ncaaf",
        "label": "NCAA Football",
    },
    "world cup": {
        "odds_key": "soccer_fifa_world_cup",
        "label": "FIFA World Cup",
    },
}

NBA_TEAMS = {
    "lakers": "Los Angeles Lakers",
    "celtics": "Boston Celtics",
    "warriors": "Golden State Warriors",
    "bucks": "Milwaukee Bucks",
    "76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "nuggets": "Denver Nuggets",
    "heat": "Miami Heat",
    "suns": "Phoenix Suns",
    "knicks": "New York Knicks",
    "nets": "Brooklyn Nets",
    "clippers": "Los Angeles Clippers",
    "mavericks": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",
    "grizzlies": "Memphis Grizzlies",
    "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",
    "kings": "Sacramento Kings",
    "timberwolves": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "thunder": "Oklahoma City Thunder",
    "pelicans": "New Orleans Pelicans",
    "hawks": "Atlanta Hawks",
    "raptors": "Toronto Raptors",
    "bulls": "Chicago Bulls",
    "pacers": "Indiana Pacers",
    "magic": "Orlando Magic",
    "spurs": "San Antonio Spurs",
    "trail blazers": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",
    "rockets": "Houston Rockets",
    "jazz": "Utah Jazz",
    "pistons": "Detroit Pistons",
    "hornets": "Charlotte Hornets",
    "wizards": "Washington Wizards",
}

NBA_TEAM_ABBREVS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
NBA_FULLNAME_TO_ABBREV = {v: k for k, v in NBA_TEAM_ABBREVS.items()}

NBA_CONFERENCES = {
    "East": ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
             "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
             "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
             "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards"],
    "West": ["Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets",
             "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
             "Minnesota Timberwolves", "New Orleans Pelicans", "Oklahoma City Thunder",
             "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
             "San Antonio Spurs", "Utah Jazz"],
}

MLB_TEAMS = {
    "yankees": "New York Yankees",
    "dodgers": "Los Angeles Dodgers",
    "astros": "Houston Astros",
    "braves": "Atlanta Braves",
    "mets": "New York Mets",
    "phillies": "Philadelphia Phillies",
    "padres": "San Diego Padres",
    "red sox": "Boston Red Sox",
    "cubs": "Chicago Cubs",
    "white sox": "Chicago White Sox",
    "guardians": "Cleveland Guardians",
    "rangers": "Texas Rangers",
    "mariners": "Seattle Mariners",
    "orioles": "Baltimore Orioles",
    "rays": "Tampa Bay Rays",
    "twins": "Minnesota Twins",
    "blue jays": "Toronto Blue Jays",
    "cardinals": "St. Louis Cardinals",
    "brewers": "Milwaukee Brewers",
    "giants": "San Francisco Giants",
    "angels": "Los Angeles Angels",
    "tigers": "Detroit Tigers",
    "reds": "Cincinnati Reds",
    "diamondbacks": "Arizona Diamondbacks",
    "d-backs": "Arizona Diamondbacks",
    "pirates": "Pittsburgh Pirates",
    "royals": "Kansas City Royals",
    "rockies": "Colorado Rockies",
    "nationals": "Washington Nationals",
    "marlins": "Miami Marlins",
    "athletics": "Oakland Athletics",
}

NFL_TEAMS = {
    "chiefs": "Kansas City Chiefs",
    "eagles": "Philadelphia Eagles",
    "bills": "Buffalo Bills",
    "49ers": "San Francisco 49ers",
    "niners": "San Francisco 49ers",
    "cowboys": "Dallas Cowboys",
    "ravens": "Baltimore Ravens",
    "lions": "Detroit Lions",
    "dolphins": "Miami Dolphins",
    "bengals": "Cincinnati Bengals",
    "packers": "Green Bay Packers",
    "jets": "New York Jets",
    "steelers": "Pittsburgh Steelers",
    "bears": "Chicago Bears",
    "broncos": "Denver Broncos",
    "chargers": "Los Angeles Chargers",
    "seahawks": "Seattle Seahawks",
    "vikings": "Minnesota Vikings",
    "saints": "New Orleans Saints",
    "falcons": "Atlanta Falcons",
    "raiders": "Las Vegas Raiders",
    "colts": "Indianapolis Colts",
    "titans": "Tennessee Titans",
    "texans": "Houston Texans",
    "jaguars": "Jacksonville Jaguars",
    "patriots": "New England Patriots",
    "commanders": "Washington Commanders",
    "panthers": "Carolina Panthers",
    "browns": "Cleveland Browns",
    "cardinals": "Arizona Cardinals",
    "rams": "Los Angeles Rams",
    "giants": "New York Giants",
    "buccaneers": "Tampa Bay Buccaneers",
    "bucs": "Tampa Bay Buccaneers",
}

NHL_TEAMS = {
    "avalanche": "Colorado Avalanche",
    "jets": "Winnipeg Jets",
    "hurricanes": "Carolina Hurricanes",
    "panthers": "Florida Panthers",
    "stars": "Dallas Stars",
    "oilers": "Edmonton Oilers",
    "maple leafs": "Toronto Maple Leafs",
    "leafs": "Toronto Maple Leafs",
    "bruins": "Boston Bruins",
    "wild": "Minnesota Wild",
    "lightning": "Tampa Bay Lightning",
    "canucks": "Vancouver Canucks",
    "blue jackets": "Columbus Blue Jackets",
    "predators": "Nashville Predators",
    "preds": "Nashville Predators",
    "flames": "Calgary Flames",
    "golden knights": "Vegas Golden Knights",
    "penguins": "Pittsburgh Penguins",
    "red wings": "Detroit Red Wings",
    "sabres": "Buffalo Sabres",
    "canadiens": "Montréal Canadiens",
    "habs": "Montréal Canadiens",
    "senators": "Ottawa Senators",
    "sens": "Ottawa Senators",
    "kraken": "Seattle Kraken",
    "islanders": "New York Islanders",
    "blackhawks": "Chicago Blackhawks",
    "ducks": "Anaheim Ducks",
    "coyotes": "Utah Hockey Club",
    "blues": "St. Louis Blues",
    "sharks": "San Jose Sharks",
    "devils": "New Jersey Devils",
    "flyers": "Philadelphia Flyers",
    "capitals": "Washington Capitals",
    "caps": "Washington Capitals",
}

WORLD_CUP_TEAMS = {
    "argentina": "Argentina",
    "brazil": "Brazil",
    "france": "France",
    "england": "England",
    "spain": "Spain",
    "germany": "Germany",
    "portugal": "Portugal",
    "netherlands": "Netherlands",
    "belgium": "Belgium",
    "italy": "Italy",
    "croatia": "Croatia",
    "uruguay": "Uruguay",
    "colombia": "Colombia",
    "mexico": "Mexico",
    "usa": "United States",
    "united states": "United States",
    "japan": "Japan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "australia": "Australia",
    "senegal": "Senegal",
    "morocco": "Morocco",
    "nigeria": "Nigeria",
    "cameroon": "Cameroon",
    "ghana": "Ghana",
    "ecuador": "Ecuador",
    "saudi arabia": "Saudi Arabia",
    "iran": "Iran",
    "canada": "Canada",
    "poland": "Poland",
    "denmark": "Denmark",
    "switzerland": "Switzerland",
    "serbia": "Serbia",
}

# FIFA ranking points (approximate as of early 2026) — higher = better
FIFA_RANKINGS = {
    "Argentina": 1, "France": 2, "Brazil": 3, "England": 4, "Belgium": 5,
    "Spain": 6, "Netherlands": 7, "Portugal": 8, "Italy": 9, "Germany": 10,
    "Croatia": 11, "Uruguay": 12, "Colombia": 13, "Mexico": 14, "United States": 15,
    "Morocco": 16, "Switzerland": 17, "Japan": 18, "Denmark": 19, "Senegal": 20,
    "Poland": 21, "South Korea": 22, "Australia": 23, "Nigeria": 24, "Ecuador": 25,
    "Serbia": 26, "Iran": 27, "Ghana": 28, "Cameroon": 29, "Canada": 30,
    "Saudi Arabia": 31,
}

# Historical World Cup performance — weighted recent tournaments more heavily
# Format: {"titles": int, "finals": int, "semis": int, "quarters": int, "appearances": int}
WORLD_CUP_HISTORY = {
    "Argentina": {"titles": 3, "finals": 6, "semis": 7, "quarters": 11, "appearances": 18},
    "Brazil": {"titles": 5, "finals": 7, "semis": 12, "quarters": 15, "appearances": 22},
    "France": {"titles": 2, "finals": 3, "semis": 5, "quarters": 8, "appearances": 16},
    "Germany": {"titles": 4, "finals": 8, "semis": 13, "quarters": 16, "appearances": 20},
    "Italy": {"titles": 4, "finals": 6, "semis": 8, "quarters": 10, "appearances": 18},
    "England": {"titles": 1, "finals": 1, "semis": 3, "quarters": 8, "appearances": 16},
    "Spain": {"titles": 1, "finals": 1, "semis": 2, "quarters": 5, "appearances": 16},
    "Netherlands": {"titles": 0, "finals": 3, "semis": 5, "quarters": 7, "appearances": 11},
    "Uruguay": {"titles": 2, "finals": 2, "semis": 5, "quarters": 7, "appearances": 14},
    "Croatia": {"titles": 0, "finals": 2, "semis": 3, "quarters": 3, "appearances": 6},
    "Portugal": {"titles": 0, "finals": 0, "semis": 2, "quarters": 4, "appearances": 8},
    "Belgium": {"titles": 0, "finals": 0, "semis": 2, "quarters": 3, "appearances": 14},
    "Morocco": {"titles": 0, "finals": 0, "semis": 1, "quarters": 1, "appearances": 6},
    "Colombia": {"titles": 0, "finals": 0, "semis": 0, "quarters": 2, "appearances": 6},
    "Mexico": {"titles": 0, "finals": 0, "semis": 0, "quarters": 2, "appearances": 17},
    "South Korea": {"titles": 0, "finals": 0, "semis": 1, "quarters": 1, "appearances": 11},
    "Japan": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 7},
    "United States": {"titles": 0, "finals": 0, "semis": 1, "quarters": 1, "appearances": 11},
    "Senegal": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 3},
    "Switzerland": {"titles": 0, "finals": 0, "semis": 0, "quarters": 2, "appearances": 12},
    "Denmark": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 6},
    "Poland": {"titles": 0, "finals": 0, "semis": 2, "quarters": 2, "appearances": 9},
    "Ecuador": {"titles": 0, "finals": 0, "semis": 0, "quarters": 0, "appearances": 4},
    "Serbia": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 3},
    "Australia": {"titles": 0, "finals": 0, "semis": 0, "quarters": 0, "appearances": 6},
    "Nigeria": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 7},
    "Ghana": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 4},
    "Cameroon": {"titles": 0, "finals": 0, "semis": 0, "quarters": 1, "appearances": 8},
    "Iran": {"titles": 0, "finals": 0, "semis": 0, "quarters": 0, "appearances": 6},
    "Canada": {"titles": 0, "finals": 0, "semis": 0, "quarters": 0, "appearances": 2},
    "Saudi Arabia": {"titles": 0, "finals": 0, "semis": 0, "quarters": 0, "appearances": 7},
}

# Squad quality: average club strength based on league placement of key players' clubs
# Scale 0-1: 1.0 = squad full of Champions League winners, 0.0 = all from lower leagues
# Based on 2025-26 squad projections and current club form
SQUAD_QUALITY = {
    "Argentina": 0.88,  # Messi retired but strong spine at top clubs (PL, La Liga, Serie A)
    "Brazil": 0.90,     # Deep talent pool across Europe's top leagues
    "France": 0.93,     # Mbappé, depth across PL/La Liga/Ligue 1 elite
    "England": 0.91,    # Mostly Premier League starters at top-6 clubs
    "Spain": 0.87,      # La Liga core + PL contingent (Rodri, etc.)
    "Germany": 0.84,    # Bundesliga + PL representation
    "Portugal": 0.85,   # Strong PL/La Liga presence
    "Netherlands": 0.82, # Spread across PL, Bundesliga, Serie A
    "Belgium": 0.78,    # Aging golden generation, fewer top-club starters
    "Italy": 0.80,      # Serie A dominated, fewer abroad at elite level
    "Croatia": 0.79,    # Key players at Real Madrid, PL clubs
    "Uruguay": 0.76,    # Valverde, Núñez + Liga depth
    "Colombia": 0.73,   # Rising talent in PL and Serie A
    "Morocco": 0.72,    # PSG, PL, La Liga contingent post-2022 breakout
    "Mexico": 0.58,     # Mostly Liga MX, few in Europe
    "United States": 0.68, # Growing PL/Bundesliga contingent (Pulisic, McKennie, etc.)
    "Japan": 0.70,      # Strong Bundesliga/PL presence
    "South Korea": 0.63, # Son + smaller European contingent
    "Senegal": 0.71,    # PL, Ligue 1 representation
    "Denmark": 0.72,    # PL presence (Højlund, Eriksen, etc.)
    "Switzerland": 0.68, # Bundesliga/Serie A representation
    "Poland": 0.65,     # Lewandowski still anchors, but aging squad
    "Nigeria": 0.66,    # PL and Serie A scattered
    "Ecuador": 0.60,    # Growing European presence
    "Serbia": 0.67,     # PL and Serie A players
    "Australia": 0.55,  # Mostly domestic + lower European leagues
    "Ghana": 0.58,      # PL fringe + Bundesliga
    "Cameroon": 0.57,   # Ligue 1 + PL fringe
    "Iran": 0.45,       # Mostly domestic league
    "Canada": 0.62,     # MLS + growing European contingent (David, Davies)
    "Saudi Arabia": 0.42, # Mostly Saudi Pro League
}

WORLD_CUP_CONFEDERATIONS = {
    "UEFA": ["France", "England", "Spain", "Germany", "Portugal", "Netherlands",
             "Belgium", "Italy", "Croatia", "Denmark", "Switzerland", "Serbia", "Poland"],
    "CONMEBOL": ["Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador"],
    "CONCACAF": ["Mexico", "United States", "Canada"],
    "AFC": ["Japan", "South Korea", "Australia", "Iran", "Saudi Arabia"],
    "CAF": ["Morocco", "Senegal", "Nigeria", "Cameroon", "Ghana"],
}

ALL_TEAMS = {**NBA_TEAMS, **MLB_TEAMS, **NFL_TEAMS, **NHL_TEAMS, **WORLD_CUP_TEAMS}

MARKET_CITY_TO_TEAM = {
    "oklahoma city": ("Oklahoma City Thunder", "nba"),
    "san antonio": ("San Antonio Spurs", "nba"),
    "new york": ("New York Knicks", "nba"),
    "cleveland": ("Cleveland Cavaliers", "nba"),
    "detroit": ("Detroit Pistons", "nba"),
    "los angeles d": ("Los Angeles Dodgers", "mlb"),
    "los angeles r": ("Los Angeles Rams", "nfl"),
    "los angeles c": ("Los Angeles Chargers", "nfl"),
    "new york y": ("New York Yankees", "mlb"),
    "new york g": ("New York Giants", "nfl"),
    "chicago c": ("Chicago Cubs", "mlb"),
    "chicago ws": ("Chicago White Sox", "mlb"),
    "tampa bay": ("Tampa Bay Rays", "mlb"),
    "colorado avalanche": ("Colorado Avalanche", "nhl"),
    "carolina hurricanes": ("Carolina Hurricanes", "nhl"),
    "vegas golden knights": ("Vegas Golden Knights", "nhl"),
    "buffalo sabres": ("Buffalo Sabres", "nhl"),
    "montréal canadiens": ("Montréal Canadiens", "nhl"),
    "montreal canadiens": ("Montréal Canadiens", "nhl"),
}


def _detect_sport(question: str) -> dict | None:
    question_lower = question.lower()
    for keyword, info in SPORT_KEYWORDS.items():
        if keyword in question_lower:
            return info
    for team in ALL_TEAMS:
        if team in question_lower:
            if team in NBA_TEAMS:
                return SPORT_KEYWORDS["nba"]
            if team in MLB_TEAMS:
                return SPORT_KEYWORDS["mlb"]
            if team in NFL_TEAMS:
                return SPORT_KEYWORDS["nfl"]
            if team in NHL_TEAMS:
                return SPORT_KEYWORDS["nhl"]
            if team in WORLD_CUP_TEAMS:
                return SPORT_KEYWORDS["world cup"]
    for city, (_, sport) in MARKET_CITY_TO_TEAM.items():
        if city in question_lower:
            return SPORT_KEYWORDS[sport]
    return None


def _extract_teams(question: str) -> list[str]:
    question_lower = question.lower()
    found = []
    for nickname, full_name in ALL_TEAMS.items():
        if nickname in question_lower and full_name not in found:
            found.append(full_name)
    for city, (full_name, _) in MARKET_CITY_TO_TEAM.items():
        if city in question_lower and full_name not in found:
            found.append(full_name)
    return found


def _american_odds_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _fetch_odds_raw(sport_key: str) -> list:
    """Fetch raw odds events for a sport key, with caching."""
    now = time.time()
    cached = _odds_cache.get(sport_key)
    if cached and (now - cached[0]) < ODDS_CACHE_TTL:
        log.debug("Odds cache hit for %s (age=%.0fm)", sport_key, (now - cached[0]) / 60)
        return cached[1]

    if not ODDS_API_KEY:
        log.warning("THE_ODDS_API_KEY not set — skipping odds fetch")
        return []

    try:
        resp = httpx.get(
            f"{ODDS_BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,eu",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info("Odds API: fetched %s (%d events, %s requests remaining)", sport_key, len(events), remaining)
    except Exception as e:
        log.warning("Odds API error for %s: %s", sport_key, e)
        return cached[1] if cached else []

    _odds_cache[sport_key] = (now, events)
    return events


def _is_championship_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CHAMPIONSHIP_KEYWORDS)


def _fetch_outright_odds(sport_key: str, teams: list[str]) -> str | None:
    outright_key = OUTRIGHT_SPORT_KEYS.get(sport_key)
    if not outright_key or not ODDS_API_KEY:
        return None

    now = time.time()
    cache_key = f"outright_{outright_key}"
    cached = _odds_cache.get(cache_key)
    if cached and (now - cached[0]) < ODDS_CACHE_TTL:
        events = cached[1]
    else:
        try:
            resp = httpx.get(
                f"{ODDS_BASE_URL}/sports/{outright_key}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us,eu",
                    "markets": "outrights",
                    "oddsFormat": "american",
                },
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
            remaining = resp.headers.get("x-requests-remaining", "?")
            log.info("Odds API outrights: fetched %s (%d events, %s remaining)",
                     outright_key, len(events), remaining)
        except Exception as e:
            log.warning("Odds API outright error for %s: %s", outright_key, e)
            return None
        _odds_cache[cache_key] = (now, events)

    if not events:
        return None

    team_odds: dict[str, list[float]] = {}
    for event in events:
        for bm in event.get("bookmakers", [])[:8]:
            for market in bm.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    if name not in team_odds:
                        team_odds[name] = []
                    team_odds[name].append(price)

    if not team_odds:
        return None

    lines = ["Outright championship odds (bookmaker consensus):"]
    sorted_teams = sorted(
        team_odds.items(),
        key=lambda x: _american_odds_to_implied_prob(int(sum(x[1]) / len(x[1]))),
        reverse=True,
    )

    shown = set()
    for name, odds_list in sorted_teams:
        avg_odds = sum(odds_list) / len(odds_list)
        impl_prob = _american_odds_to_implied_prob(int(avg_odds))
        is_relevant = any(t.lower() in name.lower() or name.lower() in t.lower() for t in teams)
        if is_relevant or impl_prob >= 0.03 or len(shown) < 8:
            lines.append(f"  {name}: avg {avg_odds:+.0f} (implied {impl_prob:.1%}) [{len(odds_list)} books]")
            shown.add(name)
            if is_relevant:
                lines.append(f"    ^ THIS IS THE TEAM IN THE MARKET QUESTION")

    return "\n".join(lines) if len(lines) > 1 else None


def _format_odds(events: list, teams: list[str]) -> str | None:
    """Format odds events into a readable string, filtering for relevant teams."""
    if not events:
        return None

    relevant = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        if teams:
            if not any(t in home or t in away for t in teams):
                continue
        relevant.append(event)

    if not relevant:
        if teams:
            relevant = events[:3]
        else:
            relevant = events[:5]

    lines = []
    for event in relevant[:5]:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        if commence:
            try:
                dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                commence = dt.strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                pass

        lines.append(f"\n{away} @ {home} — {commence}")

        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            lines.append("  No odds available")
            continue

        h2h_probs = {}
        spread_lines_data = {}
        totals_data = {}

        for bm in bookmakers[:6]:
            for market in bm.get("markets", []):
                mkey = market.get("key")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if name not in h2h_probs:
                            h2h_probs[name] = []
                        h2h_probs[name].append(price)

                elif mkey == "spreads":
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        if name not in spread_lines_data:
                            spread_lines_data[name] = []
                        spread_lines_data[name].append(point)

                elif mkey == "totals":
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        if name not in totals_data:
                            totals_data[name] = []
                        totals_data[name].append(point)

        if h2h_probs:
            lines.append("  Moneyline (consensus from multiple books):")
            for name, odds_list in h2h_probs.items():
                avg_odds = sum(odds_list) / len(odds_list)
                impl_prob = _american_odds_to_implied_prob(int(avg_odds))
                lines.append(f"    {name}: avg {avg_odds:+.0f} (implied {impl_prob:.1%}) [{len(odds_list)} books]")

        if spread_lines_data:
            lines.append("  Spread:")
            for name, points in spread_lines_data.items():
                avg_spread = sum(points) / len(points)
                lines.append(f"    {name}: {avg_spread:+.1f}")

        if totals_data:
            over_points = totals_data.get("Over", [])
            if over_points:
                avg_total = sum(over_points) / len(over_points)
                lines.append(f"  Total (O/U): {avg_total:.1f}")

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Shared power rating computation
# ---------------------------------------------------------------------------

SOFTMAX_TEMP = 8.0


def _compute_power_scores(ratings: dict[str, dict]) -> None:
    """Add power_score and championship_prob to each team's rating dict in-place."""
    all_net = [r["net_rating"] for r in ratings.values()]
    if not all_net:
        return
    min_net = min(all_net)
    max_net = max(all_net)
    net_range = max_net - min_net if max_net != min_net else 1.0

    for r in ratings.values():
        norm_net = (r["net_rating"] - min_net) / net_range
        r["power_score"] = 0.4 * r["win_pct"] + 0.4 * norm_net + 0.2 * r["recent_form"]

    _apply_softmax_probs(ratings)


def _apply_softmax_probs(ratings: dict[str, dict], key: str = "championship_prob") -> dict[str, float]:
    """Convert power_score to probabilities via softmax. Returns the prob dict."""
    power_scores = {name: r["power_score"] for name, r in ratings.items()}
    if not power_scores:
        return {}
    max_score = max(power_scores.values())
    exp_scores = {name: math.exp(SOFTMAX_TEMP * (score - max_score)) for name, score in power_scores.items()}
    total_exp = sum(exp_scores.values())
    probs = {}
    for name in ratings:
        prob = exp_scores[name] / total_exp
        ratings[name][key] = prob
        probs[name] = prob
    return probs


def _get_cached_ratings(sport: str) -> dict[str, dict] | None:
    cached = _ratings_cache.get(sport)
    if cached and (time.time() - cached[0]) < RATINGS_TTL:
        return cached[1]
    return None


def _save_ratings_cache(sport: str, ratings: dict[str, dict]) -> None:
    _ratings_cache[sport] = (time.time(), ratings)


def _format_ratings_table(
    ratings: dict[str, dict],
    teams: list[str],
    question: str,
    sport_name: str,
    conferences: dict[str, list[str]] | None = None,
    diff_label: str = "NetRtg",
) -> str | None:
    """Generic formatter for power ratings across sports."""
    if not ratings:
        return None

    question_lower = question.lower()
    conference = None
    if conferences:
        for conf_name in conferences:
            if conf_name.lower() in question_lower:
                conference = conf_name
                break

    if conference:
        conf_teams = conferences.get(conference, [])
        filtered = {k: v for k, v in ratings.items() if k in conf_teams}
    elif teams:
        filtered = {k: v for k, v in ratings.items() if k in teams}
    else:
        filtered = ratings

    if not filtered:
        return None

    sorted_teams = sorted(filtered.items(), key=lambda x: -x[1]["power_score"])

    scope = f" ({conference})" if conference else ""
    lines = [f"{sport_name} Power Ratings{scope} — from season data:"]
    lines.append(f"{'Team':<28} {'W-L':<8} {'Win%':<6} {diff_label:<8} {'Form':<6} {'Power':<6} {'Champ%':<7}")

    if conference:
        scoped_ratings = {k: dict(v) for k, v in filtered.items()}
        _apply_softmax_probs(scoped_ratings)
        conf_probs = {name: r["championship_prob"] for name, r in scoped_ratings.items()}
    else:
        conf_probs = None

    for name, r in sorted_teams[:15]:
        prob = conf_probs[name] if conf_probs else r["championship_prob"]
        record = f"{r['wins']}-{r['losses']}"
        if "otl" in r:
            record += f"-{r['otl']}"
        lines.append(
            f"  {name:<26} {record:<8} "
            f"{r['win_pct']:.3f} {r['net_rating']:>+7.1f} "
            f"{r['recent_form']:.2f}  {r['power_score']:.3f} "
            f"{prob:.1%}"
        )

    if teams and len(teams) == 1:
        team = teams[0]
        if team in ratings:
            prob = conf_probs[team] if conf_probs and team in conf_probs else ratings[team]["championship_prob"]
            scope_label = f" {conference}" if conference else " championship"
            lines.append(f"\nModel estimate for {team}: {prob:.1%} chance to win{scope_label}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NBA Power Ratings — BallDontLie API (free)
# Blends playoff data (80%) with regular season (20%) as a baseline.
# ---------------------------------------------------------------------------

PLAYOFF_WEIGHT = 0.80
REGULAR_SEASON_WEIGHT = 0.20


def _fetch_nba_games(postseason: bool, headers: dict) -> dict[str, dict]:
    """Fetch NBA game data for a season type. Returns team stats dict."""
    team_stats: dict[str, dict] = {}
    for page in range(1, 4):
        try:
            resp = httpx.get(
                "https://api.balldontlie.io/v1/games",
                params={
                    "seasons[]": 2025,
                    "per_page": 100,
                    "cursor": (page - 1) * 100 if page > 1 else 0,
                    "postseason": "true" if postseason else "false",
                },
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                break
            games = resp.json().get("data", [])
            if not games:
                break

            for game in games:
                home_name = game.get("home_team", {}).get("full_name", "")
                away_name = game.get("visitor_team", {}).get("full_name", "")
                home_score = game.get("home_team_score", 0)
                away_score = game.get("visitor_team_score", 0)
                if not home_score and not away_score:
                    continue

                for name, pts_for, pts_against in [
                    (home_name, home_score, away_score),
                    (away_name, away_score, home_score),
                ]:
                    if name not in team_stats:
                        team_stats[name] = {"wins": 0, "losses": 0, "pts_for": 0, "pts_against": 0, "games": 0, "recent_scores": []}
                    ts = team_stats[name]
                    ts["games"] += 1
                    ts["pts_for"] += pts_for
                    ts["pts_against"] += pts_against
                    if pts_for > pts_against:
                        ts["wins"] += 1
                    else:
                        ts["losses"] += 1
                    ts["recent_scores"].append(pts_for - pts_against)
        except Exception as e:
            log.debug("BallDontLie error: %s", e)
            break
    return team_stats


def _fetch_nba_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("nba")
    if cached:
        return cached

    bdl_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    headers = {"Authorization": bdl_key} if bdl_key else {}

    playoff_stats = _fetch_nba_games(postseason=True, headers=headers)
    regular_stats = _fetch_nba_games(postseason=False, headers=headers)

    all_teams = set(playoff_stats.keys()) | set(regular_stats.keys())
    if not all_teams:
        return None

    ratings = {}
    for name in all_teams:
        ps = playoff_stats.get(name)
        rs = regular_stats.get(name)

        if ps and ps["games"] > 0:
            p_win_pct = ps["wins"] / ps["games"]
            p_net = (ps["pts_for"] - ps["pts_against"]) / ps["games"]
            p_recent = ps["recent_scores"][-10:]
            p_form = sum(1 for s in p_recent if s > 0) / len(p_recent) if p_recent else 0.5
        else:
            p_win_pct = p_net = p_form = None

        if rs and rs["games"] > 0:
            r_win_pct = rs["wins"] / rs["games"]
            r_net = (rs["pts_for"] - rs["pts_against"]) / rs["games"]
            r_recent = rs["recent_scores"][-10:]
            r_form = sum(1 for s in r_recent if s > 0) / len(r_recent) if r_recent else 0.5
        else:
            r_win_pct = r_net = r_form = None

        if p_win_pct is not None and r_win_pct is not None:
            win_pct = PLAYOFF_WEIGHT * p_win_pct + REGULAR_SEASON_WEIGHT * r_win_pct
            net_rating = PLAYOFF_WEIGHT * p_net + REGULAR_SEASON_WEIGHT * r_net
            recent_form = PLAYOFF_WEIGHT * p_form + REGULAR_SEASON_WEIGHT * r_form
        elif p_win_pct is not None:
            win_pct = p_win_pct
            net_rating = p_net
            recent_form = p_form
        elif r_win_pct is not None:
            win_pct = r_win_pct * 0.5
            net_rating = r_net * 0.5
            recent_form = r_form * 0.5
        else:
            continue

        p_wins = ps["wins"] if ps else 0
        p_losses = ps["losses"] if ps else 0

        ratings[name] = {
            "wins": p_wins,
            "losses": p_losses,
            "win_pct": win_pct,
            "net_rating": net_rating,
            "recent_form": recent_form,
            "games": (ps["games"] if ps else 0) + (rs["games"] if rs else 0),
            "playoff_games": ps["games"] if ps else 0,
            "regular_season_record": f"{rs['wins']}-{rs['losses']}" if rs else "N/A",
        }

    if not ratings:
        return None

    _compute_power_scores(ratings)
    _save_ratings_cache("nba", ratings)
    return ratings


# ---------------------------------------------------------------------------
# NBA Playoff Bracket — detects active matchups from recent games
# ---------------------------------------------------------------------------

_bracket_cache: tuple[float, str] | None = None
BRACKET_CACHE_TTL = 6 * 3600


def _fetch_nba_playoff_bracket() -> str | None:
    """Detect active playoff matchups from recent postseason games."""
    global _bracket_cache
    now = time.time()
    if _bracket_cache and (now - _bracket_cache[0]) < BRACKET_CACHE_TTL:
        return _bracket_cache[1]

    bdl_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    headers = {"Authorization": bdl_key} if bdl_key else {}

    try:
        resp = httpx.get(
            "https://api.balldontlie.io/v1/games",
            params={
                "seasons[]": 2025,
                "per_page": 100,
                "postseason": "true",
            },
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        games = resp.json().get("data", [])
    except Exception as e:
        log.debug("BallDontLie bracket error: %s", e)
        return None

    if not games:
        return None

    matchups: dict[tuple, dict] = {}
    for game in games:
        home = game.get("home_team", {}).get("full_name", "")
        away = game.get("visitor_team", {}).get("full_name", "")
        home_score = game.get("home_team_score", 0)
        away_score = game.get("visitor_team_score", 0)
        if not home_score and not away_score:
            continue

        pair = tuple(sorted([home, away]))
        if pair not in matchups:
            matchups[pair] = {"teams": pair, "games": 0, "wins": {pair[0]: 0, pair[1]: 0}}
        m = matchups[pair]
        m["games"] += 1
        if home_score > away_score:
            m["wins"][home] = m["wins"].get(home, 0) + 1
        else:
            m["wins"][away] = m["wins"].get(away, 0) + 1

    if not matchups:
        return None

    active = [(pair, m) for pair, m in matchups.items() if m["games"] < 7 and max(m["wins"].values()) < 4]
    completed = [(pair, m) for pair, m in matchups.items() if max(m["wins"].values()) >= 4]

    lines = ["NBA Playoff Bracket (2025):"]

    if active:
        lines.append("  Active series:")
        for pair, m in sorted(active, key=lambda x: -x[1]["games"]):
            t1, t2 = pair
            lines.append(f"    {t1} vs {t2} — Series: {m['wins'][t1]}-{m['wins'][t2]}")

    if completed:
        lines.append("  Completed series:")
        for pair, m in completed:
            t1, t2 = pair
            winner = t1 if m["wins"][t1] > m["wins"][t2] else t2
            loser = t2 if winner == t1 else t1
            lines.append(f"    {winner} def. {loser} ({m['wins'][winner]}-{m['wins'][loser]})")

    result = "\n".join(lines)
    _bracket_cache = (now, result)
    return result


# ---------------------------------------------------------------------------
# NBA Injury Reports — via Tavily search
# ---------------------------------------------------------------------------

_injury_cache: tuple[float, str] | None = None
INJURY_CACHE_TTL = 4 * 3600


def _fetch_nba_injuries() -> str | None:
    """Fetch current NBA injury reports via Tavily search."""
    global _injury_cache
    now = time.time()
    if _injury_cache and (now - _injury_cache[0]) < INJURY_CACHE_TTL:
        return _injury_cache[1]

    if not _tavily_client:
        return None

    try:
        response = _tavily_client.search(
            query="NBA playoff injury report today 2025",
            search_depth="basic",
            max_results=5,
            include_answer=False,
        )
        results = response.get("results", [])
        if not results:
            return None

        lines = ["NBA Injury Report (recent):"]
        for r in results:
            title = r.get("title", "")
            snippet = r.get("content", "")[:300]
            lines.append(f"  - {title}")
            if snippet:
                lines.append(f"    {snippet}")

        result = "\n".join(lines)
        _injury_cache = (now, result)
        return result
    except Exception as e:
        log.debug("Tavily injury fetch error: %s", e)
        return None


# ---------------------------------------------------------------------------
# NHL Power Ratings — official NHL API (free, no key)
# ---------------------------------------------------------------------------

NHL_CONFERENCES = {
    "Eastern": [],
    "Western": [],
}

NHL_TEAM_NAME_MAP = {
    "utah hockey club": "Utah Hockey Club",
}


def _fetch_nhl_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("nhl")
    if cached:
        return cached

    try:
        client = httpx.Client(follow_redirects=True, timeout=15)
        resp = client.get("https://api-web.nhle.com/v1/standings/now")
        resp.raise_for_status()
        standings = resp.json().get("standings", [])
        client.close()
    except Exception as e:
        log.warning("NHL API error: %s", e)
        return None

    if not standings:
        return None

    NHL_CONFERENCES["Eastern"] = []
    NHL_CONFERENCES["Western"] = []

    ratings = {}
    for team in standings:
        full_name = team.get("teamName", {}).get("default", "")

        wins = team.get("wins", 0)
        losses = team.get("losses", 0)
        otl = team.get("otLosses", 0)
        gp = team.get("gamesPlayed", 0)
        gf = team.get("goalFor", 0)
        ga = team.get("goalAgainst", 0)
        pts = team.get("points", 0)
        streak = team.get("streakCode", "")
        l10_wins = team.get("l10Wins", 0)
        l10_losses = team.get("l10Losses", 0)
        l10_otl = team.get("l10OtLosses", 0)
        conf = team.get("conferenceName", "")

        if conf in NHL_CONFERENCES:
            NHL_CONFERENCES[conf].append(full_name)

        if gp == 0:
            continue

        net_rating = (gf - ga) / gp
        win_pct = (wins + 0.5 * otl) / gp  # OT losses are half-wins in NHL points
        l10_total = l10_wins + l10_losses + l10_otl
        recent_form = l10_wins / l10_total if l10_total > 0 else 0.5

        ratings[full_name] = {
            "wins": wins,
            "losses": losses,
            "otl": otl,
            "win_pct": win_pct,
            "net_rating": net_rating,
            "recent_form": recent_form,
            "games": gp,
            "points": pts,
        }

    if not ratings:
        return None

    _compute_power_scores(ratings)
    _save_ratings_cache("nhl", ratings)
    log.info("NHL power ratings: %d teams computed", len(ratings))
    return ratings


# ---------------------------------------------------------------------------
# MLB Power Ratings — official MLB Stats API (free, no key)
# ---------------------------------------------------------------------------

MLB_LEAGUES = {
    "American League": [],
    "National League": [],
}


def _fetch_mlb_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("mlb")
    if cached:
        return cached

    try:
        client = httpx.Client(follow_redirects=True, timeout=15)
        resp = client.get("https://statsapi.mlb.com/api/v1/standings", params={
            "leagueId": "103,104",
            "season": 2026,
            "standingsTypes": "regularSeason",
            "hydrate": "team",
        })
        resp.raise_for_status()
        data = resp.json()
        client.close()
    except Exception as e:
        log.warning("MLB API error: %s", e)
        return None

    records = data.get("records", [])
    if not records:
        return None

    # AL division IDs: 200 (West), 201 (East), 202 (Central)
    # NL division IDs: 203 (West), 204 (East), 205 (Central)
    AL_DIVISIONS = {200, 201, 202}
    NL_DIVISIONS = {203, 204, 205}

    MLB_LEAGUES["American League"] = []
    MLB_LEAGUES["National League"] = []

    ratings = {}
    for division in records:
        div_id = division.get("division", {}).get("id", 0)
        for team_rec in division.get("teamRecords", []):
            name = team_rec.get("team", {}).get("name", "")
            wins = team_rec.get("wins", 0)
            losses = team_rec.get("losses", 0)
            gp = team_rec.get("gamesPlayed", 0)
            rd = team_rec.get("runDifferential", 0)
            rs = team_rec.get("runsScored", 0)
            ra = team_rec.get("runsAllowed", 0)

            if div_id in AL_DIVISIONS:
                MLB_LEAGUES["American League"].append(name)
            elif div_id in NL_DIVISIONS:
                MLB_LEAGUES["National League"].append(name)

            if gp == 0:
                continue

            net_rating = rd / gp
            win_pct = wins / gp

            # Pythagorean win expectation (Bill James formula) as a more stable estimate
            if rs > 0 and ra > 0:
                pyth_pct = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)
            else:
                pyth_pct = win_pct

            # Recent form: blend actual win% with pythagorean (pythagorean is more predictive)
            recent_form = pyth_pct

            ratings[name] = {
                "wins": wins,
                "losses": losses,
                "win_pct": win_pct,
                "net_rating": net_rating,
                "recent_form": recent_form,
                "games": gp,
                "runs_scored": rs,
                "runs_allowed": ra,
                "pyth_pct": pyth_pct,
            }

    if not ratings:
        return None

    _compute_power_scores(ratings)
    _save_ratings_cache("mlb", ratings)
    log.info("MLB power ratings: %d teams computed", len(ratings))
    return ratings


# ---------------------------------------------------------------------------
# NFL Power Ratings — ESPN public API (free, no key)
# ---------------------------------------------------------------------------

NFL_CONFERENCES = {
    "AFC": [],
    "NFC": [],
}

ESPN_CONF_MAP = {
    "American Football Conference": "AFC",
    "National Football Conference": "NFC",
}


def _fetch_nfl_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("nfl")
    if cached:
        return cached

    try:
        client = httpx.Client(follow_redirects=True, timeout=15)
        resp = client.get(
            "https://site.api.espn.com/apis/v2/sports/football/nfl/standings",
            params={"season": 2025},
        )
        resp.raise_for_status()
        data = resp.json()
        client.close()
    except Exception as e:
        log.warning("ESPN NFL API error: %s", e)
        return None

    conferences = data.get("children", [])
    if not conferences:
        return None

    NFL_CONFERENCES["AFC"] = []
    NFL_CONFERENCES["NFC"] = []

    ratings = {}
    for conf in conferences:
        conf_name = ESPN_CONF_MAP.get(conf.get("name", ""), "")
        entries = conf.get("standings", {}).get("entries", [])

        for entry in entries:
            name = entry.get("team", {}).get("displayName", "")
            if not name:
                continue

            if conf_name in NFL_CONFERENCES:
                NFL_CONFERENCES[conf_name].append(name)

            stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", []) if "name" in s}
            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            pf = stats.get("pointsFor", 0)
            pa = stats.get("pointsAgainst", 0)
            gp = wins + losses + int(stats.get("ties", 0))

            if gp == 0:
                continue

            net_rating = (pf - pa) / gp
            win_pct = wins / gp

            if pf > 0 and pa > 0:
                pyth_pct = pf ** 2.37 / (pf ** 2.37 + pa ** 2.37)
            else:
                pyth_pct = win_pct

            recent_form = pyth_pct

            ratings[name] = {
                "wins": wins,
                "losses": losses,
                "win_pct": win_pct,
                "net_rating": net_rating,
                "recent_form": recent_form,
                "games": gp,
                "points_for": pf,
                "points_against": pa,
                "pyth_pct": pyth_pct,
            }

    if not ratings:
        return None

    _compute_power_scores(ratings)
    _save_ratings_cache("nfl", ratings)
    log.info("NFL power ratings: %d teams computed", len(ratings))
    return ratings


# ---------------------------------------------------------------------------
# World Cup Power Ratings — hardcoded from FIFA rankings, history, squad quality
# ---------------------------------------------------------------------------

def _fetch_world_cup_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("world_cup")
    if cached:
        return cached

    total_teams = len(FIFA_RANKINGS)
    ratings = {}

    for country, rank in FIFA_RANKINGS.items():
        history = WORLD_CUP_HISTORY.get(country, {})
        squad = SQUAD_QUALITY.get(country, 0.5)

        # Normalize FIFA ranking to 0-1 (rank 1 = 1.0, rank 31 = 0.0)
        rank_score = 1.0 - (rank - 1) / (total_teams - 1)

        # Historical performance score
        titles = history.get("titles", 0)
        finals = history.get("finals", 0)
        semis = history.get("semis", 0)
        quarters = history.get("quarters", 0)
        appearances = history.get("appearances", 0)
        # Weighted: titles matter most, then depth of runs
        history_raw = titles * 5.0 + finals * 2.0 + semis * 1.0 + quarters * 0.5
        # Normalize — Brazil has max history_raw = 5*5 + 7*2 + 12*1 + 15*0.5 = 58.5
        history_score = min(history_raw / 40.0, 1.0)

        # Composite: 25% FIFA rank, 15% history, 60% squad quality (current form)
        win_pct = 0.25 * rank_score + 0.15 * history_score + 0.60 * squad
        net_rating = (rank_score - 0.5) * 10.0  # synthetic scale for display
        recent_form = squad  # squad quality is our best proxy for current form

        ratings[country] = {
            "wins": titles,
            "losses": appearances - titles,
            "win_pct": win_pct,
            "net_rating": net_rating,
            "recent_form": recent_form,
            "games": appearances,
            "fifa_rank": rank,
            "squad_quality": squad,
            "history_score": history_score,
        }

    _compute_power_scores(ratings)
    _save_ratings_cache("world_cup", ratings)
    log.info("World Cup power ratings: %d teams computed", len(ratings))
    return ratings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_sports_market(question: str) -> bool:
    """Check if a market question is sports-related."""
    return _detect_sport(question) is not None


def fetch_sports_context(question: str) -> str | None:
    """Fetch sports data relevant to a market question.

    Returns formatted odds/stats context, or None if not a sports market.
    """
    sport_info = _detect_sport(question)
    if not sport_info:
        return None

    teams = _extract_teams(question)
    sport_key = sport_info["odds_key"]
    label = sport_info["label"]

    sections = [f"Sports data for {label}:"]
    is_championship = _is_championship_question(question)

    if is_championship:
        outright_data = _fetch_outright_odds(sport_key, teams)
        if outright_data:
            sections.append(f"\n{outright_data}")

    events = _fetch_odds_raw(sport_key)
    odds_data = _format_odds(events, teams)
    if not odds_data and "odds_keys_fallback" in sport_info:
        for fallback_key in sport_info["odds_keys_fallback"]:
            events = _fetch_odds_raw(fallback_key)
            odds_data = _format_odds(events, teams)
            if odds_data:
                break
    if odds_data:
        prefix = "Upcoming match odds (NOT outright winner odds)" if is_championship else "Bookmaker consensus odds"
        sections.append(f"\n{prefix}:{odds_data}")

    if sport_key == "basketball_nba":
        ratings = _fetch_nba_power_ratings()
        table = _format_ratings_table(ratings, teams, question, "NBA", NBA_CONFERENCES)
        if table:
            sections.append(f"\n{table}")

        bracket = _fetch_nba_playoff_bracket()
        if bracket:
            sections.append(f"\n{bracket}")

        injuries = _fetch_nba_injuries()
        if injuries:
            sections.append(f"\n{injuries}")

    elif sport_key in ("americanfootball_nfl", "americanfootball_ncaaf"):
        if sport_key == "americanfootball_nfl":
            ratings = _fetch_nfl_power_ratings()
            table = _format_ratings_table(ratings, teams, question, "NFL", NFL_CONFERENCES, diff_label="PD/GP")
            if table:
                sections.append(f"\n{table}")

    elif sport_key == "icehockey_nhl":
        ratings = _fetch_nhl_power_ratings()
        table = _format_ratings_table(ratings, teams, question, "NHL", NHL_CONFERENCES, diff_label="GD/GP")
        if table:
            sections.append(f"\n{table}")

    elif sport_key == "baseball_mlb":
        ratings = _fetch_mlb_power_ratings()
        table = _format_ratings_table(ratings, teams, question, "MLB", MLB_LEAGUES, diff_label="RD/GP")
        if table:
            sections.append(f"\n{table}")

    elif sport_key == "soccer_fifa_world_cup":
        ratings = _fetch_world_cup_power_ratings()
        table = _format_ratings_table(ratings, teams, question, "World Cup", WORLD_CUP_CONFEDERATIONS)
        if table:
            sections.append(f"\n{table}")

    if len(sections) == 1:
        return None

    return "\n".join(sections)
