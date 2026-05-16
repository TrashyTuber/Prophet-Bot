"""Fetch sports data from The Odds API and free stats APIs.

Provides consensus bookmaker odds, recent team stats, and head-to-head records
for sports prediction markets.

Requires THE_ODDS_API_KEY in environment (free at https://the-odds-api.com).
"""

import logging
import os
import re
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

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
        "label": "Tennis",
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

ALL_TEAMS = {**NBA_TEAMS, **MLB_TEAMS, **NFL_TEAMS}


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
    return None


def _extract_teams(question: str) -> list[str]:
    question_lower = question.lower()
    found = []
    for nickname, full_name in ALL_TEAMS.items():
        if nickname in question_lower and full_name not in found:
            found.append(full_name)
    return found


def _american_odds_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _decimal_odds_to_implied_prob(odds: float) -> float:
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def _fetch_odds(sport_key: str, teams: list[str]) -> str | None:
    if not ODDS_API_KEY:
        log.warning("THE_ODDS_API_KEY not set — skipping odds fetch")
        return None

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
    except Exception as e:
        log.warning("Odds API error for %s: %s", sport_key, e)
        return None

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
            bm_name = bm.get("title", "Unknown")
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


def _fetch_nba_stats(teams: list[str]) -> str | None:
    """Fetch recent NBA game results from BallDontLie API (free, no key)."""
    if not teams:
        return None

    try:
        resp = httpx.get(
            "https://api.balldontlie.io/v1/games",
            params={
                "per_page": 10,
                "dates[]": [],
            },
            headers={"Authorization": os.environ.get("BALLDONTLIE_API_KEY", "")},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        games = resp.json().get("data", [])
    except Exception as e:
        log.debug("BallDontLie API error: %s", e)
        return None

    if not games:
        return None

    relevant = []
    for game in games:
        home = game.get("home_team", {}).get("full_name", "")
        away = game.get("visitor_team", {}).get("full_name", "")
        if any(t in home or t in away for t in teams):
            relevant.append(game)

    if not relevant:
        return None

    lines = ["Recent games:"]
    for game in relevant[:5]:
        home = game.get("home_team", {}).get("full_name", "")
        away = game.get("visitor_team", {}).get("full_name", "")
        home_score = game.get("home_team_score", 0)
        away_score = game.get("visitor_team_score", 0)
        date = game.get("date", "")[:10]
        status = game.get("status", "")
        if home_score or away_score:
            lines.append(f"  {date}: {away} {away_score} @ {home} {home_score} ({status})")

    return "\n".join(lines) if len(lines) > 1 else None


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

    odds_data = _fetch_odds(sport_key, teams)
    if odds_data:
        sections.append(f"\nBookmaker consensus odds:{odds_data}")

    if sport_info["odds_key"] == "basketball_nba" and teams:
        stats = _fetch_nba_stats(teams)
        if stats:
            sections.append(f"\n{stats}")

    if len(sections) == 1:
        return None

    return "\n".join(sections)
