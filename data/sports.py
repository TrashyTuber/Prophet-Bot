"""Fetch sports data from The Odds API and free stats APIs.

Provides consensus bookmaker odds, recent team stats, and power ratings
for sports prediction markets.

Requires THE_ODDS_API_KEY in environment (free at https://the-odds-api.com).
"""

import logging
import os
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_CACHE_TTL = 4 * 3600  # 4 hours — championship odds barely move

_odds_cache: dict[str, tuple[float, list]] = {}  # sport_key -> (timestamp, events)
_nba_ratings_cache: tuple[float, dict] | None = None
NBA_RATINGS_TTL = 6 * 3600

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
# NBA Power Ratings — built from BallDontLie season stats (free, no key needed)
# ---------------------------------------------------------------------------

def _fetch_nba_power_ratings() -> dict[str, dict] | None:
    """Build NBA power ratings from current season standings and recent games.

    Returns a dict of team_name -> {
        wins, losses, win_pct, net_rating, recent_form, power_score, championship_prob
    }
    """
    global _nba_ratings_cache
    now = time.time()
    if _nba_ratings_cache and (now - _nba_ratings_cache[0]) < NBA_RATINGS_TTL:
        return _nba_ratings_cache[1]

    bdl_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    headers = {"Authorization": bdl_key} if bdl_key else {}

    # Fetch current season standings
    try:
        resp = httpx.get(
            "https://api.balldontlie.io/v1/season_averages",
            params={"season": 2025},
            headers=headers,
            timeout=15,
        )
    except Exception as e:
        log.debug("BallDontLie season_averages failed: %s", e)

    # Fetch recent games to compute team records and net ratings
    team_stats: dict[str, dict] = {}
    for page in range(1, 4):
        try:
            resp = httpx.get(
                "https://api.balldontlie.io/v1/games",
                params={
                    "seasons[]": 2025,
                    "per_page": 100,
                    "cursor": (page - 1) * 100 if page > 1 else 0,
                    "postseason": "true",
                },
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                log.debug("BallDontLie games page %d: status %d", page, resp.status_code)
                break
            data = resp.json()
            games = data.get("data", [])
            if not games:
                break

            for game in games:
                home_name = game.get("home_team", {}).get("full_name", "")
                away_name = game.get("visitor_team", {}).get("full_name", "")
                home_score = game.get("home_team_score", 0)
                away_score = game.get("visitor_team_score", 0)

                if not home_score and not away_score:
                    continue

                for name, pts_for, pts_against, is_home in [
                    (home_name, home_score, away_score, True),
                    (away_name, away_score, home_score, False),
                ]:
                    if name not in team_stats:
                        team_stats[name] = {
                            "wins": 0, "losses": 0,
                            "pts_for": 0, "pts_against": 0,
                            "games": 0, "recent_scores": [],
                        }
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
            log.debug("BallDontLie games fetch error: %s", e)
            break

    if not team_stats:
        return None

    ratings = {}
    all_net = []
    for name, ts in team_stats.items():
        if ts["games"] == 0:
            continue
        net_rating = (ts["pts_for"] - ts["pts_against"]) / ts["games"]
        win_pct = ts["wins"] / ts["games"] if ts["games"] > 0 else 0.5
        recent = ts["recent_scores"][-10:]
        recent_form = sum(1 for s in recent if s > 0) / len(recent) if recent else 0.5

        ratings[name] = {
            "wins": ts["wins"],
            "losses": ts["losses"],
            "win_pct": win_pct,
            "net_rating": net_rating,
            "recent_form": recent_form,
            "games": ts["games"],
        }
        all_net.append(net_rating)

    if not ratings:
        return None

    # Compute power scores and championship probabilities
    # Power score = weighted combo of win%, net rating (normalized), and recent form
    min_net = min(all_net)
    max_net = max(all_net)
    net_range = max_net - min_net if max_net != min_net else 1.0

    for name, r in ratings.items():
        norm_net = (r["net_rating"] - min_net) / net_range
        r["power_score"] = 0.4 * r["win_pct"] + 0.4 * norm_net + 0.2 * r["recent_form"]

    # Convert power scores to championship probabilities via softmax
    import math
    TEMP = 8.0  # higher = more spread out, lower = winner-take-all
    power_scores = {name: r["power_score"] for name, r in ratings.items()}
    max_score = max(power_scores.values())
    exp_scores = {name: math.exp(TEMP * (score - max_score)) for name, score in power_scores.items()}
    total_exp = sum(exp_scores.values())

    for name in ratings:
        ratings[name]["championship_prob"] = exp_scores[name] / total_exp

    _nba_ratings_cache = (now, ratings)
    return ratings


def _format_nba_ratings(teams: list[str], question: str) -> str | None:
    """Format NBA power ratings for specific teams or as a league overview."""
    ratings = _fetch_nba_power_ratings()
    if not ratings:
        return None

    question_lower = question.lower()
    is_conference = "east" in question_lower or "west" in question_lower
    conference = "East" if "east" in question_lower else "West" if "west" in question_lower else None

    if conference:
        conf_teams = NBA_CONFERENCES.get(conference, [])
        filtered = {k: v for k, v in ratings.items() if k in conf_teams}
    elif teams:
        filtered = {k: v for k, v in ratings.items() if k in teams}
    else:
        filtered = ratings

    if not filtered:
        return None

    sorted_teams = sorted(filtered.items(), key=lambda x: -x[1]["power_score"])

    scope = f" ({conference}ern Conference)" if conference else ""
    lines = [f"NBA Power Ratings{scope} — from season game data:"]
    lines.append(f"{'Team':<28} {'W-L':<8} {'Win%':<6} {'NetRtg':<8} {'Form':<6} {'Power':<6} {'Champ%':<7}")

    # Recompute conference championship probs if needed
    if conference:
        import math
        TEMP = 8.0
        power_scores = {name: r["power_score"] for name, r in filtered.items()}
        max_score = max(power_scores.values()) if power_scores else 0
        exp_scores = {name: math.exp(TEMP * (score - max_score)) for name, score in power_scores.items()}
        total_exp = sum(exp_scores.values()) if exp_scores else 1
        conf_probs = {name: exp_scores[name] / total_exp for name in filtered}
    else:
        conf_probs = None

    for name, r in sorted_teams[:15]:
        prob = conf_probs[name] if conf_probs else r["championship_prob"]
        lines.append(
            f"  {name:<26} {r['wins']}-{r['losses']:<5} "
            f"{r['win_pct']:.3f} {r['net_rating']:>+7.1f} "
            f"{r['recent_form']:.2f}  {r['power_score']:.3f} "
            f"{prob:.1%}"
        )

    if teams and len(teams) == 1:
        team = teams[0]
        if team in ratings:
            r = ratings[team]
            prob = conf_probs[team] if conf_probs and team in conf_probs else r["championship_prob"]
            lines.append(f"\nModel estimate for {team}: {prob:.1%} chance to win" +
                         (f" {conference}ern Conference" if conference else " championship"))

    return "\n".join(lines)


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

    events = _fetch_odds_raw(sport_key)
    odds_data = _format_odds(events, teams)
    if odds_data:
        sections.append(f"\nBookmaker consensus odds:{odds_data}")

    if sport_key == "basketball_nba":
        nba_ratings = _format_nba_ratings(teams, question)
        if nba_ratings:
            sections.append(f"\n{nba_ratings}")

    if len(sections) == 1:
        return None

    return "\n".join(sections)
