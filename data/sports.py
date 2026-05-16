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

log = logging.getLogger(__name__)

ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_CACHE_TTL = 4 * 3600  # 4 hours — championship odds barely move

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

ALL_TEAMS = {**NBA_TEAMS, **MLB_TEAMS, **NFL_TEAMS, **NHL_TEAMS}


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
# ---------------------------------------------------------------------------

def _fetch_nba_power_ratings() -> dict[str, dict] | None:
    cached = _get_cached_ratings("nba")
    if cached:
        return cached

    bdl_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    headers = {"Authorization": bdl_key} if bdl_key else {}

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

    if not team_stats:
        return None

    ratings = {}
    for name, ts in team_stats.items():
        if ts["games"] == 0:
            continue
        recent = ts["recent_scores"][-10:]
        ratings[name] = {
            "wins": ts["wins"],
            "losses": ts["losses"],
            "win_pct": ts["wins"] / ts["games"],
            "net_rating": (ts["pts_for"] - ts["pts_against"]) / ts["games"],
            "recent_form": sum(1 for s in recent if s > 0) / len(recent) if recent else 0.5,
            "games": ts["games"],
        }

    _compute_power_scores(ratings)
    _save_ratings_cache("nba", ratings)
    return ratings


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

    events = _fetch_odds_raw(sport_key)
    odds_data = _format_odds(events, teams)
    if odds_data:
        sections.append(f"\nBookmaker consensus odds:{odds_data}")

    if sport_key == "basketball_nba":
        ratings = _fetch_nba_power_ratings()
        table = _format_ratings_table(ratings, teams, question, "NBA", NBA_CONFERENCES)
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

    if len(sections) == 1:
        return None

    return "\n".join(sections)
