import json
import logging
import os
import re
from datetime import datetime, timezone

import requests
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger(__name__)

EDGE_THRESHOLD_DEFAULT = 0.10
EDGE_THRESHOLD_WITH_DATA = 0.05
EDGE_THRESHOLD_WEATHER_WITH_DATA = 0.03
OPENROUTER_MODEL = "google/gemini-2.5-flash"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

_http_session = requests.Session()
_retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_http_session.mount("https://", HTTPAdapter(max_retries=_retry))

FRED_API_KEY = os.environ.get("FRED_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
NEWS_DAYS_THRESHOLD = 3.0
CACHE_PRICE_TOLERANCE = 0.02

_estimate_cache = {}

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

ECON_KEYWORDS = {
    "cpi": ["CPIAUCSL"],
    "inflation": ["CPIAUCSL", "PCEPILFE"],
    "pce": ["PCEPILFE"],
    "unemployment": ["UNRATE"],
    "jobs": ["PAYEMS", "UNRATE"],
    "nonfarm": ["PAYEMS"],
    "payroll": ["PAYEMS"],
    "gdp": ["GDP", "GDPC1"],
    "interest rate": ["FEDFUNDS", "DFF"],
    "fed funds": ["FEDFUNDS", "DFF"],
    "federal reserve": ["FEDFUNDS"],
    "retail sales": ["RSAFS"],
    "housing": ["HOUST"],
    "consumer confidence": ["UMCSENT"],
    "treasury": ["DGS10", "DGS2"],
    "yield": ["DGS10", "DGS2"],
}

WEATHER_KEYWORDS = [
    "temperature", "weather", "rain", "snow", "precipitation",
    "heat", "cold", "storm", "hurricane", "tornado", "wind",
    "celsius", "fahrenheit", "degrees",
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _fetch_news(question, days_left):
    if not tavily_client or days_left > NEWS_DAYS_THRESHOLD:
        return None

    try:
        response = tavily_client.search(
            query=question,
            search_depth="basic",
            max_results=5,
            include_answer=False,
        )
        results = response.get("results", [])
        if not results:
            return None

        lines = ["Recent news headlines:"]
        for r in results:
            title = r.get("title", "")
            snippet = r.get("content", "")[:200]
            lines.append(f"  - {title}")
            if snippet:
                lines.append(f"    {snippet}")
        return "\n".join(lines)
    except Exception:
        log.exception("Tavily search failed")
        return None


def _classify_market(market):
    question_lower = market.question.lower()
    topic = (market.topic or "").lower()
    family = (market.family or "").lower()

    for keyword in WEATHER_KEYWORDS:
        if keyword in question_lower or keyword in topic:
            return "weather"

    for keyword in ECON_KEYWORDS:
        if keyword in question_lower or keyword in topic:
            return "economics"

    return "general"


def _fetch_fred_data(series_ids):
    if not FRED_API_KEY:
        return None

    lines = []
    for sid in series_ids[:3]:
        try:
            resp = _http_session.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": sid,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 6,
                },
                timeout=10,
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if obs:
                values = [
                    f"  {o['date']}: {o['value']}"
                    for o in reversed(obs)
                    if o["value"] != "."
                ]
                lines.append(f"{sid} (last {len(values)} observations):")
                lines.extend(values)
        except Exception:
            log.exception("FRED fetch failed for %s", sid)

    return "\n".join(lines) if lines else None


def _match_fred_series(question):
    question_lower = question.lower()
    matched = []
    for keyword, series in ECON_KEYWORDS.items():
        if keyword in question_lower:
            for s in series:
                if s not in matched:
                    matched.append(s)
    return matched


def _fetch_weather_data(question):
    lat, lon, city = _extract_location(question)
    if lat is None:
        return None

    try:
        resp = _http_session.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("daily", {})
        dates = data.get("time", [])
        highs = data.get("temperature_2m_max", [])
        lows = data.get("temperature_2m_min", [])
        precip = data.get("precipitation_sum", [])

        lines = [f"7-day forecast for {city} ({lat}, {lon}):"]
        for i, d in enumerate(dates):
            lines.append(
                f"  {d}: High {highs[i]}°C / Low {lows[i]}°C, Precip {precip[i]}mm"
            )
        return "\n".join(lines)
    except Exception:
        log.exception("Open-Meteo fetch failed")
        return None


CITY_COORDS = {
    "new york": (40.71, -74.01, "New York"),
    "nyc": (40.71, -74.01, "New York"),
    "los angeles": (34.05, -118.24, "Los Angeles"),
    "chicago": (41.88, -87.63, "Chicago"),
    "houston": (29.76, -95.37, "Houston"),
    "phoenix": (33.45, -112.07, "Phoenix"),
    "philadelphia": (39.95, -75.17, "Philadelphia"),
    "san antonio": (29.42, -98.49, "San Antonio"),
    "san diego": (32.72, -117.16, "San Diego"),
    "dallas": (32.78, -96.80, "Dallas"),
    "austin": (30.27, -97.74, "Austin"),
    "miami": (25.76, -80.19, "Miami"),
    "seattle": (47.61, -122.33, "Seattle"),
    "denver": (39.74, -104.99, "Denver"),
    "boston": (42.36, -71.06, "Boston"),
    "washington": (38.91, -77.04, "Washington DC"),
    "dc": (38.91, -77.04, "Washington DC"),
    "atlanta": (33.75, -84.39, "Atlanta"),
    "san francisco": (37.77, -122.42, "San Francisco"),
    "sf": (37.77, -122.42, "San Francisco"),
    "london": (51.51, -0.13, "London"),
    "paris": (48.86, 2.35, "Paris"),
    "tokyo": (35.68, 139.69, "Tokyo"),
}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_geocode_cache = {}


def _geocode_from_question(question):
    words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    for candidate in sorted(words, key=len, reverse=True):
        candidate_lower = candidate.lower()
        if candidate_lower in ("will", "the", "what", "how", "does", "may", "june",
                               "july", "august", "january", "february", "march",
                               "april", "september", "october", "november", "december"):
            continue
        if candidate_lower in _geocode_cache:
            return _geocode_cache[candidate_lower]
        try:
            resp = _http_session.get(
                GEOCODE_URL,
                params={"name": candidate, "count": 1, "language": "en"},
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                r = results[0]
                val = (r["latitude"], r["longitude"], r["name"])
                _geocode_cache[candidate_lower] = val
                return val
        except Exception:
            log.debug("Geocode failed for %s", candidate)
        _geocode_cache[candidate_lower] = (None, None, None)
    return None, None, None


def _extract_location(question):
    question_lower = question.lower()
    for city, (lat, lon, name) in CITY_COORDS.items():
        if city in question_lower:
            return lat, lon, name
    return _geocode_from_question(question)


SYSTEM_PROMPT = """\
You are an expert forecaster for prediction markets. Your job is to estimate \
the true probability that a market resolves YES, given the market question, \
current pricing, and any external data provided.

Guidelines:
- Base rates matter. Start with the historical base rate, then adjust for \
specifics in the question.
- For threshold questions ("Will X exceed Y?"), consider the current value, \
recent trend, and typical volatility. Don't anchor too heavily on the current \
market price.
- Distinguish between your genuine uncertainty and the market's implied \
probability. The market can be wrong.
- If external data is provided (economic indicators, weather forecasts), use \
it as primary evidence rather than relying on priors.
- Be well-calibrated: when you say 70%, events should happen ~70% of the time. \
Avoid the extremes (< 0.05 or > 0.95) unless the evidence is overwhelming.
- Account for the time remaining until resolution. More time = more uncertainty.
- If recent news headlines are provided, check for breaking developments that \
could shift the outcome. News near resolution is especially high-signal.\
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Market: Will US CPI year-over-year exceed 3.2% for May 2026?\n"
            "Current YES price (implied probability): 0.45\n"
            "Resolution: 2026-06-12\n\n"
            "External data:\n"
            "CPIAUCSL (last 4 observations):\n"
            "  2026-01: 315.2\n"
            "  2026-02: 315.8\n"
            "  2026-03: 316.5\n"
            "  2026-04: 317.1\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"probability": <float 0-1>, "reasoning": "brief explanation"}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"probability": 0.52, "reasoning": "Recent CPI trend shows '
            "monthly increases of ~0.2%, putting YoY around 3.1-3.3%. "
            "It's close to the 3.2% threshold with slight upward momentum, "
            'so marginally above coin-flip."}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Market: Will the high temperature in Chicago exceed 90°F on July 4, 2026?\n"
            "Current YES price (implied probability): 0.35\n"
            "Resolution: 2026-07-05\n\n"
            "External data:\n"
            "7-day forecast for Chicago (41.88, -87.63):\n"
            "  2026-07-01: High 31°C / Low 21°C, Precip 0mm\n"
            "  2026-07-02: High 33°C / Low 22°C, Precip 0mm\n"
            "  2026-07-03: High 34°C / Low 23°C, Precip 0mm\n"
            "  2026-07-04: High 35°C / Low 24°C, Precip 0mm\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"probability": <float 0-1>, "reasoning": "brief explanation"}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"probability": 0.72, "reasoning": "The 7-day forecast shows '
            "35°C (95°F) for July 4th, well above the 90°F threshold. "
            "Weather forecasts 3 days out are fairly reliable. Market at 0.35 "
            'significantly underprices this — strong YES edge."}'
        ),
    },
]


def _build_user_message(market, implied_prob, external_data):
    parts = [
        f"Market: {market.question}",
        f"Current YES price (implied probability): {implied_prob:.2f}",
    ]

    if market.description:
        parts.append(f"Description: {market.description}")

    parts.append(f"Resolution: {market.resolution_time.strftime('%Y-%m-%d')}")

    now = datetime.now(timezone.utc)
    days_left = (market.resolution_time - now).total_seconds() / 86400
    parts.append(f"Days until resolution: {days_left:.1f}")

    if external_data:
        parts.append(f"\nExternal data:\n{external_data}")

    parts.append(
        "\nRespond with ONLY a JSON object, no other text:\n"
        '{"probability": <float between 0 and 1>, "reasoning": "brief explanation"}'
    )

    return "\n".join(parts)


def analyze_market(market):
    yes_ask = float(market.quote.best_ask)
    no_ask = 1.0 - float(market.quote.best_bid)
    implied_prob = yes_ask

    market_type = _classify_market(market)

    if market_type == "general":
        log.info("[STRATEGY] %s type=general — skipping (no data advantage)", market.market_id)
        return None

    external_data = None

    if market_type == "economics":
        series = _match_fred_series(market.question)
        if series:
            external_data = _fetch_fred_data(series)
    elif market_type == "weather":
        external_data = _fetch_weather_data(market.question)

    now = datetime.now(timezone.utc)
    days_left = (market.resolution_time - now).total_seconds() / 86400
    news = _fetch_news(market.question, days_left)
    if news:
        external_data = f"{external_data}\n\n{news}" if external_data else news

    cached = _estimate_cache.get(market.market_id)
    if cached and abs(cached["implied_prob"] - implied_prob) < CACHE_PRICE_TOLERANCE:
        estimated_prob = cached["estimated_prob"]
        reasoning = cached["reasoning"]
        log.info("[STRATEGY] CACHE HIT %s | implied=%.2f estimated=%.2f",
                 market.market_id, implied_prob, estimated_prob)
    else:
        log.info("[STRATEGY] %s type=%s data=%s news=%s", market.market_id, market_type,
                 "yes" if external_data else "no", "yes" if news else "no")

        user_msg = _build_user_message(market, implied_prob, external_data)

        cached_examples = list(FEW_SHOT_EXAMPLES)
        cached_examples[-1] = {
            "role": cached_examples[-1]["role"],
            "content": [{"type": "text", "text": cached_examples[-1]["content"], "cache_control": {"type": "ephemeral"}}],
        }

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]},
        ] + cached_examples + [{"role": "user", "content": user_msg}]

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=256,
            messages=messages,
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        match = re.search(r'\{[^}]+\}', text)
        if match:
            text = match.group(0)
        result = json.loads(text)
        estimated_prob = float(result["probability"])
        reasoning = result.get("reasoning", "")

        _estimate_cache[market.market_id] = {
            "implied_prob": implied_prob,
            "estimated_prob": estimated_prob,
            "reasoning": reasoning,
        }

    log.info(
        "[STRATEGY] %s | implied=%.2f estimated=%.2f | %s",
        market.market_id, implied_prob, estimated_prob, reasoning,
    )

    if market_type == "weather" and external_data:
        threshold = EDGE_THRESHOLD_WEATHER_WITH_DATA
    elif external_data:
        threshold = EDGE_THRESHOLD_WITH_DATA
    else:
        threshold = EDGE_THRESHOLD_DEFAULT

    yes_edge = estimated_prob - implied_prob
    no_edge = (1.0 - no_ask) - estimated_prob

    log.info("[STRATEGY] %s | threshold=%.2f", market.market_id, threshold)

    if yes_edge > threshold:
        return ("BUY", "YES", yes_edge)
    elif no_edge > threshold:
        return ("BUY", "NO", no_edge)
    return None
