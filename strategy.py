"""Trading strategy with domain-specific data enrichment for weather and economics markets."""

import json
import logging

import anthropic
from dotenv import load_dotenv

from data import fetch_weather_context, fetch_economics_context

load_dotenv()

log = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.05
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

client = anthropic.Anthropic()


def _classify_market(question: str) -> str:
    q = question.lower()
    weather_keywords = ["temperature", "rain", "snow", "precipitation", "weather", "degrees", "°f", "°c", "wind", "hurricane", "storm", "tornado", "heat", "cold", "freeze", "frost"]
    econ_keywords = ["cpi", "inflation", "gdp", "unemployment", "fed", "fomc", "interest rate", "rate cut", "rate hike", "payroll", "nonfarm", "jobless", "pce", "treasury", "yield", "retail sales", "housing", "oil", "gas price", "s&p", "sp500", "consumer confidence", "ism"]

    for kw in weather_keywords:
        if kw in q:
            return "weather"
    for kw in econ_keywords:
        if kw in q:
            return "economics"
    return "general"


def _build_prompt(market, market_type: str, context: str | None) -> str:
    yes_ask = float(market.quote.best_ask)
    no_ask = 1.0 - float(market.quote.best_bid)
    implied_prob = yes_ask

    parts = []
    parts.append(
        f"You are an expert prediction market analyst specializing in {market_type} markets. "
        f"Estimate the probability that the following market resolves YES.\n"
    )
    parts.append(f"Market question: {market.question}")
    if market.description:
        parts.append(f"Description: {market.description}")
    parts.append(f"Resolution time: {market.resolution_time.isoformat()}")
    if market.topic:
        parts.append(f"Topic: {market.topic}")
    parts.append(f"Current YES price (implied probability): {implied_prob:.3f}")
    parts.append(f"Current NO price: {no_ask:.3f}")
    parts.append(f"24h volume: {market.quote.volume_24h}")

    if context:
        parts.append(f"\n--- RELEVANT DATA ---\n{context}")

    parts.append(
        f"\nUsing the data above, estimate the TRUE probability this resolves YES. "
        f"Consider: base rates, historical patterns, current conditions, and how close "
        f"the resolution date is. Be precise — a 1-2% edge matters in prediction markets."
        f"\n\nRespond with ONLY a JSON object:\n"
        f'{{"probability": <float 0-1>, "confidence": <float 0-1>, "reasoning": "1-2 sentences"}}'
    )

    return "\n".join(parts)


def analyze_market(market) -> tuple[str, str] | None:
    """Analyze a market and return (action, side) or None to skip."""
    market_type = _classify_market(market.question)

    if market_type == "general":
        log.info("[SKIP] Non-target market: %s", market.market_id)
        return None

    context = None
    if market_type == "weather":
        context = fetch_weather_context(market.question, market.resolution_time)
    elif market_type == "economics":
        context = fetch_economics_context(market.question)

    prompt = _build_prompt(market, market_type, context)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    estimated_prob = float(result["probability"])
    confidence = float(result.get("confidence", 0.5))
    reasoning = result.get("reasoning", "")

    yes_ask = float(market.quote.best_ask)
    no_ask = 1.0 - float(market.quote.best_bid)

    log.info(
        "[%s] %s | implied=%.3f estimated=%.3f conf=%.2f | %s",
        market_type.upper(), market.market_id, yes_ask, estimated_prob, confidence, reasoning,
    )

    if confidence < 0.4:
        log.info("  Low confidence (%.2f), skipping", confidence)
        return None

    edge_yes = estimated_prob - yes_ask
    edge_no = (1.0 - estimated_prob) - no_ask

    if edge_yes > EDGE_THRESHOLD:
        return ("BUY", "YES")
    elif edge_no > EDGE_THRESHOLD:
        return ("BUY", "NO")

    return None
