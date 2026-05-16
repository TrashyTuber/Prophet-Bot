"""Fetch economic indicator data from FRED (Federal Reserve Economic Data).

Requires FRED_API_KEY in environment (free at https://fred.stlouisfed.org/docs/api/api_key.html).
Falls back to a no-auth summary if key is unavailable.
"""

import logging
import os
from datetime import datetime, timedelta

import httpx

log = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

INDICATOR_KEYWORDS = {
    "cpi": {
        "series": ["CPIAUCSL", "CPILFESL"],
        "label": "CPI (Consumer Price Index)",
    },
    "inflation": {
        "series": ["CPIAUCSL", "CPILFESL", "T5YIE"],
        "label": "Inflation Indicators",
    },
    "pce": {
        "series": ["PCEPI", "PCEPILFE"],
        "label": "PCE Price Index",
    },
    "unemployment": {
        "series": ["UNRATE", "ICSA"],
        "label": "Unemployment / Jobless Claims",
    },
    "jobless": {
        "series": ["ICSA", "CCSA"],
        "label": "Jobless Claims",
    },
    "nonfarm": {
        "series": ["PAYEMS"],
        "label": "Nonfarm Payrolls",
    },
    "payroll": {
        "series": ["PAYEMS"],
        "label": "Nonfarm Payrolls",
    },
    "gdp": {
        "series": ["GDP", "GDPC1"],
        "label": "GDP",
    },
    "fed funds": {
        "series": ["FEDFUNDS", "DFEDTARU", "DFEDTARL"],
        "label": "Federal Funds Rate",
    },
    "interest rate": {
        "series": ["FEDFUNDS", "DGS10", "DGS2"],
        "label": "Interest Rates",
    },
    "fomc": {
        "series": ["FEDFUNDS", "DFEDTARU"],
        "label": "FOMC / Fed Funds Rate",
    },
    "rate cut": {
        "series": ["FEDFUNDS", "DFEDTARU", "DFEDTARL"],
        "label": "Fed Funds Rate (for rate cut/hike analysis)",
    },
    "rate hike": {
        "series": ["FEDFUNDS", "DFEDTARU", "DFEDTARL"],
        "label": "Fed Funds Rate (for rate cut/hike analysis)",
    },
    "treasury": {
        "series": ["DGS10", "DGS2", "DGS30"],
        "label": "Treasury Yields",
    },
    "yield": {
        "series": ["DGS10", "DGS2", "T10Y2Y"],
        "label": "Treasury Yields / Yield Curve",
    },
    "retail sales": {
        "series": ["RSXFS", "MRTSSM44X72USS"],
        "label": "Retail Sales",
    },
    "housing": {
        "series": ["HOUST", "HSN1F", "CSUSHPINSA"],
        "label": "Housing Indicators",
    },
    "consumer confidence": {
        "series": ["UMCSENT"],
        "label": "Consumer Sentiment",
    },
    "ism": {
        "series": ["MANEMP"],
        "label": "ISM Manufacturing",
    },
    "oil": {
        "series": ["DCOILWTICO"],
        "label": "WTI Crude Oil Price",
    },
    "gas": {
        "series": ["GASREGW"],
        "label": "Regular Gas Price",
    },
    "sp500": {
        "series": ["SP500"],
        "label": "S&P 500",
    },
    "s&p": {
        "series": ["SP500"],
        "label": "S&P 500",
    },
}


def _match_indicators(question: str) -> list[dict]:
    question_lower = question.lower()
    matched = []
    seen_series = set()
    for keyword, info in INDICATOR_KEYWORDS.items():
        if keyword in question_lower:
            for series_id in info["series"]:
                if series_id not in seen_series:
                    seen_series.add(series_id)
                    matched.append({"series_id": series_id, "label": info["label"]})
    return matched


def _fetch_fred_series(series_id: str, api_key: str, n_observations: int = 12) -> list[dict] | None:
    try:
        resp = httpx.get(
            f"{FRED_BASE_URL}/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": n_observations,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        return [
            {"date": obs["date"], "value": obs["value"]}
            for obs in observations
            if obs.get("value") != "."
        ]
    except Exception as e:
        log.warning("FRED API error for %s: %s", series_id, e)
        return None


def _fetch_fred_series_info(series_id: str, api_key: str) -> dict | None:
    try:
        resp = httpx.get(
            f"{FRED_BASE_URL}/series",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        series_list = data.get("seriess", [])
        if series_list:
            s = series_list[0]
            return {
                "title": s.get("title"),
                "units": s.get("units"),
                "frequency": s.get("frequency"),
                "last_updated": s.get("last_updated"),
            }
    except Exception as e:
        log.warning("FRED series info error for %s: %s", series_id, e)
    return None


def fetch_economics_context(question: str) -> str | None:
    """Fetch economic data relevant to a market question.

    Returns a formatted string of recent indicator values, or None if no match.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        log.warning("FRED_API_KEY not set — skipping economics data fetch")
        return None

    indicators = _match_indicators(question)
    if not indicators:
        return None

    lines = ["Recent economic data relevant to this market:"]

    for indicator in indicators[:4]:
        series_id = indicator["series_id"]

        info = _fetch_fred_series_info(series_id, api_key)
        observations = _fetch_fred_series(series_id, api_key, n_observations=6)

        if not observations:
            continue

        title = info["title"] if info else series_id
        units = info.get("units", "") if info else ""
        freq = info.get("frequency", "") if info else ""

        lines.append(f"\n--- {title} ({series_id}) ---")
        if units:
            lines.append(f"Units: {units} | Frequency: {freq}")
        lines.append(f"{'Date':<12} {'Value':<15}")
        for obs in observations:
            lines.append(f"{obs['date']:<12} {obs['value']:<15}")

    if len(lines) == 1:
        return None

    return "\n".join(lines)
