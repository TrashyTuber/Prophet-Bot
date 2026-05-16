"""Fetch weather forecast data from Open-Meteo (free, no API key required).

Supports temperature, precipitation, wind, and snow forecasts for US cities.
"""

import logging
from datetime import datetime, timedelta

import httpx

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

US_CITIES = {
    "new york": (40.71, -74.01),
    "nyc": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "la": (34.05, -118.24),
    "chicago": (41.88, -87.63),
    "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07),
    "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49),
    "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80),
    "miami": (25.76, -80.19),
    "atlanta": (33.75, -84.39),
    "boston": (42.36, -71.06),
    "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99),
    "washington": (38.91, -77.04),
    "dc": (38.91, -77.04),
    "nashville": (36.16, -86.78),
    "detroit": (42.33, -83.05),
    "minneapolis": (44.98, -93.27),
    "san francisco": (37.77, -122.42),
    "sf": (37.77, -122.42),
    "las vegas": (36.17, -115.14),
    "portland": (45.52, -122.68),
    "austin": (30.27, -97.74),
    "orlando": (28.54, -81.38),
    "tampa": (27.95, -82.46),
    "charlotte": (35.23, -80.84),
    "st louis": (38.63, -90.20),
    "pittsburgh": (40.44, -79.99),
    "salt lake city": (40.76, -111.89),
    "kansas city": (39.10, -94.58),
    "sacramento": (38.58, -121.49),
    "new orleans": (29.95, -90.07),
    "cleveland": (41.50, -81.69),
    "milwaukee": (43.04, -87.91),
    "oklahoma city": (35.47, -97.52),
    "memphis": (35.15, -90.05),
    "louisville": (38.25, -85.76),
    "baltimore": (39.29, -76.61),
    "buffalo": (42.89, -78.88),
    "anchorage": (61.22, -149.90),
    "honolulu": (21.31, -157.86),
}


def _find_city(text: str) -> tuple[str, float, float] | None:
    text_lower = text.lower()
    for city, (lat, lon) in US_CITIES.items():
        if city in text_lower:
            return city, lat, lon
    return None


def fetch_weather_context(question: str, resolution_time: datetime) -> str | None:
    """Fetch weather forecast relevant to a market question.

    Returns a formatted string of forecast data, or None if no city is detected.
    """
    city_match = _find_city(question)
    if not city_match:
        return None

    city_name, lat, lon = city_match

    now = datetime.now()
    days_ahead = (resolution_time - now).days
    if days_ahead < 0:
        return None
    forecast_days = min(days_ahead + 2, 16)

    try:
        resp = httpx.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "snowfall_sum",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                ]),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "America/New_York",
                "forecast_days": forecast_days,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Weather API error: %s", e)
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return None

    lines = [f"Weather forecast for {city_name.title()} (next {len(dates)} days):"]
    lines.append(f"{'Date':<12} {'High°F':<8} {'Low°F':<8} {'Precip':<8} {'Prob%':<7} {'Snow':<8} {'Wind':<8} {'Gusts':<8}")

    for i, date in enumerate(dates):
        high = daily.get("temperature_2m_max", [None])[i]
        low = daily.get("temperature_2m_min", [None])[i]
        precip = daily.get("precipitation_sum", [None])[i]
        prob = daily.get("precipitation_probability_max", [None])[i]
        snow = daily.get("snowfall_sum", [None])[i]
        wind = daily.get("wind_speed_10m_max", [None])[i]
        gusts = daily.get("wind_gusts_10m_max", [None])[i]

        lines.append(
            f"{date:<12} {high or '-':<8} {low or '-':<8} "
            f"{f'{precip:.2f}in' if precip is not None else '-':<8} "
            f"{f'{prob}%' if prob is not None else '-':<7} "
            f"{f'{snow:.1f}in' if snow is not None else '-':<8} "
            f"{f'{wind:.0f}mph' if wind is not None else '-':<8} "
            f"{f'{gusts:.0f}mph' if gusts is not None else '-':<8}"
        )

    return "\n".join(lines)
