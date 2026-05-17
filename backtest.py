"""Backtest the strategy against synthetic markets with known outcomes.

Runs predictions with and without external data to measure:
- Brier score (overall and per-category)
- Calibration curve (predicted vs actual hit rate by bucket)
- Value of external data pipeline
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from strategy import (
    _classify_market,
    _calibrate,
    _build_user_message,
    _fetch_fred_data,
    _match_fred_series,
    _fetch_weather_data,
    _fetch_news,
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    OPENROUTER_MODEL,
)
from data.sports import fetch_sports_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# ---------------------------------------------------------------------------
# Synthetic markets with known outcomes
# ---------------------------------------------------------------------------
# outcome: 1 = resolved YES, 0 = resolved NO
# implied_prob: the YES price the market was trading at (what we'd see as best_ask)

SYNTHETIC_MARKETS = [
    # --- SPORTS ---
    {
        "market_id": "wc-2022-argentina-winner",
        "question": "Will Argentina win the 2022 FIFA World Cup?",
        "description": "Resolves YES if Argentina wins the 2022 FIFA World Cup in Qatar.",
        "implied_prob": 0.18,
        "resolution_days": 5,
        "outcome": 1,
        "topic": "soccer",
        "family": "world-cup-2022-winner",
    },
    {
        "market_id": "wc-2022-brazil-winner",
        "question": "Will Brazil win the 2022 FIFA World Cup?",
        "description": "Resolves YES if Brazil wins the 2022 FIFA World Cup in Qatar.",
        "implied_prob": 0.22,
        "resolution_days": 5,
        "outcome": 0,
        "topic": "soccer",
        "family": "world-cup-2022-winner",
    },
    {
        "market_id": "nba-nuggets-2023-champ",
        "question": "Will the Denver Nuggets win the 2023 NBA Championship?",
        "description": "Resolves YES if Denver wins the 2023 NBA Finals.",
        "implied_prob": 0.30,
        "resolution_days": 10,
        "outcome": 1,
        "topic": "basketball",
        "family": "nba-2023-champion",
    },
    {
        "market_id": "nba-heat-2023-champ",
        "question": "Will the Miami Heat win the 2023 NBA Championship?",
        "description": "Resolves YES if Miami wins the 2023 NBA Finals.",
        "implied_prob": 0.20,
        "resolution_days": 10,
        "outcome": 0,
        "topic": "basketball",
        "family": "nba-2023-champion",
    },
    {
        "market_id": "sb-2024-chiefs",
        "question": "Will the Kansas City Chiefs win Super Bowl LVIII?",
        "description": "Resolves YES if the Chiefs win Super Bowl LVIII in February 2024.",
        "implied_prob": 0.52,
        "resolution_days": 3,
        "outcome": 1,
        "topic": "football",
        "family": "super-bowl-2024",
    },
    {
        "market_id": "ws-2024-dodgers",
        "question": "Will the Los Angeles Dodgers win the 2024 World Series?",
        "description": "Resolves YES if the Dodgers win the 2024 MLB World Series.",
        "implied_prob": 0.55,
        "resolution_days": 7,
        "outcome": 1,
        "topic": "baseball",
        "family": "world-series-2024",
    },
    {
        "market_id": "nba-celtics-2024-champ",
        "question": "Will the Boston Celtics win the 2024 NBA Championship?",
        "description": "Resolves YES if Boston wins the 2024 NBA Finals.",
        "implied_prob": 0.45,
        "resolution_days": 10,
        "outcome": 1,
        "topic": "basketball",
        "family": "nba-2024-champion",
    },

    # --- ECONOMICS ---
    {
        "market_id": "cpi-apr-2025-above-3",
        "question": "Will US CPI year-over-year exceed 3.0% for April 2025?",
        "description": "Resolves YES if the BLS reports CPI YoY above 3.0% for April 2025.",
        "implied_prob": 0.35,
        "resolution_days": 14,
        "outcome": 0,
        "topic": "economics",
        "family": "cpi-apr-2025",
    },
    {
        "market_id": "cpi-apr-2025-above-2",
        "question": "Will US CPI year-over-year exceed 2.0% for April 2025?",
        "description": "Resolves YES if the BLS reports CPI YoY above 2.0% for April 2025.",
        "implied_prob": 0.92,
        "resolution_days": 14,
        "outcome": 1,
        "topic": "economics",
        "family": "cpi-apr-2025",
    },
    {
        "market_id": "unemployment-apr-2025-above-4",
        "question": "Will the US unemployment rate exceed 4.0% in April 2025?",
        "description": "Resolves YES if BLS reports unemployment above 4.0%.",
        "implied_prob": 0.55,
        "resolution_days": 14,
        "outcome": 1,
        "topic": "economics",
        "family": "unemployment-apr-2025",
    },
    {
        "market_id": "fed-rate-hold-may-2025",
        "question": "Will the Federal Reserve hold interest rates steady at the May 2025 FOMC meeting?",
        "description": "Resolves YES if the Fed keeps the fed funds rate unchanged.",
        "implied_prob": 0.88,
        "resolution_days": 7,
        "outcome": 1,
        "topic": "economics",
        "family": "fomc-may-2025",
    },
    {
        "market_id": "gdp-q1-2025-positive",
        "question": "Will US GDP growth be positive in Q1 2025?",
        "description": "Resolves YES if BEA reports positive real GDP growth for Q1 2025.",
        "implied_prob": 0.75,
        "resolution_days": 30,
        "outcome": 0,
        "topic": "economics",
        "family": "gdp-q1-2025",
    },
    {
        "market_id": "sp500-above-5500-apr-2025",
        "question": "Will the S&P 500 close above 5,500 on April 30, 2025?",
        "description": "Resolves YES if SPX closing price exceeds 5,500 on April 30.",
        "implied_prob": 0.60,
        "resolution_days": 14,
        "outcome": 0,
        "topic": "economics",
        "family": "sp500-apr-2025",
    },
    {
        "market_id": "treasury-10y-above-4-5",
        "question": "Will the 10-year Treasury yield exceed 4.5% on May 1, 2025?",
        "description": "Resolves YES if 10Y yield closes above 4.5%.",
        "implied_prob": 0.45,
        "resolution_days": 14,
        "outcome": 0,
        "topic": "economics",
        "family": "treasury-may-2025",
    },

    # --- WEATHER ---
    {
        "market_id": "nyc-temp-high-90-jul2025",
        "question": "Will the high temperature in New York City exceed 90°F on July 4, 2025?",
        "description": "Resolves YES if NWS reports high temp above 90°F in Central Park.",
        "implied_prob": 0.35,
        "resolution_days": 3,
        "outcome": 0,
        "topic": "weather",
        "family": "nyc-temp-jul4",
    },
    {
        "market_id": "miami-temp-above-85-jan2025",
        "question": "Will the high temperature in Miami exceed 85°F on January 15, 2025?",
        "description": "Resolves YES if NWS reports high temp above 85°F in Miami.",
        "implied_prob": 0.30,
        "resolution_days": 3,
        "outcome": 0,
        "topic": "weather",
        "family": "miami-temp-jan",
    },
    {
        "market_id": "chicago-snow-dec2024",
        "question": "Will Chicago receive more than 2 inches of snow in December 2024?",
        "description": "Resolves YES if total December snowfall exceeds 2 inches.",
        "implied_prob": 0.82,
        "resolution_days": 14,
        "outcome": 1,
        "topic": "weather",
        "family": "chicago-snow-dec",
    },
    {
        "market_id": "phoenix-above-110-jun2025",
        "question": "Will the high temperature in Phoenix exceed 110°F at any point in June 2025?",
        "description": "Resolves YES if any day in June 2025 has a high above 110°F.",
        "implied_prob": 0.75,
        "resolution_days": 20,
        "outcome": 1,
        "topic": "weather",
        "family": "phoenix-temp-jun",
    },
    {
        "market_id": "seattle-rain-apr2025",
        "question": "Will Seattle have more than 15 days of measurable rain in April 2025?",
        "description": "Resolves YES if NWS reports more than 15 rainy days in April.",
        "implied_prob": 0.55,
        "resolution_days": 14,
        "outcome": 1,
        "topic": "weather",
        "family": "seattle-rain-apr",
    },
    {
        "market_id": "denver-temp-below-32-may2025",
        "question": "Will Denver see a low temperature below 32°F in May 2025?",
        "description": "Resolves YES if any day in May has a low below freezing.",
        "implied_prob": 0.40,
        "resolution_days": 20,
        "outcome": 1,
        "topic": "weather",
        "family": "denver-temp-may",
    },
]


# ---------------------------------------------------------------------------
# Mock market object to match the interface analyze_market expects
# ---------------------------------------------------------------------------

class MockQuote:
    def __init__(self, best_ask, best_bid):
        self.best_ask = str(best_ask)
        self.best_bid = str(best_bid)


class MockMarket:
    def __init__(self, m):
        self.market_id = m["market_id"]
        self.question = m["question"]
        self.description = m.get("description", "")
        self.topic = m.get("topic", "")
        self.family = m.get("family", "")
        implied = m["implied_prob"]
        self.quote = MockQuote(best_ask=implied, best_bid=1.0 - implied)
        self.resolution_time = datetime.now(timezone.utc) + timedelta(days=m["resolution_days"])


# ---------------------------------------------------------------------------
# Prediction function (mirrors analyze_market but returns probability)
# ---------------------------------------------------------------------------

def predict_market(market, use_external_data=True):
    """Get model's estimated probability for a market.

    Returns (estimated_prob, reasoning, market_type, had_data) or None on failure.
    """
    implied_prob = float(market.quote.best_ask)
    market_type = _classify_market(market)

    external_data = None
    if use_external_data:
        if market_type == "economics":
            series = _match_fred_series(market.question)
            if series:
                external_data = _fetch_fred_data(series)
        elif market_type == "weather":
            external_data = _fetch_weather_data(market.question)
        elif market_type == "sports":
            external_data = fetch_sports_context(market.question)

        days_left = (market.resolution_time - datetime.now(timezone.utc)).total_seconds() / 86400
        news = _fetch_news(market.question, days_left)
        if news:
            external_data = f"{external_data}\n\n{news}" if external_data else news

    user_msg = _build_user_message(market, implied_prob, external_data)

    cached_examples = list(FEW_SHOT_EXAMPLES)
    cached_examples[-1] = {
        "role": cached_examples[-1]["role"],
        "content": [{"type": "text", "text": cached_examples[-1]["content"],
                      "cache_control": {"type": "ephemeral"}}],
    }

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT,
                                         "cache_control": {"type": "ephemeral"}}]},
    ] + cached_examples + [{"role": "user", "content": user_msg}]

    for attempt in range(2):
        try:
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
            raw_prob = float(result["probability"])
            reasoning = result.get("reasoning", "")
            has_data = external_data is not None
            calibrated_prob = _calibrate(raw_prob, implied_prob, market_type, has_data)
            return raw_prob, calibrated_prob, reasoning, market_type, has_data
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("Parse error on %s (attempt %d): %s", market.market_id, attempt + 1, e)
        except Exception as e:
            log.warning("LLM error on %s (attempt %d): %s", market.market_id, attempt + 1, e)

    return None


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def brier_score(predictions):
    """Compute Brier score: mean((predicted - actual)^2). Lower is better."""
    if not predictions:
        return float("nan")
    return sum((p["estimated"] - p["outcome"]) ** 2 for p in predictions) / len(predictions)


def calibration_curve(predictions, n_bins=5):
    """Bin predictions and compute actual hit rate per bin."""
    bins = [{"low": i / n_bins, "high": (i + 1) / n_bins, "preds": []} for i in range(n_bins)]

    for p in predictions:
        for b in bins:
            if b["low"] <= p["estimated"] < b["high"] or (b["high"] == 1.0 and p["estimated"] == 1.0):
                b["preds"].append(p)
                break

    results = []
    for b in bins:
        if b["preds"]:
            avg_predicted = sum(p["estimated"] for p in b["preds"]) / len(b["preds"])
            actual_rate = sum(p["outcome"] for p in b["preds"]) / len(b["preds"])
            results.append({
                "range": f"{b['low']*100:.0f}-{b['high']*100:.0f}%",
                "count": len(b["preds"]),
                "avg_predicted": avg_predicted,
                "actual_rate": actual_rate,
                "gap": abs(avg_predicted - actual_rate),
            })
    return results


def print_report(title, predictions):
    """Print a full calibration report for a set of predictions."""
    if not predictions:
        print(f"\n{title}: No predictions to report.")
        return

    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

    # Overall Brier score
    bs = brier_score(predictions)
    print(f"\n  Overall Brier Score:  {bs:.4f}  (0.250 = random, lower = better)")

    # Per market type
    by_type = {}
    for p in predictions:
        mt = p["market_type"]
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(p)

    print(f"\n  {'Category':<14} {'Brier':>8} {'Count':>7} {'Avg Edge':>10} {'Hit Rate':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*10} {'-'*10}")
    for mt in ["sports", "economics", "weather"]:
        group = by_type.get(mt, [])
        if group:
            bs_mt = brier_score(group)
            avg_edge = sum(abs(p["estimated"] - p["implied"]) for p in group) / len(group)
            hit_rate = sum(1 for p in group if (p["estimated"] > 0.5) == (p["outcome"] == 1)) / len(group)
            print(f"  {mt:<14} {bs_mt:>8.4f} {len(group):>7} {avg_edge:>10.4f} {hit_rate:>9.0%}")

    # Calibration curve
    curve = calibration_curve(predictions)
    print(f"\n  Calibration Curve:")
    print(f"  {'Bucket':<12} {'Count':>6} {'Avg Pred':>10} {'Actual':>10} {'Gap':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for b in curve:
        status = "✓" if b["gap"] < 0.10 else ("~" if b["gap"] < 0.20 else "✗")
        print(f"  {b['range']:<12} {b['count']:>6} {b['avg_predicted']:>10.2%} {b['actual_rate']:>10.2%} {b['gap']:>7.2%} {status}")

    # Overconfidence analysis
    over = [p for p in predictions if p["estimated"] > 0.70]
    under = [p for p in predictions if p["estimated"] < 0.30]
    if over:
        over_hit = sum(p["outcome"] for p in over) / len(over)
        over_avg = sum(p["estimated"] for p in over) / len(over)
        print(f"\n  High-confidence (>70%): predicted avg {over_avg:.0%}, actual {over_hit:.0%} ({len(over)} markets)")
    if under:
        under_hit = sum(p["outcome"] for p in under) / len(under)
        under_avg = sum(p["estimated"] for p in under) / len(under)
        print(f"  Low-confidence (<30%):  predicted avg {under_avg:.0%}, actual {under_hit:.0%} ({len(under)} markets)")

    # Edge analysis
    correct = [p for p in predictions if (p["estimated"] > 0.5) == (p["outcome"] == 1)]
    wrong = [p for p in predictions if (p["estimated"] > 0.5) != (p["outcome"] == 1)]
    if correct:
        avg_edge_correct = sum(abs(p["estimated"] - p["implied"]) for p in correct) / len(correct)
        print(f"\n  Correct predictions ({len(correct)}): avg edge = {avg_edge_correct:.4f}")
    if wrong:
        avg_edge_wrong = sum(abs(p["estimated"] - p["implied"]) for p in wrong) / len(wrong)
        print(f"  Wrong predictions ({len(wrong)}):   avg edge = {avg_edge_wrong:.4f}")

    print(f"\n{'=' * 65}")


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 65)
    print("  CALIBRATION BACKTEST")
    print(f"  {len(SYNTHETIC_MARKETS)} synthetic markets | Model: {OPENROUTER_MODEL}")
    print("=" * 65)

    raw_with_data = []
    raw_no_data = []
    cal_with_data = []
    cal_no_data = []

    for i, m in enumerate(SYNTHETIC_MARKETS):
        market = MockMarket(m)
        market_type = _classify_market(market)
        log.info("[%d/%d] %s (type=%s, implied=%.2f, outcome=%d)",
                 i + 1, len(SYNTHETIC_MARKETS), m["market_id"], market_type,
                 m["implied_prob"], m["outcome"])

        # Option A: with external data
        result_a = predict_market(market, use_external_data=True)
        if result_a:
            raw_a, cal_a, reason_a, mt_a, had_data = result_a
            base = {
                "market_id": m["market_id"],
                "market_type": market_type,
                "implied": m["implied_prob"],
                "outcome": m["outcome"],
                "reasoning": reason_a,
                "had_external_data": had_data,
            }
            raw_with_data.append({**base, "estimated": raw_a})
            cal_with_data.append({**base, "estimated": cal_a})
            log.info("  WITH DATA: raw=%.2f calibrated=%.2f implied=%.2f outcome=%d %s",
                     raw_a, cal_a, m["implied_prob"], m["outcome"],
                     "✓" if (cal_a > 0.5) == (m["outcome"] == 1) else "✗")

        time.sleep(0.5)

        # Option B: without external data
        result_b = predict_market(market, use_external_data=False)
        if result_b:
            raw_b, cal_b, reason_b, mt_b, _ = result_b
            base = {
                "market_id": m["market_id"],
                "market_type": market_type,
                "implied": m["implied_prob"],
                "outcome": m["outcome"],
                "reasoning": reason_b,
                "had_external_data": False,
            }
            raw_no_data.append({**base, "estimated": raw_b})
            cal_no_data.append({**base, "estimated": cal_b})
            log.info("  NO DATA:   raw=%.2f calibrated=%.2f implied=%.2f outcome=%d %s",
                     raw_b, cal_b, m["implied_prob"], m["outcome"],
                     "✓" if (cal_b > 0.5) == (m["outcome"] == 1) else "✗")

        time.sleep(0.5)

    # Print reports — raw vs calibrated
    print_report("RAW (uncalibrated) — WITH EXTERNAL DATA", raw_with_data)
    print_report("CALIBRATED — WITH EXTERNAL DATA", cal_with_data)
    print_report("RAW (uncalibrated) — WITHOUT EXTERNAL DATA", raw_no_data)
    print_report("CALIBRATED — WITHOUT EXTERNAL DATA", cal_no_data)

    # Comparison: calibration improvement
    if raw_with_data and cal_with_data:
        print(f"\n{'=' * 65}")
        print(f"  CALIBRATION IMPROVEMENT (with data)")
        print(f"{'=' * 65}")
        bs_raw = brier_score(raw_with_data)
        bs_cal = brier_score(cal_with_data)
        improvement = (bs_raw - bs_cal) / bs_raw * 100 if bs_raw > 0 else 0
        print(f"  Brier raw:         {bs_raw:.4f}")
        print(f"  Brier calibrated:  {bs_cal:.4f}")
        print(f"  Improvement:       {improvement:.1f}%")

    if raw_no_data and cal_no_data:
        print(f"\n{'=' * 65}")
        print(f"  CALIBRATION IMPROVEMENT (no data)")
        print(f"{'=' * 65}")
        bs_raw = brier_score(raw_no_data)
        bs_cal = brier_score(cal_no_data)
        improvement = (bs_raw - bs_cal) / bs_raw * 100 if bs_raw > 0 else 0
        print(f"  Brier raw:         {bs_raw:.4f}")
        print(f"  Brier calibrated:  {bs_cal:.4f}")
        print(f"  Improvement:       {improvement:.1f}%")

    # Best strategy comparison
    if cal_with_data and cal_no_data:
        print(f"\n{'=' * 65}")
        print(f"  BEST STRATEGY PER CATEGORY (calibrated)")
        print(f"{'=' * 65}")
        types_a = {}
        types_b = {}
        for p in cal_with_data:
            types_a.setdefault(p["market_type"], []).append(p)
        for p in cal_no_data:
            types_b.setdefault(p["market_type"], []).append(p)

        print(f"  {'Category':<14} {'With Data':>10} {'No Data':>10} {'Best':>12}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*12}")
        for mt in ["sports", "economics", "weather"]:
            ga = types_a.get(mt, [])
            gb = types_b.get(mt, [])
            if ga and gb:
                bsa = brier_score(ga)
                bsb = brier_score(gb)
                best = "with data" if bsa < bsb else "no data"
                print(f"  {mt:<14} {bsa:>10.4f} {bsb:>10.4f} {best:>12}")

        print(f"\n{'=' * 65}")

    # Save raw results
    output = {
        "raw_with_data": raw_with_data,
        "calibrated_with_data": cal_with_data,
        "raw_no_data": raw_no_data,
        "calibrated_no_data": cal_no_data,
        "summary": {
            "brier_raw_with_data": brier_score(raw_with_data),
            "brier_calibrated_with_data": brier_score(cal_with_data),
            "brier_raw_no_data": brier_score(raw_no_data),
            "brier_calibrated_no_data": brier_score(cal_no_data),
            "n_markets": len(SYNTHETIC_MARKETS),
            "model": OPENROUTER_MODEL,
        },
    }
    with open("backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw results saved to backtest_results.json")


if __name__ == "__main__":
    run_backtest()
