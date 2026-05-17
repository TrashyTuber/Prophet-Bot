"""Calibration script using Prophet-Arena-Subset-1200 dataset.

Downloads resolved markets from HuggingFace, runs scout (and optionally judge)
predictions, and reports:
- Brier scores (overall and per-category)
- Calibration curves
- Trading return simulation (buy when edge > threshold, compare to resolution)
- Scout-only vs Scout+Judge comparison

Usage:
    python calibrate.py                    # Run scout only on sports subset
    python calibrate.py --judge            # Run scout + judge
    python calibrate.py --category sports  # Filter to specific category
    python calibrate.py --limit 50         # Limit number of markets
    python calibrate.py --resume           # Resume from cached results
"""

import argparse
import ast
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)

from strategy import (
    _classify_market,
    _calibrate,
    _build_user_message,
    _fetch_external_data_for_market,
    _edge_threshold,
    _parse_json_object,
    review_trade_candidate,
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    OPENROUTER_SCOUT_MODEL,
    OPENROUTER_JUDGE_MODEL,
    SHRINKAGE_WEIGHTS,
    JUDGE_SHRINKAGE_WEIGHTS,
    PROB_FLOOR,
    PROB_CEILING,
    MAX_DAYS_TO_RESOLUTION,
)
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

CACHE_FILE = "calibration_cache.json"
RESULTS_FILE = "calibration_results.json"

CATEGORY_MAP = {
    "Sports": "sports",
    "Economics": "economics",
    "Climate and Weather": "weather",
    "Politics": "politics",
    "Entertainment": "entertainment",
    "Companies": "general",
    "Mentions": "general",
    "Other": "general",
}


def load_dataset():
    """Load the Prophet-Arena-Subset-1200 dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        raise SystemExit(1)

    log.info("Loading Prophet-Arena-Subset-1200 from HuggingFace...")
    ds = load_dataset("prophetarena/Prophet-Arena-Subset-1200", split="train")
    log.info("Loaded %d rows", len(ds))
    return ds


def _safe_parse(value):
    """Parse a field that might be JSON string, Python repr string, or already parsed."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return ast.literal_eval(value)


def parse_market_row(row):
    """Parse a dataset row into individual markets with outcomes."""
    markets_raw = _safe_parse(row["markets"])
    market_outcome = _safe_parse(row["market_outcome"])
    market_data = _safe_parse(row["market_data"])

    if not isinstance(market_outcome, dict) or not isinstance(market_data, dict):
        return []

    category = CATEGORY_MAP.get(row["category"], "general")
    close_time = row["close_time"]
    snapshot_time = row["snapshot_time"]

    # markets_raw can be a list of strings (outcome names) or list of dicts
    market_names = []
    if isinstance(markets_raw, list):
        for item in markets_raw:
            if isinstance(item, str):
                market_names.append(item)
            elif isinstance(item, dict):
                market_names.append(item.get("title") or item.get("ticker") or "")

    # If markets list is empty, use keys from market_outcome
    if not market_names:
        market_names = list(market_outcome.keys())

    parsed = []
    for name in market_names:
        if not name:
            continue

        # Match to outcome
        if name in market_outcome:
            outcome = market_outcome[name]
        else:
            continue

        # Match to price data
        if name in market_data:
            prices = market_data[name]
        else:
            continue

        yes_ask = prices.get("yes_ask")
        no_ask = prices.get("no_ask")
        yes_bid = prices.get("yes_bid")

        if yes_ask is None or no_ask is None:
            continue

        # Prices are in cents (0-100), convert to 0-1
        yes_ask_dec = yes_ask / 100.0
        yes_bid_dec = (yes_bid / 100.0) if yes_bid is not None else (1.0 - no_ask / 100.0)

        if yes_ask_dec <= 0.01 or yes_ask_dec >= 0.99:
            continue

        market_id = f"{row['event_ticker']}_{name[:30]}"

        parsed.append({
            "market_id": market_id,
            "question": row.get("augmented_title") or row["title"],
            "subtitle": name,
            "category": category,
            "yes_ask": yes_ask_dec,
            "yes_bid": yes_bid_dec,
            "outcome": int(outcome),
            "close_time": close_time,
            "snapshot_time": snapshot_time,
            "event_ticker": row["event_ticker"],
            "family": row["event_ticker"],
            "liquidity": prices.get("liquidity", 0),
        })

    return parsed


class MockQuote:
    def __init__(self, best_ask, best_bid):
        self.best_ask = str(best_ask)
        self.best_bid = str(best_bid)


class MockMarket:
    def __init__(self, m):
        self.market_id = m["market_id"]
        self.question = m["question"]
        self.description = m.get("subtitle", "")
        self.topic = m.get("category", "")
        self.family = m.get("family", "")
        self.quote = MockQuote(best_ask=m["yes_ask"], best_bid=m["yes_bid"])
        if isinstance(m["close_time"], str):
            self.resolution_time = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
        else:
            self.resolution_time = m["close_time"]
        if self.resolution_time.tzinfo is None:
            self.resolution_time = self.resolution_time.replace(tzinfo=timezone.utc)


def predict_scout(market, category):
    """Run the scout model on a market. Returns (raw_prob, reasoning) or None."""
    implied_prob = float(market.quote.best_ask)
    market_type = category

    days_left = (market.resolution_time - datetime.now(timezone.utc)).total_seconds() / 86400
    # For historical markets, use a synthetic days_left based on original timing
    if days_left < 0:
        days_left = 14.0

    external_data = None
    # Skip external data for historical markets — we can't fetch past forecasts
    # But news search might still work for recent events

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
                model=OPENROUTER_SCOUT_MODEL,
                max_tokens=256,
                messages=messages,
            )
            result = _parse_json_object(response.choices[0].message.content or "")
            raw_prob = float(result["probability"])
            reasoning = result.get("reasoning", "")
            return raw_prob, reasoning
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning("Scout parse error (attempt %d): %s", attempt + 1, e)
        except Exception as e:
            log.warning("Scout LLM error (attempt %d): %s", attempt + 1, e)

    return None


def predict_judge(market, category, scout_side, scout_edge, scout_probability):
    """Run the judge model. Returns (approved, raw_prob, reasoning) or None."""
    reviewed = review_trade_candidate(market, "BUY", scout_side, scout_edge)
    if reviewed is None:
        return False, scout_probability, "rejected"
    _, _, judged_edge = reviewed
    implied_prob = float(market.quote.best_ask)
    if scout_side == "YES":
        judged_prob = implied_prob + judged_edge
    else:
        judged_prob = float(market.quote.best_bid) - judged_edge
    return True, judged_prob, "approved"


def simulate_trade(estimated_prob, implied_prob, outcome, category, has_data):
    """Simulate a trade decision and compute return."""
    threshold = _edge_threshold(category, has_data)
    yes_edge = estimated_prob - implied_prob
    no_edge = (1.0 - implied_prob) - estimated_prob

    if yes_edge > threshold:
        # Buy YES at implied_prob, pays 1.0 if outcome=1
        cost = implied_prob
        payout = 1.0 if outcome == 1 else 0.0
        return {"action": "BUY_YES", "cost": cost, "payout": payout,
                "return": (payout - cost) / cost, "edge": yes_edge}
    elif no_edge > threshold:
        # Buy NO at (1 - implied_prob), pays 1.0 if outcome=0
        cost = 1.0 - implied_prob
        payout = 1.0 if outcome == 0 else 0.0
        return {"action": "BUY_NO", "cost": cost, "payout": payout,
                "return": (payout - cost) / cost, "edge": no_edge}

    return None


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def brier_score(predictions):
    if not predictions:
        return float("nan")
    return sum((p["estimated"] - p["outcome"]) ** 2 for p in predictions) / len(predictions)


def calibration_curve(predictions, n_bins=5):
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


def print_report(title, predictions, trades):
    if not predictions:
        print(f"\n{title}: No predictions.")
        return

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    bs = brier_score(predictions)
    market_bs = brier_score([{"estimated": p["implied"], "outcome": p["outcome"]} for p in predictions])
    print(f"\n  Model Brier Score:  {bs:.4f}")
    print(f"  Market Brier Score: {market_bs:.4f}  (baseline — just trusting the market)")
    improvement = (market_bs - bs) / market_bs * 100 if market_bs > 0 else 0
    print(f"  Improvement:        {improvement:+.1f}%")

    # Per category
    by_type = {}
    for p in predictions:
        by_type.setdefault(p["category"], []).append(p)

    print(f"\n  {'Category':<14} {'Model':>8} {'Market':>8} {'Count':>7} {'Hit Rate':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
    for cat in sorted(by_type.keys()):
        group = by_type[cat]
        bs_model = brier_score(group)
        bs_mkt = brier_score([{"estimated": p["implied"], "outcome": p["outcome"]} for p in group])
        hit_rate = sum(1 for p in group if (p["estimated"] > 0.5) == (p["outcome"] == 1)) / len(group)
        print(f"  {cat:<14} {bs_model:>8.4f} {bs_mkt:>8.4f} {len(group):>7} {hit_rate:>9.0%}")

    # Calibration curve
    curve = calibration_curve(predictions)
    print(f"\n  Calibration Curve:")
    print(f"  {'Bucket':<12} {'Count':>6} {'Avg Pred':>10} {'Actual':>10} {'Gap':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for b in curve:
        status = "+" if b["gap"] < 0.10 else ("~" if b["gap"] < 0.20 else "-")
        print(f"  {b['range']:<12} {b['count']:>6} {b['avg_predicted']:>10.2%} {b['actual_rate']:>10.2%} {b['gap']:>7.2%} {status}")

    # Trading simulation
    if trades:
        wins = [t for t in trades if t["return"] > 0]
        losses = [t for t in trades if t["return"] <= 0]
        total_invested = sum(t["cost"] for t in trades)
        total_payout = sum(t["payout"] for t in trades)
        avg_return = sum(t["return"] for t in trades) / len(trades)
        avg_edge = sum(t["edge"] for t in trades) / len(trades)

        print(f"\n  Trading Simulation ({len(trades)} trades):")
        print(f"  Win rate:        {len(wins)}/{len(trades)} ({len(wins)/len(trades):.0%})")
        print(f"  Avg return:      {avg_return:+.2%}")
        print(f"  Avg edge:        {avg_edge:.2%}")
        print(f"  Total invested:  ${total_invested:.2f}")
        print(f"  Total payout:    ${total_payout:.2f}")
        print(f"  Net P&L:         ${total_payout - total_invested:+.2f} ({(total_payout/total_invested - 1)*100:+.1f}%)")

        # By category
        trades_by_cat = {}
        for t in trades:
            trades_by_cat.setdefault(t["category"], []).append(t)

        print(f"\n  {'Category':<14} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} {'Net P&L':>10}")
        print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*9} {'-'*10}")
        for cat in sorted(trades_by_cat.keys()):
            ct = trades_by_cat[cat]
            cw = sum(1 for t in ct if t["return"] > 0)
            cr = sum(t["return"] for t in ct) / len(ct)
            cpnl = sum(t["payout"] - t["cost"] for t in ct)
            print(f"  {cat:<14} {len(ct):>7} {cw/len(ct):>6.0%} {cr:>+8.2%} {cpnl:>+9.2f}")

    print(f"\n{'=' * 70}")


def run_calibration(args):
    dataset = load_dataset()
    cache = load_cache() if args.resume else {}

    # Parse all markets from dataset
    all_markets = []
    for row in dataset:
        parsed = parse_market_row(row)
        all_markets.extend(parsed)

    log.info("Parsed %d individual markets from dataset", len(all_markets))

    # Filter by category
    if args.category:
        all_markets = [m for m in all_markets if m["category"] == args.category]
        log.info("Filtered to %d markets in category '%s'", len(all_markets), args.category)

    # Filter out extreme prices (no edge possible)
    all_markets = [m for m in all_markets if 0.05 < m["yes_ask"] < 0.95]

    # Limit
    if args.limit:
        all_markets = all_markets[:args.limit]

    log.info("Running calibration on %d markets (scout=%s, judge=%s)",
             len(all_markets), OPENROUTER_SCOUT_MODEL,
             OPENROUTER_JUDGE_MODEL if args.judge else "disabled")

    scout_predictions = []
    judge_predictions = []
    scout_trades = []
    judge_trades = []

    for i, m in enumerate(all_markets):
        market = MockMarket(m)
        market_id = m["market_id"]
        category = m["category"]
        implied_prob = m["yes_ask"]
        outcome = m["outcome"]

        # Check cache
        cache_key = f"scout_{market_id}"
        if cache_key in cache:
            raw_prob = cache[cache_key]["raw_prob"]
            reasoning = cache[cache_key]["reasoning"]
        else:
            log.info("[%d/%d] Scout: %s (cat=%s, implied=%.2f)",
                     i + 1, len(all_markets), market_id, category, implied_prob)

            result = predict_scout(market, category)
            if result is None:
                continue

            raw_prob, reasoning = result
            cache[cache_key] = {"raw_prob": raw_prob, "reasoning": reasoning}

            if (i + 1) % 10 == 0:
                save_cache(cache)
            time.sleep(0.3)

        # Calibrate scout prediction
        calibrated = _calibrate(raw_prob, implied_prob, category, False)

        pred = {
            "market_id": market_id,
            "category": category,
            "implied": implied_prob,
            "raw": raw_prob,
            "estimated": calibrated,
            "outcome": outcome,
        }
        scout_predictions.append(pred)

        # Simulate trade with scout
        trade = simulate_trade(calibrated, implied_prob, outcome, category, False)
        if trade:
            trade["category"] = category
            trade["market_id"] = market_id
            scout_trades.append(trade)

        # Judge pass
        if args.judge and trade:
            judge_cache_key = f"judge_{market_id}_{trade['action']}"
            if judge_cache_key in cache:
                approved = cache[judge_cache_key]["approved"]
                judged_prob = cache[judge_cache_key]["judged_prob"]
            else:
                side = "YES" if trade["action"] == "BUY_YES" else "NO"
                edge = trade["edge"]
                scout_probability = calibrated

                log.info("[%d/%d] Judge: %s (side=%s, edge=%.3f)",
                         i + 1, len(all_markets), market_id, side, edge)

                judge_result = predict_judge(market, category, side, edge, scout_probability)
                if judge_result is None:
                    approved, judged_prob = False, calibrated
                else:
                    approved, judged_prob, _ = judge_result

                cache[judge_cache_key] = {"approved": approved, "judged_prob": judged_prob}
                time.sleep(0.5)

            if approved:
                judge_calibrated = _calibrate(judged_prob, implied_prob, category, False, weights=JUDGE_SHRINKAGE_WEIGHTS)
                judge_pred = {
                    "market_id": market_id,
                    "category": category,
                    "implied": implied_prob,
                    "raw": judged_prob,
                    "estimated": judge_calibrated,
                    "outcome": outcome,
                }
                judge_predictions.append(judge_pred)

                judge_trade = simulate_trade(judge_calibrated, implied_prob, outcome, category, False)
                if judge_trade:
                    judge_trade["category"] = category
                    judge_trade["market_id"] = market_id
                    judge_trades.append(judge_trade)

    save_cache(cache)

    # Reports
    print_report(f"SCOUT ONLY ({OPENROUTER_SCOUT_MODEL})", scout_predictions, scout_trades)

    if args.judge:
        print_report(f"SCOUT + JUDGE ({OPENROUTER_JUDGE_MODEL})", judge_predictions, judge_trades)

        # Comparison
        if scout_trades and judge_trades:
            print(f"\n{'=' * 70}")
            print(f"  SCOUT vs JUDGE COMPARISON")
            print(f"{'=' * 70}")
            scout_ret = sum(t["return"] for t in scout_trades) / len(scout_trades)
            judge_ret = sum(t["return"] for t in judge_trades) / len(judge_trades)
            scout_wr = sum(1 for t in scout_trades if t["return"] > 0) / len(scout_trades)
            judge_wr = sum(1 for t in judge_trades if t["return"] > 0) / len(judge_trades)
            print(f"  Scout: {len(scout_trades)} trades, {scout_wr:.0%} win rate, {scout_ret:+.2%} avg return")
            print(f"  Judge: {len(judge_trades)} trades, {judge_wr:.0%} win rate, {judge_ret:+.2%} avg return")
            print(f"  Judge filtered out {len(scout_trades) - len(judge_trades)} trades ({(1 - len(judge_trades)/len(scout_trades))*100:.0f}% rejection rate)")
            print(f"{'=' * 70}")

    # Save full results
    results = {
        "config": {
            "scout_model": OPENROUTER_SCOUT_MODEL,
            "judge_model": OPENROUTER_JUDGE_MODEL if args.judge else None,
            "category_filter": args.category,
            "n_markets": len(all_markets),
            "n_scout_predictions": len(scout_predictions),
            "n_scout_trades": len(scout_trades),
            "n_judge_trades": len(judge_trades) if args.judge else 0,
        },
        "scout_brier": brier_score(scout_predictions),
        "market_brier": brier_score([{"estimated": p["implied"], "outcome": p["outcome"]} for p in scout_predictions]),
        "scout_predictions": scout_predictions,
        "scout_trades": scout_trades,
    }
    if args.judge:
        results["judge_brier"] = brier_score(judge_predictions) if judge_predictions else None
        results["judge_predictions"] = judge_predictions
        results["judge_trades"] = judge_trades

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration test using Prophet-Arena dataset")
    parser.add_argument("--judge", action="store_true", help="Enable judge model review")
    parser.add_argument("--category", type=str, help="Filter to category (sports, economics, weather, etc.)")
    parser.add_argument("--limit", type=int, help="Max number of markets to evaluate")
    parser.add_argument("--resume", action="store_true", help="Resume from cached predictions")
    args = parser.parse_args()
    run_calibration(args)
