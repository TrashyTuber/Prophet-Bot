"""Analyze trades.csv to report performance metrics."""

import csv
import os
import sys
from collections import defaultdict


TRADES_CSV = os.environ.get("TRADES_CSV", "trades.csv")


def load_trades(path):
    if not os.path.exists(path):
        print(f"No trades file found at {path}")
        sys.exit(1)

    trades = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["shares"] = int(row["shares"])
            row["edge"] = float(row["edge"])
            row["implied_prob"] = float(row["implied_prob"])
            row["estimated_prob"] = float(row["estimated_prob"])
            row["fill_price"] = float(row["fill_price"]) if row["fill_price"] else None
            row["notional"] = float(row["notional"]) if row["notional"] else None
            trades.append(row)
    return trades


def get_market_type(trade):
    if "market_type" in trade and trade["market_type"]:
        return trade["market_type"]
    mid = trade["market_id"].lower()
    weather_hints = ["temp", "weather", "rain", "snow", "heat", "cold", "wind", "storm", "hurricane"]
    econ_hints = ["cpi", "gdp", "inflation", "unemployment", "jobs", "payroll", "fed",
                  "treasury", "yield", "bond", "interest", "retail", "housing"]
    for hint in weather_hints:
        if hint in mid:
            return "weather"
    for hint in econ_hints:
        if hint in mid:
            return "economics"
    return "general"


def report(trades):
    if not trades:
        print("No trades to analyze.")
        return

    filled = [t for t in trades if t["status"] == "filled"]
    rejected = [t for t in trades if t["status"] == "rejected"]
    buys = [t for t in filled if t["action"] == "BUY"]
    sells = [t for t in filled if t["action"] == "SELL"]

    print("=" * 60)
    print("TRADE ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n--- Overview ---")
    print(f"Total intents:    {len(trades)}")
    print(f"Filled:           {len(filled)} ({len(filled)/len(trades)*100:.0f}%)")
    print(f"Rejected:         {len(rejected)} ({len(rejected)/len(trades)*100:.0f}%)")
    print(f"Buys (filled):    {len(buys)}")
    print(f"Sells (filled):   {len(sells)}")

    if filled:
        total_notional = sum(t["notional"] for t in filled if t["notional"])
        edges = [t["edge"] for t in filled]
        edges_sorted = sorted(edges)
        avg_edge = sum(edges) / len(edges)
        median_edge = edges_sorted[len(edges_sorted) // 2]

        print(f"\n--- Filled Trades ---")
        print(f"Total notional:   ${total_notional:,.2f}")
        print(f"Avg edge:         {avg_edge:.4f} ({avg_edge*100:.2f}%)")
        print(f"Median edge:      {median_edge:.4f} ({median_edge*100:.2f}%)")
        print(f"Min edge:         {edges_sorted[0]:.4f}")
        print(f"Max edge:         {edges_sorted[-1]:.4f}")
        print(f"Avg shares:       {sum(t['shares'] for t in filled) / len(filled):.0f}")

        expected_profit = sum(t["edge"] * t["notional"] for t in filled if t["notional"])
        print(f"Expected profit:  ${expected_profit:,.2f} (if perfectly calibrated)")

    # By side
    yes_trades = [t for t in filled if t["side"] == "YES"]
    no_trades = [t for t in filled if t["side"] == "NO"]
    print(f"\n--- By Side ---")
    print(f"{'Side':<6} {'Count':>6} {'Avg Edge':>10} {'Notional':>12}")
    print(f"{'-'*6} {'-'*6} {'-'*10} {'-'*12}")
    for label, group in [("YES", yes_trades), ("NO", no_trades)]:
        if group:
            avg_e = sum(t["edge"] for t in group) / len(group)
            total_n = sum(t["notional"] for t in group if t["notional"])
            print(f"{label:<6} {len(group):>6} {avg_e:>10.4f} ${total_n:>11,.2f}")

    # By inferred market type
    by_type = defaultdict(list)
    for t in filled:
        by_type[get_market_type(t)].append(t)

    print(f"\n--- By Market Type ---")
    print(f"{'Type':<12} {'Count':>6} {'Avg Edge':>10} {'Notional':>12} {'Est Profit':>12}")
    print(f"{'-'*12} {'-'*6} {'-'*10} {'-'*12} {'-'*12}")
    for mtype in ["weather", "economics", "general"]:
        group = by_type.get(mtype, [])
        if group:
            avg_e = sum(t["edge"] for t in group) / len(group)
            total_n = sum(t["notional"] for t in group if t["notional"])
            est_profit = sum(t["edge"] * t["notional"] for t in group if t["notional"])
            print(f"{mtype:<12} {len(group):>6} {avg_e:>10.4f} ${total_n:>11,.2f} ${est_profit:>11,.2f}")

    # Top trades by notional
    if filled:
        top = sorted(filled, key=lambda t: t["notional"] or 0, reverse=True)[:10]
        print(f"\n--- Top 10 Trades by Notional ---")
        print(f"{'Market ID':<40} {'Side':<5} {'Shares':>7} {'Edge':>7} {'Notional':>10}")
        print(f"{'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*10}")
        for t in top:
            mid = t["market_id"][:40]
            print(f"{mid:<40} {t['side']:<5} {t['shares']:>7} {t['edge']:>7.4f} ${t['notional']:>9,.2f}")

    # Sells breakdown (stop-loss vs flip)
    if sells:
        stop_losses = [t for t in sells if t["edge"] < 0]
        flips = [t for t in sells if t["edge"] >= 0]
        print(f"\n--- Sell Breakdown ---")
        print(f"Stop-loss sells:  {len(stop_losses)}")
        print(f"Flip sells:       {len(flips)}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else TRADES_CSV
    trades = load_trades(path)
    report(trades)
