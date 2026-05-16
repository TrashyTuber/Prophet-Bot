"""Prophet Arena trading bot with Claude-powered strategy."""

import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession

from strategy import analyze_market

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG = {"strategy": "claude-probability", "version": "1.0"}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:16]

STARTING_CASH = 10_000
KELLY_FRACTION = 0.25
MAX_CASH_PCT_PER_TRADE = 0.05
MAX_SINGLE_MARKET_PCT = 0.15
MAX_TOTAL_DEPLOYED_PCT = 0.70
MAX_EXISTING_POSITION_PCT = 0.10
STOP_LOSS_PCT = -0.30
TRADES_CSV = "trades.csv"
TRADES_FIELDS = [
    "timestamp", "market_id", "action", "side", "shares",
    "edge", "implied_prob", "estimated_prob",
    "status", "fill_price", "notional",
]


def compute_shares(edge: float, side: str, market, available_cash: float) -> int:
    if side == "YES":
        cost = float(market.quote.best_ask)
    else:
        cost = 1.0 - float(market.quote.best_bid)

    if cost <= 0 or cost >= 1 or edge <= 0 or available_cash <= 0:
        return 0

    kelly_fraction = KELLY_FRACTION * edge / (1.0 - cost)
    cash_fraction = min(kelly_fraction, MAX_CASH_PCT_PER_TRADE)
    return int(available_cash * cash_fraction / cost)


def _log_trades(records):
    if not records:
        return
    write_header = not os.path.exists(TRADES_CSV) or os.path.getsize(TRADES_CSV) == 0
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def run():
    api = ServerAPIClient(
        base_url=os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev"),
        api_key=os.environ["PA_SERVER_API_KEY"],
        timeout=30,
    )

    with BenchmarkSession(api) as session:
        session.create_experiment(
            slug="claude-bot-v1",
            config_hash=CONFIG_HASH,
            config_json=CONFIG,
            n_ticks=96,
        )
        participant = session.upsert_participant(
            model="custom:claude-bot",
            starting_cash=STARTING_CASH,
        )

        log.info("Experiment created, entering tick loop...")

        while True:
            lease = session.claim_tick()
            if not lease.available:
                if lease.reason == "experiment_completed":
                    log.info("Experiment completed.")
                    break
                log.info("No tick available (%s), retrying in %ds...", lease.reason, lease.retry_after_sec or 15)
                time.sleep(lease.retry_after_sec or 15)
                continue

            tick = session.load_candidates(lease)
            lease = tick.lease
            markets = tick.candidates.markets

            portfolio = session.get_portfolio(participant.participant_idx)
            available_cash = float(portfolio.cash) if portfolio else float(STARTING_CASH)
            equity = float(portfolio.equity) if portfolio else float(STARTING_CASH)
            equity = max(equity, 1.0)

            position_value = {}
            positions_by_market = {}
            total_deployed = 0.0
            if portfolio:
                for pos in portfolio.positions:
                    mv = float(pos.shares) * float(pos.current_price)
                    position_value[pos.market_id] = mv
                    positions_by_market[pos.market_id] = pos
                    total_deployed += mv

            log.info("Tick claimed — %d markets, cash=%.2f, equity=%.2f, deployed=%.0f%%",
                     len(markets), available_cash, equity, total_deployed / equity * 100)

            intents = []
            trade_records = []
            for market in markets:
                existing_pos = positions_by_market.get(market.market_id)

                if existing_pos and float(existing_pos.shares) > 0:
                    entry_cost = float(existing_pos.shares) * float(existing_pos.avg_entry_price)
                    if entry_cost > 0:
                        pnl_pct = float(existing_pos.unrealized_pnl) / entry_cost
                        if pnl_pct <= STOP_LOSS_PCT:
                            sell_shares = int(float(existing_pos.shares))
                            log.info("  STOP-LOSS SELL %s %d shares on %s (pnl=%.0f%%)",
                                     existing_pos.side, sell_shares, market.market_id, pnl_pct * 100)
                            intents.append(TradeIntentRequest(
                                market_id=market.market_id,
                                action="SELL",
                                side=existing_pos.side,
                                shares=str(sell_shares),
                                idempotency_key="",
                            ))
                            trade_records.append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "market_id": market.market_id,
                                "action": "SELL",
                                "side": existing_pos.side,
                                "shares": sell_shares,
                                "edge": round(pnl_pct, 4),
                                "implied_prob": round(float(market.quote.best_ask), 4),
                                "estimated_prob": round(float(market.quote.best_ask), 4),
                                "status": "pending",
                                "fill_price": "",
                                "notional": "",
                            })
                            continue

                try:
                    decision = analyze_market(market)
                except Exception:
                    log.exception("Strategy error on %s", market.market_id)
                    continue

                if existing_pos and float(existing_pos.shares) > 0 and decision is not None:
                    _, suggested_side, flip_edge = decision
                    if suggested_side != existing_pos.side:
                        sell_shares = int(float(existing_pos.shares))
                        log.info("  FLIP SELL %s %d shares on %s (now favoring %s, edge=%.2f)",
                                 existing_pos.side, sell_shares, market.market_id, suggested_side, flip_edge)
                        intents.append(TradeIntentRequest(
                            market_id=market.market_id,
                            action="SELL",
                            side=existing_pos.side,
                            shares=str(sell_shares),
                            idempotency_key="",
                        ))
                        implied_prob = float(market.quote.best_ask)
                        trade_records.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "market_id": market.market_id,
                            "action": "SELL",
                            "side": existing_pos.side,
                            "shares": sell_shares,
                            "edge": round(flip_edge, 4),
                            "implied_prob": round(implied_prob, 4),
                            "estimated_prob": round(implied_prob + flip_edge if suggested_side == "YES" else float(market.quote.best_bid) - flip_edge, 4),
                            "status": "pending",
                            "fill_price": "",
                            "notional": "",
                        })
                        continue

                if decision is None:
                    continue

                action, side, edge = decision
                shares = compute_shares(edge, side, market, available_cash)
                if shares <= 0:
                    log.info("  SKIP %s %s on %s (edge=%.2f) — sized to 0 shares", action, side, market.market_id, edge)
                    continue

                cost = float(market.quote.best_ask) if side == "YES" else 1.0 - float(market.quote.best_bid)
                proposed_notional = shares * cost
                existing_mv = position_value.get(market.market_id, 0.0)

                if (total_deployed + proposed_notional) / equity > MAX_TOTAL_DEPLOYED_PCT:
                    log.info("  RISK SKIP %s — total deployment would exceed %.0f%%", market.market_id, MAX_TOTAL_DEPLOYED_PCT * 100)
                    continue

                if (existing_mv + proposed_notional) / equity > MAX_SINGLE_MARKET_PCT:
                    log.info("  RISK SKIP %s — single market exposure would exceed %.0f%%", market.market_id, MAX_SINGLE_MARKET_PCT * 100)
                    continue

                if existing_mv / equity > MAX_EXISTING_POSITION_PCT:
                    log.info("  RISK SKIP %s — existing position already %.0f%% of equity", market.market_id, existing_mv / equity * 100)
                    continue

                total_deployed += proposed_notional

                implied_prob = float(market.quote.best_ask)
                if side == "YES":
                    estimated_prob = implied_prob + edge
                else:
                    estimated_prob = float(market.quote.best_bid) - edge

                log.info("  TRADE %s %s %d shares on %s (edge=%.2f)", action, side, shares, market.market_id, edge)
                intents.append(TradeIntentRequest(
                    market_id=market.market_id,
                    action=action,
                    side=side,
                    shares=str(shares),
                    idempotency_key="",
                ))
                trade_records.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market_id": market.market_id,
                    "action": action,
                    "side": side,
                    "shares": shares,
                    "edge": round(edge, 4),
                    "implied_prob": round(implied_prob, 4),
                    "estimated_prob": round(estimated_prob, 4),
                    "status": "pending",
                    "fill_price": "",
                    "notional": "",
                })

            session.put_plan(lease, participant.participant_idx, CONFIG)

            if intents:
                result = session.submit_intents(lease, participant.participant_idx, intents)
                filled = {f.market_id: f for f in result.fills}
                for record in trade_records:
                    fill = filled.get(record["market_id"])
                    if fill:
                        record["status"] = "filled"
                        record["fill_price"] = fill.price
                        record["notional"] = fill.notional
                    else:
                        record["status"] = "rejected"
                _log_trades(trade_records)
                log.info("Submitted %d intents: %d accepted, %d rejected", len(intents), result.accepted, result.rejected)
            else:
                log.info("No trades this tick.")

            session.finalize(lease, participant.participant_idx)
            session.complete_tick(lease)


if __name__ == "__main__":
    run()
