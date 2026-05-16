"""Prophet Arena trading bot with Claude-powered strategy."""

import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(override=True)

from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession

from strategy import analyze_market, _classify_market
from data.sports import _extract_teams, _detect_sport

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG = {"strategy": "claude-probability", "version": "1.0"}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:16]

STARTING_CASH = 10_000
KELLY_FRACTION = 0.25
MAX_CASH_PCT_PER_TRADE = 0.05
MAX_NOTIONAL_PER_MARKET = 1_000
MAX_TOTAL_DEPLOYED_PCT = 0.70
MAX_EXISTING_POSITION_PCT = 0.10
MAX_INTENTS_PER_TICK = 10
MAX_TEAM_EXPOSURE_PCT = 0.10
MAX_GAME_EXPOSURE_PCT = 0.08
MAX_LEAGUE_EXPOSURE_PCT = 0.30
STOP_LOSS_PCT = -0.30
TAKE_PROFIT_PCT = 0.20
TRADES_CSV = "trades.csv"
TRADES_FIELDS = [
    "timestamp", "market_id", "market_type", "action", "side", "shares",
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


def _get_correlation_groups(market):
    """Extract correlation groups for a sports market: teams, game key, league."""
    question = market.question
    teams = _extract_teams(question)
    sport_info = _detect_sport(question)
    league = sport_info["label"] if sport_info else None

    if len(teams) >= 2:
        game_key = "-vs-".join(sorted(teams[:2]))
    elif teams:
        game_key = teams[0]
    else:
        game_key = None

    return {"teams": teams, "game": game_key, "league": league}


class CorrelationTracker:
    """Tracks cumulative exposure by team, game, and league within a tick."""

    def __init__(self, equity, existing_positions, markets_by_id):
        self.equity = equity
        self.team_exposure = {}
        self.game_exposure = {}
        self.league_exposure = {}

        for market_id, pos in existing_positions.items():
            market = markets_by_id.get(market_id)
            if not market or _classify_market(market) != "sports":
                continue
            mv = float(pos.shares) * float(pos.current_price)
            groups = _get_correlation_groups(market)
            for team in groups["teams"]:
                self.team_exposure[team] = self.team_exposure.get(team, 0.0) + mv
            if groups["game"]:
                self.game_exposure[groups["game"]] = self.game_exposure.get(groups["game"], 0.0) + mv
            if groups["league"]:
                self.league_exposure[groups["league"]] = self.league_exposure.get(groups["league"], 0.0) + mv

    def check(self, market, proposed_notional):
        """Return (allowed, reason) for a proposed sports trade."""
        if _classify_market(market) != "sports":
            return True, ""

        groups = _get_correlation_groups(market)

        for team in groups["teams"]:
            current = self.team_exposure.get(team, 0.0)
            if (current + proposed_notional) / self.equity > MAX_TEAM_EXPOSURE_PCT:
                return False, f"team '{team}' exposure would exceed {MAX_TEAM_EXPOSURE_PCT*100:.0f}%"

        if groups["game"]:
            current = self.game_exposure.get(groups["game"], 0.0)
            if (current + proposed_notional) / self.equity > MAX_GAME_EXPOSURE_PCT:
                return False, f"game '{groups['game']}' exposure would exceed {MAX_GAME_EXPOSURE_PCT*100:.0f}%"

        if groups["league"]:
            current = self.league_exposure.get(groups["league"], 0.0)
            if (current + proposed_notional) / self.equity > MAX_LEAGUE_EXPOSURE_PCT:
                return False, f"league '{groups['league']}' exposure would exceed {MAX_LEAGUE_EXPOSURE_PCT*100:.0f}%"

        return True, ""

    def record(self, market, notional):
        """Record a trade that passed checks."""
        if _classify_market(market) != "sports":
            return
        groups = _get_correlation_groups(market)
        for team in groups["teams"]:
            self.team_exposure[team] = self.team_exposure.get(team, 0.0) + notional
        if groups["game"]:
            self.game_exposure[groups["game"]] = self.game_exposure.get(groups["game"], 0.0) + notional
        if groups["league"]:
            self.league_exposure[groups["league"]] = self.league_exposure.get(groups["league"], 0.0) + notional


def run():
    api = ServerAPIClient(
        base_url=os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev"),
        api_key=os.environ["PA_SERVER_API_KEY"],
        timeout=30,
    )

    with BenchmarkSession(api) as session:
        session.create_experiment(
            slug="claude-bot-v8",
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

            markets_by_id = {m.market_id: m for m in markets}
            corr_tracker = CorrelationTracker(equity, positions_by_market, markets_by_id)

            log.info("Tick claimed — %d markets, cash=%.2f, equity=%.2f, deployed=%.0f%%",
                     len(markets), available_cash, equity, total_deployed / equity * 100)

            intents = []
            trade_records = []
            buy_candidates = []
            for market in markets:
                existing_pos = positions_by_market.get(market.market_id)

                if existing_pos and float(existing_pos.shares) > 0:
                    entry_cost = float(existing_pos.shares) * float(existing_pos.avg_entry_price)
                    if entry_cost > 0:
                        pnl_pct = float(existing_pos.unrealized_pnl) / entry_cost
                        if pnl_pct <= STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT:
                            sell_shares = int(float(existing_pos.shares))
                            reason = "STOP-LOSS" if pnl_pct < 0 else "TAKE-PROFIT"
                            log.info("  %s SELL %s %d shares on %s (pnl=%.0f%%)",
                                     reason, existing_pos.side, sell_shares, market.market_id, pnl_pct * 100)
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
                                "market_type": _classify_market(market),
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

                if total_deployed / equity >= MAX_TOTAL_DEPLOYED_PCT:
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
                            "market_type": _classify_market(market),
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

                family = getattr(market, "family", None) or market.market_id
                buy_candidates.append((market, action, side, edge, shares, family))

            # Deduplicate: keep only the best edge per family
            best_by_family = {}
            for candidate in buy_candidates:
                market, action, side, edge, shares, family = candidate
                if family not in best_by_family or edge > best_by_family[family][3]:
                    if family in best_by_family:
                        log.info("  FAMILY SKIP %s — replaced by %s (edge=%.2f > %.2f)",
                                 best_by_family[family][0].market_id, market.market_id, edge, best_by_family[family][3])
                    best_by_family[family] = candidate
                else:
                    log.info("  FAMILY SKIP %s — better edge in %s", market.market_id, best_by_family[family][0].market_id)

            for market, action, side, edge, shares, family in sorted(best_by_family.values(), key=lambda x: -x[3]):
                cost = float(market.quote.best_ask) if side == "YES" else 1.0 - float(market.quote.best_bid)
                proposed_notional = shares * cost
                existing_mv = position_value.get(market.market_id, 0.0)

                if (total_deployed + proposed_notional) / equity > MAX_TOTAL_DEPLOYED_PCT:
                    log.info("  RISK SKIP %s — total deployment would exceed %.0f%%", market.market_id, MAX_TOTAL_DEPLOYED_PCT * 100)
                    continue

                if (existing_mv + proposed_notional) > MAX_NOTIONAL_PER_MARKET:
                    log.info("  RISK SKIP %s — would exceed $%d per-market limit", market.market_id, MAX_NOTIONAL_PER_MARKET)
                    continue

                if existing_mv / equity > MAX_EXISTING_POSITION_PCT:
                    log.info("  RISK SKIP %s — existing position already %.0f%% of equity", market.market_id, existing_mv / equity * 100)
                    continue

                corr_ok, corr_reason = corr_tracker.check(market, proposed_notional)
                if not corr_ok:
                    log.info("  CORR SKIP %s — %s", market.market_id, corr_reason)
                    continue

                if len(intents) >= MAX_INTENTS_PER_TICK:
                    log.info("  RISK SKIP %s — already at %d intents (max per tick)", market.market_id, MAX_INTENTS_PER_TICK)
                    break

                total_deployed += proposed_notional
                corr_tracker.record(market, proposed_notional)

                implied_prob = float(market.quote.best_ask)
                if side == "YES":
                    estimated_prob = implied_prob + edge
                else:
                    estimated_prob = float(market.quote.best_bid) - edge

                log.info("  TRADE %s %s %d shares on %s (edge=%.2f, family=%s)", action, side, shares, market.market_id, edge, family)
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
                    "market_type": _classify_market(market),
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
