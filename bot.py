"""Prophet Arena trading bot with Claude-powered strategy."""

import hashlib
import json
import logging
import os
import time

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

SHARES_PER_TRADE = "10"


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
            starting_cash=10_000,
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

            log.info("Tick claimed — %d candidate markets", len(markets))

            intents = []
            for market in markets:
                try:
                    decision = analyze_market(market)
                except Exception:
                    log.exception("Strategy error on %s", market.market_id)
                    continue

                if decision is None:
                    continue

                action, side = decision
                log.info("  TRADE %s %s %s shares on %s", action, side, SHARES_PER_TRADE, market.market_id)
                intents.append(TradeIntentRequest(
                    market_id=market.market_id,
                    action=action,
                    side=side,
                    shares=SHARES_PER_TRADE,
                    idempotency_key="",
                ))

            session.put_plan(lease, participant.participant_idx, CONFIG)

            if intents:
                result = session.submit_intents(lease, participant.participant_idx, intents)
                log.info("Submitted %d intents: %d accepted, %d rejected", len(intents), result.accepted, result.rejected)
            else:
                log.info("No trades this tick.")

            session.finalize(lease, participant.participant_idx)
            session.complete_tick(lease)


if __name__ == "__main__":
    run()
