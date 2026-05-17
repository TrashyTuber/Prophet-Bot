"""One-shot test: claim a tick, dump all candidate markets, don't trade."""

import hashlib
import json
import os
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from ai_prophet_core import ServerAPIClient
from ai_prophet_core.arena import BenchmarkSession
from strategy import _classify_market

CONFIG = {"strategy": "claude-probability", "version": "1.0", "mode": "test-inspect"}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]

api = ServerAPIClient(
    base_url=os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev"),
    api_key=os.environ["PA_SERVER_API_KEY"],
    timeout=30,
)

print("API health:", api.health_check().status)

with BenchmarkSession(api) as session:
    session.create_experiment(
        slug="test-inspect-markets-v1",
        config_hash=CONFIG_HASH,
        config_json=CONFIG,
        n_ticks=2,
    )
    participant = session.upsert_participant(
        model="custom:claude-bot-inspect",
        starting_cash=10_000,
    )

    print("\nClaiming tick...")
    lease = session.claim_tick()
    if not lease.available:
        print(f"No tick available: {lease.reason} (retry in {lease.retry_after_sec}s)")
        exit(1)

    tick = session.load_candidates(lease)
    lease = tick.lease
    markets = tick.candidates.markets

    print(f"\n{'='*80}")
    print(f"GOT {len(markets)} CANDIDATE MARKETS")
    print(f"{'='*80}\n")

    categories = {}
    for m in markets:
        cat = _classify_market(m)
        categories.setdefault(cat, []).append(m)

    for cat in sorted(categories.keys()):
        cat_markets = categories[cat]
        print(f"\n--- {cat.upper()} ({len(cat_markets)} markets) ---")
        for m in cat_markets:
            question = getattr(m, "question", "???")
            ask = float(m.quote.best_ask)
            bid = float(m.quote.best_bid)
            res_time = getattr(m, "resolution_time", "???")
            family = getattr(m, "family", None) or "no-family"
            vol = getattr(m.quote, "volume_24h", 0)
            print(f"  [{ask:.2f}/{bid:.2f}] {question}")
            print(f"           resolves: {res_time}  family: {family}  vol: {vol}")

    print(f"\n{'='*80}")
    print("SUMMARY:")
    for cat in sorted(categories.keys()):
        print(f"  {cat}: {len(categories[cat])} markets")
    print(f"  TOTAL: {len(markets)}")
    print(f"{'='*80}")

    # Don't trade, just finalize
    session.put_plan(lease, participant.participant_idx, CONFIG)
    session.finalize(lease, participant.participant_idx)
    session.complete_tick(lease)
    print("\nTick completed (no trades placed).")
