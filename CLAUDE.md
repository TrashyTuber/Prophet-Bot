# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A prediction-market trading bot for the Prophet Arena platform. It fetches live market snapshots via `ai_prophet_core.ServerAPIClient`, runs a pluggable betting strategy on each market, and places orders (paper or live) through the `BettingEngine`.

## Running

```bash
# Activate the virtualenv (Python 3.11)
source venv/bin/activate

# Run the bot
python bot.py
```

Requires a `.env` with `PA_SERVER_API_KEY` and (optionally) `PA_SERVER_URL`. The Anthropic API key must also be set (`ANTHROPIC_API_KEY`) for the Claude-based strategy. For economics data, set `FRED_API_KEY` (free from https://fred.stlouisfed.org/docs/api/api_key.html).

## Architecture

- **bot.py** — Entry point. Runs the tick loop: claims ticks, loads candidate markets, runs strategy, submits trade intents.
- **strategy.py** — Market classifier + Claude-powered analysis. Filters to weather/economics markets only, enriches prompts with external data.
- **data/** — Data fetchers for external sources:
  - `weather.py` — Open-Meteo API (free, no key) for temperature/precip/wind forecasts
  - `economics.py` — FRED API for CPI, GDP, unemployment, Fed funds, etc.
- **ai_prophet_core** (installed package, v0.1.5) — Provides `ServerAPIClient`, `BenchmarkSession`, `TradeIntentRequest`, and related types. Not editable in this repo.

### Strategy Interface

Custom strategies subclass `ai_prophet_core.betting.BettingStrategy`:

```python
class MyStrategy(BettingStrategy):
    name = "my-strategy"

    def evaluate(self, market_id: str, p_yes: float, yes_ask: float, no_ask: float) -> BetSignal | None:
        ...
```

`evaluate()` returns a `BetSignal(side, shares, price, cost, metadata)` to trade, or `None` to skip. The engine sets `self.portfolio` (a `PortfolioSnapshot`) before each call.

### Key Types (from ai_prophet_core)

- `ServerAPIClient(base_url, api_key)` — HTTP client; `.get_market_snapshot()` returns a `MarketSnapshot`
- `BettingEngine(strategy, paper=True)` — orchestrates strategy evaluation and order placement
- `MarketSnapshot.markets` — list of `Market` objects with `.market_id`, `.quote.best_ask`, `.quote.best_bid`
- `BetSignal` — strategy output: `side` ("yes"/"no"), `shares`, `price`, `cost`, `metadata`

### ClaudeStrategy (bot.py)

Sends each market's prices to Claude (`claude-sonnet-4`) and parses the JSON response into a trade decision. Filters out HOLD actions and low-confidence (<0.6) signals. Position size scales linearly with confidence.
