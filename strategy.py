import json
import logging

import anthropic
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.10
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

client = anthropic.Anthropic()


def analyze_market(market):
    yes_ask = float(market.quote.best_ask)
    no_ask = 1.0 - float(market.quote.best_bid)
    implied_prob = yes_ask

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a prediction market analyst. Estimate the probability "
                    f"that the following market resolves YES.\n\n"
                    f"Market: {market.question}\n"
                    f"Current YES price (implied probability): {implied_prob:.2f}\n\n"
                    f"Respond with ONLY a JSON object, no other text:\n"
                    f'{{"probability": <float between 0 and 1>, "reasoning": "one sentence"}}'
                ),
            }
        ],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    estimated_prob = float(result["probability"])
    reasoning = result.get("reasoning", "")

    log.info(
        "[STRATEGY] %s | implied=%.2f estimated=%.2f | %s",
        market.market_id, implied_prob, estimated_prob, reasoning,
    )

    if estimated_prob > implied_prob + EDGE_THRESHOLD:
        return ("BUY", "YES")
    elif estimated_prob < (1.0 - no_ask) - EDGE_THRESHOLD:
        return ("BUY", "NO")
    return None
