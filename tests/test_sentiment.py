"""Lexicon scoring and graceful degradation when sources are down."""

from __future__ import annotations

import json

import pytest

from agents.sentiment import (
    SentimentAgent,
    build_sentiment,
    parse_cryptopanic,
    parse_fear_greed,
    parse_rss,
    score_fear_greed,
    score_headlines,
)
from config.settings import Settings
from core.models import SentimentBias


def test_fear_greed_is_centered_on_fifty() -> None:
    assert score_fear_greed(50) == pytest.approx(0.0)
    assert score_fear_greed(0) == pytest.approx(-1.0)
    assert score_fear_greed(100) == pytest.approx(1.0)
    assert score_fear_greed(25) == pytest.approx(-0.5)


def test_headline_scores_shrink_thin_evidence() -> None:
    one_crash, bulls, bears = score_headlines(["Bitcoin crash hits the market"])
    assert bulls == 0
    assert bears >= 1
    assert one_crash == pytest.approx(-0.25)
    four, _, _ = score_headlines(["crash plunge hack ban"])
    assert four == pytest.approx(-1.0)
    assert score_headlines(["Developers published the weekly notes"])[0] == 0.0


def test_word_boundaries_skip_partial_matches() -> None:
    score, bulls, bears = score_headlines(["The bank announced a new savings product"])
    assert bulls == 0
    assert bears == 0
    assert score == 0.0


def test_parse_sources() -> None:
    value, label = parse_fear_greed({"data": [{"value": "24", "value_classification": "Extreme Fear"}]})
    assert value == 24
    assert label == "Extreme Fear"
    titles = parse_cryptopanic({"results": [{"title": "ETF approval"}, {"title": ""}]})
    assert titles == ["ETF approval"]
    xml = """
    <rss xmlns="http://purl.org/rss/1.0/">
      <channel><item><title>Market crash deepens</title></item></channel>
    </rss>
    """
    assert parse_rss(xml) == ["Market crash deepens"]


def test_all_sources_down_is_neutral() -> None:
    signal = build_sentiment(
        fear_greed=None,
        fear_greed_label_text=None,
        headlines=None,
        errors=["fear_greed: timeout", "headlines: timeout"],
    )
    assert signal.bias == SentimentBias.NEUTRAL
    assert signal.score == 0.0
    assert "NEUTRAL" in signal.reasoning
    assert signal.sources_used == ()


def test_average_of_fear_and_headlines() -> None:
    signal = build_sentiment(
        fear_greed=0,
        fear_greed_label_text="Extreme Fear",
        headlines=["quiet day on the desk"],
        errors=[],
    )
    # Fear & Greed maps to -1, headlines have no hits and map to 0. Average is -0.5.
    assert signal.score == pytest.approx(-0.5)
    assert signal.bias == SentimentBias.BEARISH
    assert signal.sources_used == ("fear_greed", "headlines")


@pytest.mark.asyncio
async def test_agent_evaluate_uses_injected_http_and_survives_failures() -> None:
    settings = Settings(_env_file=None, cryptopanic_token="")

    async def broken(_url: str) -> str:
        raise TimeoutError("down")

    agent = SentimentAgent(settings, http_get=broken)
    signal = await agent.evaluate()
    assert signal.bias == SentimentBias.NEUTRAL
    assert signal.score == 0.0

    payload = {"data": [{"value": "80", "value_classification": "Extreme Greed"}]}
    rss = "<rss><channel><item><title>Bitcoin rally continues</title></item></channel></rss>"

    async def ok(url: str) -> str:
        if "alternative.me" in url:
            return json.dumps(payload)
        return rss

    agent = SentimentAgent(settings, http_get=ok)
    signal = await agent.evaluate()
    assert "fear_greed" in signal.sources_used
    assert "headlines" in signal.sources_used
    assert signal.fear_greed == 80
    assert signal.score > 0
