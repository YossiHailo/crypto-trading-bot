"""Paper defaults and the two-flag live-trading gate."""

from __future__ import annotations

import pytest

from config.settings import assert_trading_mode
from core.errors import LiveTradingNotConfirmedError
from tests.conftest import make_settings


def test_defaults_are_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYMBOLS", "BTC/USDT, SOL/USDT")
    settings = make_settings()
    assert settings.paper_trading is True
    assert settings.live_trading_confirm is False
    assert settings.symbol_list() == ["BTC/USDT", "SOL/USDT"]
    assert assert_trading_mode(settings) == "paper"


def test_paper_wins_even_if_confirm_is_set() -> None:
    settings = make_settings(paper_trading=True, live_trading_confirm=True)
    assert assert_trading_mode(settings) == "paper"


def test_live_requires_confirm_flag_and_keys() -> None:
    unconfirmed = make_settings(paper_trading=False, live_trading_confirm=False)
    with pytest.raises(LiveTradingNotConfirmedError):
        assert_trading_mode(unconfirmed)
    missing_keys = make_settings(paper_trading=False, live_trading_confirm=True)
    with pytest.raises(LiveTradingNotConfirmedError):
        assert_trading_mode(missing_keys)
    ready = make_settings(
        paper_trading=False,
        live_trading_confirm=True,
        exchange_api_key="key",
        exchange_api_secret="secret",
    )
    assert assert_trading_mode(ready) == "live"


def test_false_string_parses_as_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "false")
    settings = make_settings()
    assert settings.paper_trading is False
    assert settings.live_trading_confirm is False


def test_exchange_id_is_limited() -> None:
    with pytest.raises(ValueError):
        make_settings(exchange_id="ftx")
