"""Exceptions raised by the bot itself (not by CCXT)."""

from __future__ import annotations


class TradingBotError(Exception):
    """Base error for expected, handled failures."""


class LiveTradingNotConfirmedError(TradingBotError):
    """Live mode was requested without the explicit confirmation flag or keys."""


class InsufficientFundsError(TradingBotError):
    """Paper account cannot fund the order. Not retried."""


class InvalidBracketError(TradingBotError):
    """Stop-loss and take-profit are on the wrong side of the fill price."""
