"""Risk management agent. This is the only place a signal becomes an order.

Position size uses fixed-fractional risk:

    risk_budget   = equity * risk_per_trade          # default 2%
    stop_distance = ATR * atr_multiplier             # or price * fixed_stop_pct
    size          = risk_budget / stop_distance

If that stop fills at the planned price, the loss is about ``risk_per_trade``
of equity before fees. Take-profit is ``stop_distance * reward_risk_ratio``
away from the entry. Size is then capped three ways:

* by cash (and leverage on swap),
* by ``max_notional_pct`` per position (default 50% of equity), so one
  tight stop cannot turn into an all-in trade, and
* by ``max_total_exposure_pct`` across all open positions, because BTC and
  ETH move together and two "independent" 50% positions are one bet.

When a cap binds, the real risk is lower than ``risk_per_trade``. The
decision text reports the *effective* risk so this is never hidden.

Reward:risk is checked **after** round-trip fees and slippage. With a 1%
stop and 0.1% fees per side, a nominal 2R trade is really about 1.5R.

Exits are decided before entry gates. A pause, the max-drawdown kill switch,
a daily loss limit, a cooldown, a confidence floor, or a sentiment veto can
block a new trade. None of them block closing a position the bot already holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from config.settings import Settings
from core.models import (
    PortfolioView,
    Position,
    RiskAction,
    RiskDecision,
    SentimentSignal,
    SignalAction,
    TechnicalSignal,
    utcnow,
)

# Neutral sentiment leaves technical confidence unchanged. Full agreement
# multiplies it by 1.15. Full disagreement multiplies it by 0.60.
_AGREEMENT_BOOST = 0.15
_DISAGREEMENT_PENALTY = 0.40
# Hard veto when direction * sentiment_score is at or below this value.
_SENTIMENT_VETO = 0.50


@dataclass(frozen=True)
class _Sizing:
    size: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_risk: float
    stop_distance: float
    notes: str


class RiskManager:
    """Combine a technical signal, sentiment, and the account into one decision."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(
        self,
        technical: TechnicalSignal,
        sentiment: SentimentSignal,
        portfolio: PortfolioView,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or utcnow()
        position = portfolio.positions.get(technical.symbol)

        if position is not None and technical.wants_exit(position.side):
            closing = "long" if position.side == "long" else "short"
            return self._exit(position, technical, f"Exit signal closes the open {closing}.")

        if technical.action == SignalAction.HOLD:
            return self._no(
                technical,
                "Technical signal is HOLD.",
                confidence=technical.confidence,
                now=now,
            )
        if position is not None:
            return self._no(
                technical,
                f"A {position.side} position is already open.",
                confidence=technical.confidence,
                now=now,
            )
        if technical.symbol in portfolio.pending_symbols:
            return self._no(technical, "A limit order is already resting for this symbol.", now=now)

        opening_short = technical.action == SignalAction.SELL
        if opening_short and not self.settings.allow_short:
            return self._no(
                technical,
                "Sell signal with no long to close, and shorting is disabled.",
                confidence=technical.confidence,
                now=now,
            )
        if opening_short and not self.settings.paper_trading and self.settings.market_type == "spot":
            return self._no(
                technical,
                "Live short entries require MARKET_TYPE=swap. Spot mode only closes longs.",
                confidence=technical.confidence,
                now=now,
            )

        if portfolio.paused:
            return self._no(
                technical,
                "New entries are paused by the operator.",
                confidence=technical.confidence,
                now=now,
            )
        if portfolio.halted_reason:
            return self._no(
                technical,
                f"Trading halted: {portfolio.halted_reason} Reset it from the dashboard after review.",
                confidence=technical.confidence,
                now=now,
            )

        if not portfolio.balance_synced:
            return self._no(
                technical,
                "Exchange balance has not synced yet, so new entries are blocked.",
                confidence=technical.confidence,
                now=now,
            )

        if portfolio.day_start_equity > 0:
            drawdown = (portfolio.day_start_equity - portfolio.equity) / portfolio.day_start_equity
            if drawdown >= self.settings.daily_max_loss_pct - 1e-12:
                return self._no(
                    technical,
                    f"Daily max-loss circuit breaker is active "
                    f"(equity drawdown {drawdown:.2%} vs limit {self.settings.daily_max_loss_pct:.2%}).",
                    confidence=technical.confidence,
                    now=now,
                )

        expiry = portfolio.cooldowns.get(technical.symbol)
        if expiry is not None and expiry > now:
            return self._no(
                technical,
                f"Cooldown after a stop-out lasts until {expiry.isoformat()}.",
                confidence=technical.confidence,
                now=now,
            )

        occupied = set(portfolio.positions) | set(portfolio.pending_symbols)
        if technical.symbol not in occupied and len(occupied) >= self.settings.max_open_positions:
            return self._no(
                technical,
                f"Max open positions ({self.settings.max_open_positions}) is reached.",
                confidence=technical.confidence,
                now=now,
            )

        alignment = _direction(technical.action) * sentiment.score
        if alignment <= -_SENTIMENT_VETO:
            return self._no(
                technical,
                f"Sentiment veto. {sentiment.bias.value} score {sentiment.score:+.2f} "
                f"strongly disagrees with {technical.action.value}.",
                confidence=technical.confidence,
                now=now,
            )

        combined, alignment_note = _combined_confidence(technical, sentiment)
        if combined < self.settings.min_confidence:
            return self._no(
                technical,
                f"Combined confidence {combined:.2f} is below the minimum "
                f"{self.settings.min_confidence:.2f}. {alignment_note}.",
                confidence=combined,
                now=now,
            )

        sizing = self._size(technical, portfolio)
        if sizing is None:
            return self._no(
                technical,
                "Position size is zero, below the minimum notional, the stop distance "
                "is invalid, the exposure cap is full, or the bracket would cross zero. "
                "No order sent.",
                confidence=combined,
                now=now,
            )
        if sizing.reward_risk + 1e-9 < self.settings.min_reward_risk:
            return self._no(
                technical,
                f"Reward:risk {sizing.reward_risk:.2f} after fees and slippage is below "
                f"the minimum {self.settings.min_reward_risk:.2f}.",
                confidence=combined,
                now=now,
            )

        action = RiskAction.EXECUTE_BUY if technical.action == SignalAction.BUY else RiskAction.EXECUTE_SELL
        side_word = "long" if action == RiskAction.EXECUTE_BUY else "short"
        reasoning = (
            f"Execute {side_word}. Size {sizing.size:.8g} {technical.symbol} at {sizing.entry:.8g}, "
            f"stop {sizing.stop_loss:.8g}, target {sizing.take_profit:.8g}, "
            f"Net R:R {sizing.reward_risk:.2f}. Risk at stop {sizing.risk_amount:.4f} "
            f"{portfolio.quote_currency} = {sizing.risk_amount / portfolio.equity:.2%} of equity "
            f"(target {self.settings.risk_per_trade:.2%}; {sizing.stop_distance:.8g} stop distance). "
            f"{sizing.notes} Combined confidence {combined:.2f}. {alignment_note}. "
            f"Sentiment: {sentiment.reasoning} Technical: {technical.reasoning}"
        )
        return RiskDecision(
            action=action,
            symbol=technical.symbol,
            position_size=sizing.size,
            entry=sizing.entry,
            stop_loss=sizing.stop_loss,
            take_profit=sizing.take_profit,
            confidence=combined,
            reasoning=reasoning,
            risk_amount=sizing.risk_amount,
            reward_risk=sizing.reward_risk,
            timestamp=now,
        )

    def _exit(self, position: Position, technical: TechnicalSignal, reason: str) -> RiskDecision:
        side = position.side
        size = float(position.size)
        action = RiskAction.EXECUTE_SELL if side == "long" else RiskAction.EXECUTE_BUY
        return RiskDecision(
            action=action,
            symbol=technical.symbol,
            position_size=size,
            entry=technical.price,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=technical.confidence,
            reasoning=(
                f"{reason} Closing size {size:.8g} at market near {technical.price:.8g}. "
                f"{technical.reasoning}"
            ),
            timestamp=technical.timestamp,
        )

    def _no(
        self,
        technical: TechnicalSignal,
        reason: str,
        confidence: float = 0.0,
        now: datetime | None = None,
    ) -> RiskDecision:
        return RiskDecision(
            action=RiskAction.NO_ACTION,
            symbol=technical.symbol,
            position_size=0.0,
            entry=technical.price,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=f"{reason} Technical {technical.action.value} confidence {technical.confidence:.2f}.",
            timestamp=now or utcnow(),
        )

    def _size(self, technical: TechnicalSignal, portfolio: PortfolioView) -> _Sizing | None:
        settings = self.settings
        entry = technical.price
        if entry <= 0 or portfolio.equity <= 0:
            return None
        notes: list[str] = []
        if settings.use_atr_stops and technical.atr > 0:
            stop_distance = technical.atr * settings.atr_multiplier
            notes.append(f"ATR stop {technical.atr:.8g} x {settings.atr_multiplier:g}")
        else:
            stop_distance = entry * settings.fixed_stop_pct
            if settings.use_atr_stops:
                notes.append("ATR missing, using the fixed percent stop")
            else:
                notes.append(f"Fixed stop {settings.fixed_stop_pct:.2%} of price")
        if stop_distance <= 0 or not math.isfinite(stop_distance):
            return None

        risk_budget = portfolio.equity * settings.risk_per_trade
        size = risk_budget / stop_distance
        is_short = technical.action == SignalAction.SELL
        if is_short:
            stop_loss = entry + stop_distance
            take_profit = entry - stop_distance * settings.reward_risk_ratio
        else:
            stop_loss = entry - stop_distance
            take_profit = entry + stop_distance * settings.reward_risk_ratio
        if stop_loss <= 0 or take_profit <= 0:
            return None

        # Cash needed per unit: full notional for a spot long, margin otherwise.
        uses_margin = settings.market_type == "swap" or is_short
        leverage = settings.leverage if settings.market_type == "swap" else 1.0
        per_unit = entry * ((1.0 / leverage) if uses_margin else 1.0) + entry * settings.fee_rate
        affordable = (portfolio.cash * settings.cash_utilization) / per_unit if per_unit > 0 else 0.0
        max_position = (portfolio.equity * settings.max_notional_pct) / entry
        exposure_room = portfolio.equity * settings.max_total_exposure_pct - portfolio.open_notional
        max_by_exposure = max(0.0, exposure_room) / entry
        for limit, label in (
            (affordable, "capped by available cash"),
            (max_position, f"capped at {settings.max_notional_pct:.0%} of equity per position"),
            (max_by_exposure, f"capped by total exposure {settings.max_total_exposure_pct:.0%}"),
        ):
            if size > limit:
                size = limit
                notes.append(label)
        size = math.floor(size * 1e8) / 1e8
        if size <= 0 or size * entry < settings.min_notional:
            return None

        # Round trip: fee and slippage on entry and on exit, per unit.
        cost_per_unit = entry * 2 * (settings.fee_rate + settings.slippage_pct)
        risk_per_unit = abs(entry - stop_loss) + cost_per_unit
        reward_per_unit = abs(take_profit - entry) - cost_per_unit
        reward_risk = max(0.0, reward_per_unit) / risk_per_unit if risk_per_unit > 0 else 0.0
        return _Sizing(
            size=size,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=size * risk_per_unit,
            reward_risk=reward_risk,
            stop_distance=stop_distance,
            notes=" ".join(notes) + ".",
        )


def _direction(action: SignalAction) -> int:
    if action == SignalAction.BUY:
        return 1
    if action == SignalAction.SELL:
        return -1
    return 0


def _combined_confidence(technical: TechnicalSignal, sentiment: SentimentSignal) -> tuple[float, str]:
    alignment = _direction(technical.action) * sentiment.score
    if alignment >= 0:
        factor = 1.0 + _AGREEMENT_BOOST * alignment
    else:
        factor = 1.0 + _DISAGREEMENT_PENALTY * alignment
    combined = max(0.0, min(1.0, technical.confidence * factor))
    note = (
        f"Sentiment {sentiment.bias.value} ({sentiment.score:+.2f}) "
        f"alignment {alignment:+.2f}"
    )
    return combined, note
