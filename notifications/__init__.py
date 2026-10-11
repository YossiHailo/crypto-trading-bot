"""Outbound alerts. Telegram is a no-op until a bot token and chat id are set."""

from notifications.telegram import TelegramNotifier

__all__ = ["TelegramNotifier"]
