"""Secret redaction for logs, Telegram messages and the dashboard.

Exchange and HTTP errors often echo the request URL. CryptoPanic takes its
token in the query string and Telegram puts the bot token in the path, so a
plain ``str(exc)`` can leak credentials into the log file or a chat. Every
outgoing text goes through ``redact``. The logging filter applies it to all
records, including tracebacks.
"""

from __future__ import annotations

import logging
import re
import threading

_LOCK = threading.Lock()
_SECRETS: set[str] = set()
_PATTERNS = (
    # ?auth_token=...  &api_key=...  &signature=...
    re.compile(r"(?i)([?&](?:auth_token|token|api_?key|apikey|secret|signature|key)=)[^&\s'\"]+"),
    # https://api.telegram.org/bot123456:ABC.../sendMessage
    re.compile(r"(/bot)\d+:[A-Za-z0-9_-]+"),
)
MASK = "***"


def register_secret(value: str | None) -> None:
    """Remember a value that must never appear in output. Short values are ignored."""
    if value and len(value) >= 6:
        with _LOCK:
            _SECRETS.add(value)


def register_secrets(values: list[str]) -> None:
    for value in values:
        register_secret(value)


def redact(text: object) -> str:
    out = str(text)
    with _LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for secret in secrets:
        if secret in out:
            out = out.replace(secret, MASK)
    for pattern in _PATTERNS:
        out = pattern.sub(lambda match: match.group(1) + MASK, out)
    return out


class RedactingFilter(logging.Filter):
    """Rewrites the final message (args merged) and any exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed format args: leave the record alone
            return True
        record.msg = redact(message)
        record.args = None
        if record.exc_info:
            formatter = logging.Formatter()
            record.exc_text = redact(formatter.formatException(record.exc_info))
            record.exc_info = None
        return True
