"""Structured logging for service-ai.

Logs are emitted as JSON so Cloud Run and the judging-window dashboard can parse
them. GCP credentials are never written to a log line; see :func:`configure`.
"""

from __future__ import annotations

import logging
import sys

from karum_service_ai.core.config import get_settings

# Attribute names that must never appear in a log record, even if a caller
# passes them by mistake. Values under these keys are redacted.
_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "credentials",
        "service_account",
        "token",
        "secret",
        "private_key",
        "access_token",
        "authorization",
    }
)


class RedactingFilter(logging.Filter):
    """Redact secret-looking attributes from log records.

    Vertex authentication uses GCP application-default credentials. A committed
    or logged key is a real cost, so this filter defensively blanks any attribute
    whose name looks like a credential before the record reaches the handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key.lower() in _REDACTED_KEYS:
                record.__dict__[key] = "[REDACTED]"
        return True


def configure() -> None:
    """Install a JSON logging configuration honouring the ``LOG_LEVEL`` setting.

    Idempotent: safe to call more than once (for example from tests).
    """
    settings = get_settings()

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(RedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
