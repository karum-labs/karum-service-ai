"""Logging setup for service-ai.

Logs are emitted as single plain-text lines (timestamp, level, logger name,
message) that Cloud Run and the judging-window dashboard can filter. GCP
credentials are never written to a log line; see :func:`configure`.
"""

from __future__ import annotations

import logging
import re
import sys

from karum_service_ai.core.config import get_settings

# Attribute names that must never appear in a log record, even if a caller
# passes them by mistake. Values under these keys are redacted.
_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "gemini_api_key",
        "credentials",
        "service_account",
        "token",
        "secret",
        "private_key",
        "access_token",
        "authorization",
    }
)

# Control characters a name or message could use to forge a second log line.
_CONTROL_RE = re.compile(r"[\r\n\t\x00-\x1f\x7f]+")


def sanitise_text(value: object) -> str:
    """Render one untrusted value as a single safe log line, no delimiters.

    Newlines and control characters are collapsed to spaces so a name or
    message cannot forge a second log record.
    """
    return _CONTROL_RE.sub(" ", " ".join(str(value).split()))


class RedactingFilter(logging.Filter):
    """Redact secret-looking attributes from log records.

    Vertex authentication uses GCP application-default credentials. A committed
    or logged key is a real cost, so this filter defensively blanks any attribute
    whose name looks like a credential before the record reaches the handler.

    The *formatted* message is redacted too: blanking attribute names is not
    enough, because ``record.getMessage()`` interpolates untrusted values into
    the format string, and a name containing a newline would forge a second line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key.lower() in _REDACTED_KEYS:
                record.__dict__[key] = "[REDACTED]"
        record.msg = sanitise_text(record.msg)
        if record.args:
            record.args = tuple(
                sanitise_text(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def configure() -> None:
    """Install the logging configuration honouring the ``LOG_LEVEL`` setting.

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
