"""Shared logging helpers for the Aria ML project.

This module centralizes process-wide logger configuration so every script and
desktop UI component emits consistent, timestamped messages. It depends only on
the Python standard library and produces configured :class:`logging.Logger`
instances for callers.
"""

import logging
import sys


_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the current process.

    Args:
        level: Root log level applied when logging is first initialized.

    Returns:
        None.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-scoped logger.

    Args:
        name: Logger name, typically the caller's ``__name__``.

    Returns:
        A ready-to-use logger instance.
    """
    setup_logging()
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging()
    get_logger(__name__).info("logging_utils provides shared logger configuration.")
