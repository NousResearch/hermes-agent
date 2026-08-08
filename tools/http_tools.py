import logging
import ssl
import time
import urllib.error
from typing import Any, Callable, Dict, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 0.5,
    "backoff_factor": 2.0,
    "max_delay": 10.0,
}

RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.TransportError,
    ssl.SSLError,
)


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (OSError, TimeoutError))
    return isinstance(exc, RETRYABLE_ERRORS)


def retryable_get(
    request_fn: Callable[[], T],
    max_attempts: int = DEFAULT_RETRY_CONFIG["max_attempts"],
    base_delay: float = DEFAULT_RETRY_CONFIG["base_delay"],
    backoff_factor: float = DEFAULT_RETRY_CONFIG["backoff_factor"],
    max_delay: float = DEFAULT_RETRY_CONFIG["max_delay"],
    logger_extra: Optional[Dict[str, Any]] = None,
) -> T:
    """Retry an idempotent GET operation after transient transport failures."""
    extra = logger_extra or {}
    attempt = 0
    delay = base_delay

    while True:
        attempt += 1
        try:
            return request_fn()
        except Exception as exc:
            if not _is_retryable_exception(exc) or attempt >= max_attempts:
                logger.warning(
                    "HTTP call failed (non-retryable or max attempts reached).",
                    exc_info=True,
                    extra=extra,
                )
                raise
            sleep_time = min(delay, max_delay)
            logger.warning(
                f"HTTP call failed with retryable error ({type(exc).__name__}): {exc}. "
                f"Retry attempt {attempt + 1}/{max_attempts} in {sleep_time:.2f}s.",
                exc_info=True,
                extra=extra,
            )
            time.sleep(sleep_time)
            delay *= backoff_factor
