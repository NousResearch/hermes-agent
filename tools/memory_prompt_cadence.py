"""Retry missed memory prompts and remind about unanswered proposals in the current profile."""

from __future__ import annotations

import logging
import math
import time

from tools import write_approval as wa

logger = logging.getLogger(__name__)


def tick(*, now: float | None = None) -> int:
    """Run from the gateway scheduler under the owning profile's runtime scope.

    Delivery rechecks the durable clock under its record lock. A failed delivery leaves
    the record due; a completed delivery advances that clock before another tick can send.
    """
    from hermes_cli.config import cfg_get, load_config
    from tools.memory_prompt_send import send_prompt

    from tools.memory_review_conversion import convert_pending_batches, retire_combined_prompts
    convert_pending_batches()
    try:
        retire_combined_prompts()
    except Exception:
        logger.warning("Could not retire an old combined memory card; will retry", exc_info=True)
    from tools.memory_space_refresh import refresh_space_reviews
    refresh_space_reviews()
    hours = cfg_get(load_config(), "tools", "write_approval", "nag_hours", default=2)
    try:
        seconds = float(hours) * 3600
    except (TypeError, ValueError):
        seconds = 7200
    if not math.isfinite(seconds) or seconds <= 0:
        return 0
    now = time.time() if now is None else now
    sent = 0
    for record in wa.list_pending(wa.MEMORY):
        try:
            sent += bool(send_prompt(wa.MEMORY, record, due_before=now, nag_seconds=seconds))
        except Exception:
            # One damaged record must not suppress the other profiles' or records' prompts.
            logger.warning("Memory reminder failed for %s", record.get("id"), exc_info=True)
    from tools.skill_prompt_send import retry_prompts
    sent += retry_prompts(now=now, nag_seconds=seconds)
    return sent
