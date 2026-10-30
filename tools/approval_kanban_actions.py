"""Hosted GitHub Actions budget policy for autonomous Kanban workers."""

import logging
import os
import re

from tools.approval_detection import (
    _command_detection_variants,
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_word_spans,
    _iter_top_level_shell_segments,
    _shell_segment_tokens,
)

logger = logging.getLogger(__name__)


_GITHUB_ACTIONS_MUTATION_ENDPOINT = re.compile(
    r"(?:^|/)actions/(?:"
    r"runs/[0-9]+/(?:rerun|rerun-failed-jobs|cancel)"
    r"|workflows/[^/]+/dispatches"
    r")(?:$|[/?#])"
    r"|(?:^|/)repos/[^/]+/[^/]+/dispatches(?:$|[/?#])",
    re.IGNORECASE,
)


def _kanban_github_actions_mutation(command: str) -> bool:
    """Detect GitHub Actions mutations forbidden to autonomous Kanban workers.

    Kanban workers may inspect run/check state, but starting, rerunning, or
    cancelling hosted CI is an operator-budget action. This guard uses the
    existing ``HERMES_KANBAN_TASK`` execution identity and runs before all
    container/yolo bypasses so a worker cannot gain that authority by changing
    terminal backend or approval mode.
    """

    if not os.environ.get("HERMES_KANBAN_TASK", "").strip():
        return False
    for variant in _command_detection_variants(command):
        for segment in _iter_top_level_shell_segments(variant):
            for start, _, word in _iter_shell_command_word_spans(segment):
                executable = os.path.basename(
                    _deobfuscate_shell_word_for_detection(word)
                ).casefold()
                if executable not in {"gh", "curl"}:
                    continue
                tokens = _shell_segment_tokens(segment, start)
                if not tokens:
                    continue
                lowered = [token.casefold() for token in tokens[1:]]
                if executable == "gh":
                    pairs = set(zip(lowered, lowered[1:]))
                    if pairs.intersection(
                        {("run", "rerun"), ("run", "cancel"), ("workflow", "run")}
                    ):
                        return True
                if any(_GITHUB_ACTIONS_MUTATION_ENDPOINT.search(token) for token in lowered):
                    return True
    return False


def _kanban_github_actions_block_result() -> dict:
    return {
        "approved": False,
        "kanban_policy": "github_actions_mutation",
        "message": (
            "BLOCKED: Kanban workers may inspect GitHub Actions, but may not "
            "start, rerun, or cancel hosted CI. This consumes operator-owned "
            "Actions budget and requires an explicit operator action outside "
            "the worker."
        ),
    }


def kanban_github_actions_block(command: str) -> dict | None:
    """Return the unconditional worker-budget refusal, otherwise leave approval policy unchanged."""
    if not _kanban_github_actions_mutation(command):
        return None
    logger.warning("Kanban GitHub Actions mutation blocked: %s", command[:200])
    return _kanban_github_actions_block_result()
