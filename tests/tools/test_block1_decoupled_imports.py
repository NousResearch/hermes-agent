"""Block 1 focal tests for run_agent.py — uses simple line iteration only.

No regex with DOTALL or nested wildcards. Each test reads the file
once and inspects exact lines.
"""
from pathlib import Path


RUN_AGENT = Path(__file__).resolve().parents[2] / "run_agent.py"


def _read_lines():
    return RUN_AGENT.read_text(encoding="utf-8").splitlines()


def test_block_1_repair_block_is_present():
    """The Block 1 repair contract must be present in run_agent.py."""
    lines = _read_lines()
    found = False
    for i, line in enumerate(lines):
        if "Block 1 repair" in line:
            found = True
            break
    assert found, "Block 1 repair marker must be present in run_agent.py"


def test_decoupling_uses_separate_try_blocks():
    """The two import authorities must be in separate try/except ImportError
    blocks. We check by counting ``try:`` lines that import one of the
    two modules and the corresponding ``except ImportError`` is on a
    different line region."""
    lines = _read_lines()
    in_block_1 = False
    swp_try = -1
    si_try = -1
    n_excepts = 0
    for i, line in enumerate(lines):
        if "Block 1 repair" in line:
            in_block_1 = True
            continue
        if not in_block_1:
            continue
        # Stop at the next major section boundary
        if "finish_logical_calls" in line and i > 8950:
            break
        if line.strip() == "try:":
            # Look ahead for the import
            for j in range(i + 1, min(i + 8, len(lines))):
                if "from agent.session_write_policy" in lines[j]:
                    swp_try = i
                if "from agent.self_improvement_decision_context" in lines[j]:
                    si_try = i
        if "except ImportError" in line:
            n_excepts += 1
    assert swp_try >= 0, "session_write_policy import must be in a try block"
    assert si_try >= 0, "self_improvement_decision_context import must be in a try block"
    assert swp_try != si_try, (
        "The two import authorities must be in separate try blocks "
        f"(swp_try={swp_try}, si_try={si_try})"
    )
    assert n_excepts >= 2, (
        f"Block 1 must have at least two separate try/except ImportError "
        f"blocks (got {n_excepts})"
    )


def test_swp_module_ok_flag_is_set():
    """The _swp_module_ok flag must be set after the try block."""
    lines = _read_lines()
    swp_module_set = False
    for line in lines:
        if "_swp_module_ok = True" in line:
            swp_module_set = True
            break
    assert swp_module_set, "_swp_module_ok = True must be present"


def test_swp_fail_closed_branch_exists():
    """When session_write_policy import fails, the turn body must NOT run.
    The branch must be reached via ``if not _swp_module_ok:`` and must
    log an error and return early."""
    lines = _read_lines()
    fail_idx = -1
    for i, line in enumerate(lines):
        if "if not _swp_module_ok:" in line:
            fail_idx = i
            break
    assert fail_idx >= 0, (
        "Must have explicit fail-closed branch on _swp_module_ok=False"
    )
    # Look at the next ~20 lines for the logger.error and return
    end_window = lines[fail_idx:fail_idx + 25]
    window = "\n".join(end_window)
    assert "logger.error" in window, "Fail-closed branch must log the error"
    assert "refusing" in window, (
        "Fail-closed branch must explicitly refuse to run the turn body"
    )
    assert "return" in window, "Fail-closed branch must return early"


def test_si_decoupled_preserves_swp_scope():
    """When self_improvement_decision_context import fails but
    session_write_policy succeeds, the session_write_policy_scope must
    still be active. We check for the elif branch ``elif _swp_scope is
    not None:`` that opens ``_swp_scope(_swp_policy)``."""
    lines = _read_lines()
    elif_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "elif _swp_scope is not None:":
            elif_idx = i
            break
    assert elif_idx >= 0, (
        "Must have a branch for _swp_scope active but decision context "
        "unavailable"
    )
    # The branch must contain _swp_scope(_swp_policy) as a context manager
    end_window = lines[elif_idx:elif_idx + 15]
    window = "\n".join(end_window)
    assert "_swp_scope(_swp_policy)" in window, (
        "session_write_policy_scope must be active in this branch"
    )


def test_swp_module_ok_appears_before_swp_scope_check():
    """The _swp_module_ok check must come before _swp_scope check."""
    lines = _read_lines()
    fail_idx = -1
    swp_idx = -1
    for i, line in enumerate(lines):
        if "if not _swp_module_ok:" in line:
            fail_idx = i
        if "_swp_scope is not None" in line and "if " in line:
            swp_idx = i
            break
    assert fail_idx >= 0
    assert swp_idx >= 0
    assert fail_idx < swp_idx, (
        "_swp_module_ok check must precede _swp_scope check"
    )


def test_refuses_to_run_turn_body_text():
    """The fail-closed branch must include the canonical refusal text."""
    lines = _read_lines()
    for line in lines:
        if "refusing to run turn body" in line or "refusing to" in line:
            return
    assert False, "Fail-closed branch must contain a 'refusing to' message"
