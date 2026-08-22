"""Tests for tools/skill_verify.py — the per-skill mechanical verifier (Layer 1).

Real subprocesses against a throwaway task dir + temp skill dir, no mocks.
The trust boundary (un-opted-in skill's verifier never executes) is exercised
with a sentinel file the script would write if it ran.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def verify_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a skills/ dir and a throwaway task cwd."""
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    task_cwd = tmp_path / "task"
    task_cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import tools.skill_usage as skill_usage
    importlib.reload(skill_usage)
    return {
        "home": home,
        "skills": skills,
        "task_cwd": task_cwd,
        "skill_usage": skill_usage,
    }


def _write_skill_with_verify(skills_dir: Path, name: str, verify_block: str):
    """Create a skill whose frontmatter carries a metadata.hermes.verify block."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    scripts = d / "scripts"
    scripts.mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: test skill\n"
        "version: 1.0.0\n"
        "metadata:\n"
        "  hermes:\n"
        f"    verify: {verify_block}\n"
        "---\n"
        f"# {name}\n",
        encoding="utf-8",
    )
    return d


def _verify_block(run: str, applicability: str | None = None, timeout: int = 30) -> str:
    """Render a nested metadata.hermes.verify dict as inline flow-mapping YAML.

    One line, comma-separated: the frontmatter parser is full YAML (CSafeLoader)
    and a multi-line flow mapping without commas trips the fallback flatten.
    """
    parts = ["run: " + json.dumps(run)]
    if applicability:
        parts.append("applicability_check: " + json.dumps(applicability))
    parts.append(f"timeout_seconds: {timeout}")
    return "{" + ", ".join(parts) + "}"


def _write_script(skill_dir: Path, relative: str, body: str) -> Path:
    p = skill_dir / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _json_script(success: bool, reason: str) -> str:
    payload = json.dumps({"success": success, "reason": reason})
    return "import json\nprint(" + repr(payload) + ")\n"


def _sleep_script(seconds: int) -> str:
    return f"import time\ntime.sleep({seconds})\n"


# ---------------------------------------------------------------------------
# Trust boundary — verifier must NEVER execute without the user's opt-in
# ---------------------------------------------------------------------------

def test_unopted_in_skill_verifier_never_executes(verify_env):
    """A declared verify block is not consent. Without set_verify_enabled the
    run script must not even start — even if the skill is otherwise eligible."""
    env = verify_env
    sentinel = env["task_cwd"] / "ran-sentinel"
    d = _write_skill_with_verify(
        env["skills"], "svc_consent", _verify_block("scripts/verify.py")
    )
    _write_script(
        d,
        "scripts/verify.py",
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
        + _json_script(True, "should never be reached"),
    )

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_consent", d, env["task_cwd"])
    assert outcome is None
    assert not sentinel.exists()


def test_verify_enabled_is_a_per_skill_flag(verify_env):
    """Opting in one skill must not enable a sibling."""
    env = verify_env
    sentinel = env["task_cwd"] / "ran-a"
    da = _write_skill_with_verify(
        env["skills"], "svc_a", _verify_block("scripts/verify.py")
    )
    _write_script(
        da,
        "scripts/verify.py",
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
        + _json_script(True, "ok"),
    )
    db = _write_skill_with_verify(
        env["skills"], "svc_b", _verify_block("scripts/verify.py")
    )
    _write_script(
        db,
        "scripts/verify.py",
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
        + _json_script(True, "ok"),
    )

    env["skill_usage"].set_verify_enabled("svc_a", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_a", da, env["task_cwd"]) is not None
    assert run_verification("svc_b", db, env["task_cwd"]) is None


# ---------------------------------------------------------------------------
# Judgment — JSON success / JSON fail / exit-code fallback
# ---------------------------------------------------------------------------

def test_json_success_verdict(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_json_ok", _verify_block("scripts/verify.py")
    )
    _write_script(d, "scripts/verify.py", _json_script(True, "clean"))
    env["skill_usage"].set_verify_enabled("svc_json_ok", True)

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_json_ok", d, env["task_cwd"])
    assert outcome is not None
    assert outcome.success is True
    assert outcome.reason == "clean"


def test_json_fail_reason_surfaces(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_json_bad", _verify_block("scripts/verify.py")
    )
    _write_script(d, "scripts/verify.py", _json_script(False, "no type prefix"))
    env["skill_usage"].set_verify_enabled("svc_json_bad", True)

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_json_bad", d, env["task_cwd"])
    assert outcome is not None
    assert outcome.success is False
    assert outcome.reason == "no type prefix"


def test_non_json_output_falls_back_to_exit_code(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_fb_ok", _verify_block("scripts/verify.py")
    )
    _write_script(d, "scripts/verify.py", "print('garbage on stdout')\n")
    env["skill_usage"].set_verify_enabled("svc_fb_ok", True)

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_fb_ok", d, env["task_cwd"])
    assert outcome is not None
    assert outcome.success is True


def test_non_json_failure_exit_code_carries_reason(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_fb_bad", _verify_block("scripts/verify.py")
    )
    _write_script(
        d,
        "scripts/verify.py",
        "import sys\nprint('no such file', file=sys.stderr)\nsys.exit(3)\n",
    )
    env["skill_usage"].set_verify_enabled("svc_fb_bad", True)

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_fb_bad", d, env["task_cwd"])
    assert outcome is not None
    assert outcome.success is False
    assert "no such file" in outcome.reason


# ---------------------------------------------------------------------------
# Applicability — SKIP is a third outcome, never recorded
# ---------------------------------------------------------------------------

def test_applicability_probe_skip_returns_none(verify_env):
    """When the applicability probe exits nonzero the turn isn't judgeable —
    None, so bump_outcome is never called with a fake success."""
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"],
        "svc_skip",
        _verify_block("scripts/verify.py", applicability="scripts/applicable.py"),
    )
    _write_script(d, "scripts/applicable.py", "import sys\nsys.exit(1)\n")
    _write_script(d, "scripts/verify.py", _json_script(True, "ok"))
    env["skill_usage"].set_verify_enabled("svc_skip", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_skip", d, env["task_cwd"]) is None


def test_applicability_probe_pass_runs_verifier(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"],
        "svc_applicable",
        _verify_block("scripts/verify.py", applicability="scripts/applicable.py"),
    )
    _write_script(d, "scripts/applicable.py", "import sys\nsys.exit(0)\n")
    _write_script(d, "scripts/verify.py", _json_script(False, "still wrong"))
    env["skill_usage"].set_verify_enabled("svc_applicable", True)

    from tools.skill_verify import run_verification

    outcome = run_verification("svc_applicable", d, env["task_cwd"])
    assert outcome is not None
    assert outcome.success is False
    assert outcome.reason == "still wrong"


# ---------------------------------------------------------------------------
# Robustness — timeout, missing spec, path escape
# ---------------------------------------------------------------------------

@pytest.mark.live_system_guard_bypass
def test_verifier_timeout_returns_failure(verify_env):
    """A runaway verifier must fail the check, not hang the turn.

    ``live_system_guard_bypass``: ``subprocess.run`` enforces ``timeout`` by
    ``Popen.kill()`` → ``os.kill`` on the child, and the harness guard blocks
    killing any PID it can't confirm as a test descendant (no psutil in the
    venv to walk the parent chain). The kill targets a child THIS test just
    spawned, so real signal delivery is genuinely required.
    """
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_hang", _verify_block("scripts/verify.py", timeout=1)
    )
    _write_script(d, "scripts/verify.py", _sleep_script(5))
    env["skill_usage"].set_verify_enabled("svc_hang", True)

    from tools.skill_verify import run_verification

    start = time.monotonic()
    outcome = run_verification("svc_hang", d, env["task_cwd"])
    elapsed = time.monotonic() - start
    assert outcome is not None
    assert outcome.success is False
    assert "timed out" in outcome.reason
    assert elapsed < 4


@pytest.mark.live_system_guard_bypass
def test_applicability_probe_timeout_is_skip_not_fail(verify_env):
    """A slow applicability probe means the turn was never judgeable — it must
    SKIP (None), never record a mechanical FAIL against the skill."""
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"],
        "svc_probe_slow",
        _verify_block(
            "scripts/verify.py",
            applicability="scripts/applicable.py",
            timeout=1,
        ),
    )
    _write_script(d, "scripts/applicable.py", _sleep_script(5))
    _write_script(d, "scripts/verify.py", _json_script(False, "should not run"))
    env["skill_usage"].set_verify_enabled("svc_probe_slow", True)

    from tools.skill_verify import run_verification

    start = time.monotonic()
    outcome = run_verification("svc_probe_slow", d, env["task_cwd"])
    elapsed = time.monotonic() - start
    assert outcome is None  # skip, not a FAIL
    assert elapsed < 4  # bounded by min(timeout, 10), not the 5s sleep


def test_missing_verify_block_returns_none(verify_env):
    env = verify_env
    d = env["skills"] / "svc_no_verify"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: svc_no_verify\ndescription: no verify here\n---\n# body\n",
        encoding="utf-8",
    )
    env["skill_usage"].set_verify_enabled("svc_no_verify", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_no_verify", d, env["task_cwd"]) is None


def test_script_escaping_skill_dir_is_refused(verify_env):
    """A verifier run path pointing outside the skill dir is a hostile or buggy
    author — refuse to execute rather than follow it.

    The run path resolves to a REAL file outside the skill dir (created
    below), so ``_resolve_command``'s escape check — not a missing script —
    is the only thing that can produce None.
    """
    env = verify_env
    outside = env["task_cwd"] / "outside.py"
    outside.write_text(_json_script(True, "nope"), encoding="utf-8")
    # Relative path from <home>/skills/svc_escape to <tmp_path>/task/outside.py
    escape_path = "../../../task/outside.py"
    d = _write_skill_with_verify(
        env["skills"],
        "svc_escape",
        _verify_block(escape_path),
    )
    assert (d / escape_path).resolve() == outside.resolve()
    env["skill_usage"].set_verify_enabled("svc_escape", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_escape", d, env["task_cwd"]) is None


def test_missing_script_returns_none(verify_env):
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_missing", _verify_block("scripts/verify.py")
    )
    env["skill_usage"].set_verify_enabled("svc_missing", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_missing", d, env["task_cwd"]) is None


def test_unrecognized_script_suffix_is_refused(verify_env):
    """A verifier run path whose suffix has no mapped interpreter must be
    refused (None), not executed as a raw shell command."""
    env = verify_env
    d = _write_skill_with_verify(
        env["skills"], "svc_bad_suffix", _verify_block("scripts/verify.xyz")
    )
    _write_script(d, "scripts/verify.xyz", "#!/bin/sh\necho ok\n")
    env["skill_usage"].set_verify_enabled("svc_bad_suffix", True)

    from tools.skill_verify import run_verification

    assert run_verification("svc_bad_suffix", d, env["task_cwd"]) is None
