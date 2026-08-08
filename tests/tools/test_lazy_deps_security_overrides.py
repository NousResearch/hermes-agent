"""Lazy installs must not downgrade a security-pinned package.

``uv pip install`` and ``pip install`` do not read ``[tool.uv]
override-dependencies``. A backend whose transitive dependencies cap a
pinned package below its patched version therefore downgrades the core venv
the first time a user enables that backend.

The measured case: the venv holds ``cryptography==50.0.0``, and enabling
DingTalk pulls ``alibabacloud-tea-openapi==0.4.5``, which caps
``cryptography<49``, so the install resolves 48.0.1 and re-opens
GHSA-m2h6-j472-rp4c, GHSA-jwv3-5hgf-82ww and CVE-2026-69247.

tools/lazy_deps.py reads the override list out of pyproject.toml and gives
it to each installer tier. These tests check that both tiers receive it.
There is no test that two lists agree, because there is one list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import lazy_deps as ld


class TestOverridesReachBothInstallerTiers:
    """Both tiers of the install ladder must receive the floor."""

    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        """Run ``_venv_pip_install`` with both tiers stubbed, capturing argv.

        Temp files are read *during* the stubbed call, because
        ``_venv_pip_install`` unlinks them in its ``finally`` block.
        """
        calls: list[list[str]] = []
        contents: dict[str, str] = {}

        def fake_run(cmd, *a, **kw):
            cmd = list(cmd)
            calls.append(cmd)
            for flag in ("--overrides", "--constraint"):
                if flag in cmd:
                    p = Path(cmd[cmd.index(flag) + 1])
                    if p.exists():
                        contents[flag] = p.read_text(encoding="utf-8")

            class R:
                # uv tier fails so the ladder falls through to pip; the pip
                # probe and the pip install itself succeed, so the --no-deps
                # repair pass runs.
                returncode = 1 if (cmd and "uv" in cmd[0]) else 0
                stdout = "pip 24.0"
                stderr = "stubbed"

            return R()

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        monkeypatch.setattr(ld.shutil, "which", lambda _n: "/usr/bin/uv")
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        ld._venv_pip_install(("alibabacloud-dingtalk==2.2.42",))
        return calls, contents

    def test_uv_tier_receives_overrides_flag(self, captured):
        calls, contents = captured
        uv_calls = [c for c in calls if "uv" in c[0] and "pip" in c]
        assert uv_calls, f"no uv tier invocation captured: {calls}"
        cmd = uv_calls[0]
        assert "--overrides" in cmd, (
            f"uv tier must pass --overrides so [tool.uv] semantics apply: {cmd}"
        )
        body = contents.get("--overrides", "")
        for spec in ld._security_overrides():
            assert spec in body, (
                f"override {spec!r} missing from the file handed to uv: {body!r}"
            )

    def test_pip_tier_reasserts_the_floor_with_no_deps(self, captured):
        """pip has no --overrides; it must re-assert the floor via --no-deps.

        Passing the floor as a --constraint instead would hold the pinned
        package but resolve the *backend* backwards, so the repair pass is the
        behaviour under test.
        """
        calls, _ = captured
        repair = [
            c for c in calls if "install" in c and "--no-deps" in c
        ]
        assert repair, (
            f"pip tier must re-assert security overrides with --no-deps: {calls}"
        )
        cmd = repair[0]
        for spec in ld._security_overrides():
            assert spec in cmd, (
                f"override {spec!r} missing from the pip repair pass: {cmd}"
            )

    def test_pip_repair_pass_does_not_reinstall_the_backend(self, captured):
        """The repair pass must touch only the overridden packages.

        Including the backend specs would re-run resolution and undo the point
        of --no-deps.
        """
        calls, _ = captured
        repair = [c for c in calls if "install" in c and "--no-deps" in c]
        assert repair
        assert "alibabacloud-dingtalk==2.2.42" not in repair[0], (
            f"repair pass must not re-install the backend: {repair[0]}"
        )

    def test_temp_files_are_cleaned_up(self, captured):
        calls, _ = captured
        for cmd in calls:
            for flag in ("--overrides", "--constraint"):
                if flag in cmd:
                    leaked = Path(cmd[cmd.index(flag) + 1])
                    assert not leaked.exists(), (
                        f"{flag} temp file leaked after install: {leaked}"
                    )

    def test_specs_still_reach_the_installer(self, captured):
        """The override plumbing must not displace the actual packages."""
        calls, _ = captured
        # Exclude the --no-deps repair pass, which deliberately carries only
        # the overridden packages (see test_pip_repair_pass_does_not_reinstall).
        installs = [
            c for c in calls if "install" in c and "--no-deps" not in c
        ]
        assert installs, f"no install invocation captured: {calls}"
        for cmd in installs:
            assert "alibabacloud-dingtalk==2.2.42" in cmd, (
                f"requested spec missing from install command: {cmd}"
            )
