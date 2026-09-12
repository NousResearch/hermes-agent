"""Contract tests for the deprecated docker/entrypoint.sh back-compat shim.

Legacy callers (NAS App Center / UGOS packages, old compose) hard-code
``docker/entrypoint.sh`` as the container ENTRYPOINT. The shim must
forward to ``entrypoint-dispatch.sh`` so CMD still runs:

- PID 1 → dispatcher execs ``/init`` + ``main-wrapper.sh``
- non-PID-1 → dispatcher runs stage2 then execs ``main-wrapper.sh``

The shim must NOT exec ``stage2-hook.sh`` itself (that is bootstrap-only
and never starts CMD). See issue #107263.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "docker" / "entrypoint.sh"


def test_deprecated_shim_execs_dispatcher_not_only_stage2():
    text = SHIM.read_text(encoding="utf-8")
    # strip comments so a comment mentioning dispatch cannot fake the contract
    code = "\n".join(
        ln for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    assert 'exec /opt/hermes/docker/entrypoint-dispatch.sh "$@"' in code
    assert "stage2-hook.sh" not in code  # shim must not exec stage2 itself
    assert "WARNING" in code  # deprecation still surfaced


def test_dockerfile_entrypoint_unchanged_control():
    df = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT [ "/opt/hermes/docker/entrypoint-dispatch.sh" ]' in df
