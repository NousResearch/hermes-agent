"""SDK httpx floor: the MCP SDK 2.0 transport depends on httpx2, not the legacy httpx.

Regression contract for the 6 GHSA advisories that ship in httpx2 2.7.0
(https://github.com/NousResearch/hermes-agent/issues/108219, also #125668):
decompression amplification (GHSA-8xx6-hgc6-gc2m, fixed 2.12.0), SSE CPU DoS
(GHSA-f2fp-rgf2-35cp, fixed 2.10.0), multipart header injection
(GHSA-h4x7-gw46-3wm6, fixed 2.11.0), SOCKS TLS bypass (GHSA-7mj9-2mp8-4m2p,
fixed 2.10.0), and the conflicting Content-Length / Transfer-Encoding
auto-generation (GHSA-pf96-p4fj-6566, fixed 2.11.0). The pin in
``pyproject.toml`` is the only thing that pins httpx2 — there is no upper
bound enforced elsewhere — so a silent downgrade would re-introduce every
finding. This file pins the contract from two sides: the declared floor
must clear 2.13.0 (the first release after all six fixes), and the SDK's
resolved httpx module must satisfy it.

These tests do not assume a specific exact version (that would be a
change-detector snapshot). They assert the relationship: declared pin
and resolved module agree, and both clear the security floor.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


# Floor chosen so that all six published GHSAs are closed regardless of
# which fixes were backported; the first release after the latest fix is
# 2.12.0, so 2.13.0 leaves a buffer.
_SECURITY_FLOOR = (2, 13, 0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pin_specs() -> list[str]:
    """Every httpx2 pin declared in pyproject.toml (one per extra + dev)."""
    data = tomllib.loads(_repo_root().joinpath("pyproject.toml").read_text())
    out: list[str] = []
    for extra in (data.get("project", {}).get("optional-dependencies") or {}).values():
        for spec in extra:
            match = re.match(r"^httpx2([<>=!~]=.+)$", spec.strip())
            if match:
                out.append(match.group(1))
    for spec in data.get("dependency-groups", {}).get("dev", []):
        match = re.match(r"^httpx2([<>=!~]=.+)$", spec.strip())
        if match:
            out.append(match.group(1))
    return out


def _parse_spec(spec: str) -> tuple[int, ...]:
    match = re.match(r"^==(\d+)\.(\d+)\.(\d+)$", spec.strip())
    assert match, f"httpx2 pin {spec!r} is not ==X.Y.Z; bounds drift would break the security floor"
    return tuple(int(group) for group in match.groups())


def test_httpx2_pin_clears_security_floor() -> None:
    pins = _pin_specs()
    assert pins, "no httpx2 pin found in pyproject.toml; SDK HTTP stack is unpinned"
    for spec in pins:
        version = _parse_spec(spec)
        assert version >= _SECURITY_FLOOR, (
            f"httpx2 pin {spec} is below the {_SECURITY_FLOOR} security floor; "
            "httpx2 < 2.13.0 re-introduces at least one published GHSA"
        )


def test_sdk_httpx_resolves_and_clears_security_floor() -> None:
    """The MCP SDK 2.0 transport binds to httpx2, not the legacy httpx.

    ``sdk_httpx()`` returns the module the SDK's streamable_http transport
    actually imports, or the newest httpx2/httpx available. If the
    transport fell back to legacy httpx (e.g. httpx2 missing on the path)
    the SDK's AsyncClient / OAuth Request objects would TypeError at the
    transport boundary.
    """
    pytest.importorskip("mcp", reason="MCP SDK extra not installed")
    from tools.mcp_tool import sdk_httpx

    resolved = sdk_httpx()
    assert resolved is not None, "sdk_httpx() returned None; MCP HTTP transport is unreachable"
    assert resolved.__name__ == "httpx2", (
        f"sdk_httpx() resolved to {resolved.__name__!r}; the MCP SDK 2.0 transport requires httpx2"
    )

    resolved_version = tuple(int(part) for part in resolved.__version__.split(".")[:3])
    assert resolved_version >= _SECURITY_FLOOR, (
        f"resolved httpx2 is {resolved.__version__}; below the {_SECURITY_FLOOR} security floor"
    )


def test_sdk_httpx_matches_declared_pin() -> None:
    """The pin in pyproject.toml must match the module the SDK actually imports.

    A mismatch means the lockfile drifted from the pin (the SDK was
    installed against one httpx2 and pyproject was edited to a different
    one), which is exactly the failure mode the original GHSA reports
    describe: the declared floor and the resolved module disagree.
    """
    pytest.importorskip("mcp", reason="MCP SDK extra not installed")
    from tools.mcp_tool import sdk_httpx

    pins = _pin_specs()
    assert pins, "no httpx2 pin found in pyproject.toml"
    pin_versions = {_parse_spec(spec) for spec in pins}
    assert len(pin_versions) == 1, (
        f"httpx2 pins disagree across extras: {sorted(pins)}; split pins re-open the floor"
    )

    resolved = sdk_httpx()
    assert resolved is not None, "sdk_httpx() returned None; SDK HTTP transport is unreachable"
    resolved_version = tuple(int(part) for part in resolved.__version__.split(".")[:3])

    declared = next(iter(pin_versions))
    assert resolved_version == declared, (
        f"resolved httpx2 {resolved.__version__} does not match declared pin {declared}; "
        "run `hermes pm lock` so the lockfile tracks pyproject.toml"
    )