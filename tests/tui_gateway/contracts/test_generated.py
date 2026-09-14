"""Runtime catalog and generated-contract invariants.

Events need no source completeness scan: ``server._event_frame`` only accepts a ``Payload`` instance,
so an undeclared event has no payload class to construct and fails with ``TypeError`` at that boundary.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "scripts" / "gen_gateway_contracts.py"
META_SCHEMA = REPO / "tests" / "tui_gateway" / "contracts" / "fixtures" / "openrpc-meta-schema.json"
PYTHON = REPO / ".venv" / "bin" / "python"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_gateway_contracts_generated", GEN)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator(*args: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [str(PYTHON), str(GEN), *args],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_registered_methods_match_contract_catalog():
    from tui_gateway import server
    from tui_gateway.contracts.registry import METHODS

    assert set(server._methods) == set(METHODS)


def test_openrpc_validates_against_vendored_meta_schema(gen):
    schema = json.loads(META_SCHEMA.read_text(encoding="utf-8"))
    document = json.loads(gen.render_openrpc())

    assert list(Draft7Validator(schema).iter_errors(document)) == []


def test_rendering_is_deterministic(gen):
    first = tuple(gen.render_all(Path("/tmp/pr3-determinism-one")).values())
    second = tuple(gen.render_all(Path("/tmp/pr3-determinism-two")).values())

    assert first == second


def test_check_detects_stale_output_in_an_explicit_output_directory(tmp_path):
    result = _generator("--out-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr

    fresh = _generator("--check", "--out-dir", str(tmp_path))
    assert fresh.returncode == 0, fresh.stderr

    generated = tmp_path / "gateway-contract.generated.ts"
    generated.write_text(generated.read_text(encoding="utf-8") + " ", encoding="utf-8")
    stale = _generator("--check", "--out-dir", str(tmp_path))
    assert stale.returncode == 1
    assert "gateway-contract.generated.ts" in stale.stderr
