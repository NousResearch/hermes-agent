"""Tests for which browser tabs the vault is willing to autofill into.

The vault picks a target tab by evaluating a per-kind JS probe against each open page and
taking the first one that answers truthy. That probe used to ask only whether a matching
control existed anywhere in the DOM, so a control parked in a hidden container was enough
to win the tab: a stale tab could be chosen over the one the user was actually signing in
on, and the fill then either found nothing to fill or reported an origin mismatch against a
site that had nothing to do with the login.

The probes now have to tell a form the user can see from one they cannot. These tests pin
that decision per kind (login, payment, address, one-time code) rather than on one string,
so dropping a kind back to the old bare query leaves a failure behind.

There is no layout engine under a plain JS stub, so each scenario states the computed style
and geometry of the control explicitly instead of measuring it. That is what lets a scenario
be rejected by exactly one term of the probe, which in turn is what shows the three terms
are each load-bearing: a control hidden by its own `display` is caught by the style lookup
alone; a control hidden by an ancestor still reports its own display as `inline-block` in a
real browser and is caught only by the zero-size rect; a control parked off-screen has
ordinary styles and ordinary size and needs the geometry.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools import browser_vault_tool  # noqa: E402


KINDS = ("login", "payment", "address", "otp")

_STUB = r"""
// Stands in for the page the probe runs against. The geometry and computed style of the
// single control under test are handed in explicitly, so a scenario can be arranged to fail
// exactly one term of the probe instead of all of them at once.
const probe = process.argv[2];
const computed = JSON.parse(process.argv[3]);
const rect = JSON.parse(process.argv[4]);
const present = process.argv[5] === "present";

// The selector the probe asks for is read back out of the probe itself, so a tab only
// yields its control to the probe if the probe is querying the kind's real selector rather
// than something that matches everything.
const asked = probe.match(/querySelectorAll\((['"])([\s\S]*?)\1\)/);
if (!asked) {
  process.stdout.write(JSON.stringify({ error: "probe does not query a fixed selector" }));
  process.exit(0);
}
const selector = asked[2];

function control() {
  return {
    tagName: "INPUT",
    getBoundingClientRect: () => ({
      x: 0, y: 0, top: 0, left: 0,
      right: rect.width, bottom: rect.height,
      width: rect.width, height: rect.height,
    }),
  };
}

const controls = present ? [control()] : [];

globalThis.window = { getComputedStyle: () => computed };
globalThis.document = { querySelectorAll: (sel) => (sel === selector ? controls : []) };

let value;
try {
  value = eval(probe);
} catch (err) {
  process.stdout.write(JSON.stringify({ error: String(err) }));
  process.exit(0);
}
process.stdout.write(JSON.stringify({ value: Boolean(value) }));
"""

# scenario name -> (computed style, geometry, control present, may the vault fill it)
SCENARIOS = {
    "visible_form": (
        {"display": "inline-block", "visibility": "visible"},
        {"width": 260, "height": 32},
        True,
        True,
    ),
    "own_display_none": (
        {"display": "none", "visibility": "visible"},
        {"width": 260, "height": 32},
        True,
        False,
    ),
    # A control inside display:none still reports its own display as inline-block in a real
    # browser, so only the zero-size rect rejects it. This is the shape that caused the report.
    "hidden_by_ancestor": (
        {"display": "inline-block", "visibility": "visible"},
        {"width": 0, "height": 0},
        True,
        False,
    ),
    "own_visibility_hidden": (
        {"display": "inline-block", "visibility": "hidden"},
        {"width": 260, "height": 32},
        True,
        False,
    ),
}

# A control parked outside the viewport still has ordinary computed style and ordinary size,
# so every term the probe reads says "fillable" and the probe accepts it. Recorded as a known
# gap rather than asserted as correct: a field the user cannot see is still a field the vault
# will type into. Strict=False so the day the probe starts consulting the viewport position
# this test flips to XPASS and gets looked at, instead of quietly pinning the gap in place.
PARKED_OFF_SCREEN = (
    {"display": "inline-block", "visibility": "visible"},
    {"width": 260, "height": 32},
)


@pytest.fixture(scope="module")
def stub_script(tmp_path_factory):
    """Materialise the Node stub once; the test needs no repo-level JS fixture."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires an already-installed Node executable")
    path = tmp_path_factory.mktemp("tab-probe") / "stub.js"
    path.write_text(_STUB)
    return path


def _run_probe(stub_script, probe, computed, rect, present=True):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires an already-installed Node executable")
    out = subprocess.run(
        [
            node,
            str(stub_script),
            probe,
            json.dumps(computed),
            json.dumps(rect),
            "present" if present else "absent",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert "error" not in payload, payload["error"]
    return payload["value"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_has_a_probe(kind):
    assert kind in browser_vault_tool._TAB_PROBES, (
        f"{kind} has no tab probe, so the vault cannot pick a target tab for it"
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_hidden_controls_never_win_the_tab(stub_script, kind, scenario):
    computed, rect, present, fillable = SCENARIOS[scenario]
    probe = browser_vault_tool._TAB_PROBES[kind]
    got = _run_probe(stub_script, probe, computed, rect, present=present)
    assert got is fillable, f"{kind}/{scenario}: probe answered {got}, expected {fillable}"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.xfail(strict=False, reason="known gap: a control parked outside the viewport "
                                         "still has ordinary style and size, so the probe accepts it")
def test_field_outside_the_viewport_is_still_selected(stub_script, kind):
    computed, rect = PARKED_OFF_SCREEN
    got = _run_probe(stub_script, browser_vault_tool._TAB_PROBES[kind], computed, rect)
    assert got is False, (
        f"{kind}: probe now rejects a field outside the viewport - tighten this xfail into "
        f"a real assertion and drop the reason"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_tab_without_a_matching_form_is_not_selected(stub_script, kind):
    probe = browser_vault_tool._TAB_PROBES[kind]
    got = _run_probe(stub_script, probe, {"display": "block", "visibility": "visible"},
                     {"width": 0, "height": 0}, present=False)
    assert got is False, f"{kind}: a tab with no matching control answered truthy"


@pytest.mark.parametrize("kind", KINDS)
def test_probe_reads_style_and_geometry(kind):
    """A probe that only asks whether a control exists can pass the table above by luck of
    the fixture, so also assert it reads the signals it is supposed to read."""
    probe = browser_vault_tool._TAB_PROBES[kind]
    assert "querySelector('input" not in probe and 'querySelector("input' not in probe, (
        f"{kind}: probe is back to a bare existence check"
    )
    assert "getComputedStyle" in probe, f"{kind}: probe ignores computed style"
    assert "getBoundingClientRect" in probe, f"{kind}: probe ignores geometry"