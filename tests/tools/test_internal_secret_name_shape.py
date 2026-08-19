"""Behavior contract for ``_is_hermes_internal_secret`` name-shape matching.

These tests assert a RELATION rather than a frozen list of suffixes:

    Under the ``AUXILIARY_`` and ``GATEWAY_RELAY_`` prefixes, any
    credential-shaped name is stripped from a child process environment,
    while non-secret routing hints and selectors stay visible.

That framing is deliberate. The predicate previously enumerated exact
suffixes (``_API_KEY`` / ``_BASE_URL`` for ``AUXILIARY_``; ``_SECRET`` /
``_KEY`` / ``_TOKEN`` for ``GATEWAY_RELAY_``), so every adjacent real spelling
of the same secret — ``…_APIKEY`` with no separator, ``…_SECRET`` under the
aux prefix, ``GATEWAY_RELAY_SECRET_V2`` with a trailing qualifier — fell
through to the child. A test that pins the suffix list would have passed
against that bug. A test that pins the relation fails against it.

Assertions run against a REAL child process (``sys.executable -c``) built
from each of the three spawn-env builders, because that is what a
model-authored shell command actually sees — not against the predicate.
"""

import json
import subprocess
import sys

import pytest

from tools.environments.local import (
    _make_run_env,
    _sanitize_subprocess_env,
    hermes_subprocess_env,
)

# Credential-shaped names under each internal prefix. Spelled several
# different ways on purpose: separator-less (``APIKEY``), bare word
# (``SECRET``), and qualifier-suffixed (``SECRET_V2``). All are the same
# secret class and must be stripped identically.
CREDENTIAL_SHAPED = [
    "AUXILIARY_VISION_API_KEY",
    "AUXILIARY_VISION_APIKEY",
    "AUXILIARY_VISION_SECRET",
    "AUXILIARY_VISION_TOKEN",
    "AUXILIARY_VISION_PASSWORD",
    "AUXILIARY_WEB_EXTRACT_API_KEY",
    "AUXILIARY_MY_PLUGIN_TASK_APIKEY",
    "GATEWAY_RELAY_SECRET",
    "GATEWAY_RELAY_SECRET_V2",
    "GATEWAY_RELAY_DELIVERY_KEY",
    "GATEWAY_RELAY_ENROLL_TOKEN",
    "GATEWAY_RELAY_IDP_CLIENT_SECRET",
    "GATEWAY_RELAY_PASSWORD",
]

# Non-secret names under the same prefixes. Routing hints and model
# selectors the docstring explicitly promises to keep visible — including
# the two real in-repo names that contain a credential-shaped word but are
# not credentials (``ROUTE_KEYS`` is a list of route ids;
# ``IDP_TOKEN_URL`` is an endpoint URL, not a token).
ROUTING_HINTS = [
    "AUXILIARY_VISION_MODEL",
    "AUXILIARY_VISION_PROVIDER",
    "GATEWAY_RELAY_URL",
    "GATEWAY_RELAY_PLATFORMS",
    "GATEWAY_RELAY_PLATFORM",
    "GATEWAY_RELAY_ENDPOINT",
    "GATEWAY_RELAY_ROUTE_KEYS",
    "GATEWAY_RELAY_IDP_TOKEN_URL",
]

_CANARY = "canary-value-{}"


def _plant(monkeypatch):
    for name in CREDENTIAL_SHAPED + ROUTING_HINTS:
        monkeypatch.setenv(name, _CANARY.format(name))


def _names_visible_to_child(env) -> set:
    """Spawn a real child with ``env`` and report which canaries it can read."""
    watched = CREDENTIAL_SHAPED + ROUTING_HINTS
    code = (
        "import os, json, sys;"
        "watched = json.loads(sys.argv[1]);"
        "print(json.dumps([n for n in watched if n in os.environ]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(watched)],
        env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    return set(json.loads(out.stdout))


def _builders(monkeypatch):
    import os

    return {
        "hermes_subprocess_env": lambda: hermes_subprocess_env(),
        "hermes_subprocess_env(inherit_credentials=True)":
            lambda: hermes_subprocess_env(inherit_credentials=True),
        "_sanitize_subprocess_env": lambda: _sanitize_subprocess_env(dict(os.environ)),
        "_make_run_env": lambda: _make_run_env(dict(os.environ)),
    }


@pytest.mark.parametrize("builder_name", [
    "hermes_subprocess_env",
    "hermes_subprocess_env(inherit_credentials=True)",
    "_sanitize_subprocess_env",
    "_make_run_env",
])
def test_credential_shaped_internal_names_never_reach_child(builder_name, monkeypatch):
    """No credential-shaped AUXILIARY_/GATEWAY_RELAY_ name reaches a child.

    The relation, not a suffix list: if a name sits under one of the two
    internal prefixes AND carries a credential-shaped word, the child must
    not be able to read it — on every spawn surface, including the
    ``inherit_credentials=True`` path a model-driving CLI gets.
    """
    _plant(monkeypatch)
    env = _builders(monkeypatch)[builder_name]()
    leaked = _names_visible_to_child(env) & set(CREDENTIAL_SHAPED)
    assert leaked == set(), (
        f"{builder_name} leaked credential-shaped internal names to a real "
        f"child process: {sorted(leaked)}"
    )


@pytest.mark.parametrize("builder_name", [
    "hermes_subprocess_env",
    "hermes_subprocess_env(inherit_credentials=True)",
    "_sanitize_subprocess_env",
    "_make_run_env",
])
def test_routing_hints_stay_visible_to_child(builder_name, monkeypatch):
    """The other half of the relation: routing hints must NOT be over-blocked.

    A shape test that ate ``GATEWAY_RELAY_URL`` / ``GATEWAY_RELAY_ROUTE_KEYS``
    would break relay routing, so the contract is two-sided.
    """
    _plant(monkeypatch)
    env = _builders(monkeypatch)[builder_name]()
    visible = _names_visible_to_child(env)
    missing = set(ROUTING_HINTS) - visible
    assert missing == set(), (
        f"{builder_name} over-blocked non-secret routing hints: {sorted(missing)}"
    )


def test_spelling_variants_of_one_secret_are_classified_alike(monkeypatch):
    """Separator/qualifier spelling must not change the verdict.

    This is the anti-regression guard for the enumeration approach: a future
    edit that reintroduces an exact-suffix list will keep the canonical
    spelling stripped but let a variant of the same secret through, and this
    test fails on the disagreement rather than on a hard-coded name.
    """
    variants = {
        "aux api key": [
            "AUXILIARY_VISION_API_KEY",
            "AUXILIARY_VISION_APIKEY",
        ],
        "relay secret": [
            "GATEWAY_RELAY_SECRET",
            "GATEWAY_RELAY_SECRET_V2",
        ],
    }
    for name in [n for group in variants.values() for n in group]:
        monkeypatch.setenv(name, _CANARY.format(name))

    visible = _names_visible_to_child(hermes_subprocess_env())
    for label, group in variants.items():
        verdicts = {name: name in visible for name in group}
        assert len(set(verdicts.values())) == 1, (
            f"spelling variants of the same {label} were classified "
            f"differently: {verdicts}"
        )
        assert not any(verdicts.values()), (
            f"{label} variants reached the child: {verdicts}"
        )


def test_unknown_internal_credential_name_fails_closed(monkeypatch):
    """A name nobody enumerated must still be stripped.

    The predicate is a shape test, so a credential-shaped name invented after
    this test was written is covered without editing any list.
    """
    invented = "GATEWAY_RELAY_FUTURE_HANDSHAKE_KEY_V9"
    monkeypatch.setenv(invented, "canary")
    code = (
        "import os, sys; "
        "print('LEAKED' if sys.argv[1] in os.environ else 'ABSENT')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, invented],
        env=hermes_subprocess_env(), capture_output=True, text=True,
        timeout=60, check=True,
    )
    assert out.stdout.strip() == "ABSENT"
