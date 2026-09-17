"""Documentation contracts for subscription-backed Gemini delegation routing."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "website" / "docs" / "user-guide" / "features"
ROUTING_DOC = DOCS / "gemini-delegation-routing.md"
SIDEBAR = REPO_ROOT / "website" / "sidebars.ts"


def _routing_doc() -> str:
    assert ROUTING_DOC.is_file(), "Gemini routing operator guide is missing"
    return ROUTING_DOC.read_text(encoding="utf-8")


def test_gemini_routing_guide_is_in_automation_navigation() -> None:
    sidebar = SIDEBAR.read_text(encoding="utf-8")

    assert "'user-guide/features/gemini-delegation-routing'" in sidebar
    assert sidebar.index("'user-guide/features/delegation'") < sidebar.index(
        "'user-guide/features/gemini-delegation-routing'"
    )


def test_guide_explains_route_and_authority_boundaries() -> None:
    doc = _routing_doc().lower()

    for contract in (
        "subscription-backed",
        "gemini-3.8-flash-low",
        "output-only",
        "no hermes tools",
        "route: sol",
        "data_classification: restricted",
        "any unknown classification",
        "fails closed to sol",
        "role: orchestrator",
        "short denylist",
        "no prose classifier",
        "final judgment",
        "side-effect authority",
        "falls back once",
    ):
        assert contract in doc


def test_guide_explains_configuration_and_activation_boundaries() -> None:
    doc = _routing_doc().lower()

    for contract in (
        "enabled: false",
        "profiles: []",
        "default_data_classification: restricted",
        "no gemini api-key fallback",
        "fresh cli or gateway session",
        "separate approval",
        "does not activate",
        "no live activation has occurred",
    ):
        assert contract in doc


def test_guide_explains_private_local_receipts() -> None:
    doc = _routing_doc().lower()

    for contract in (
        "$hermes_home/routing/gemini-routing.sqlite3",
        "mode `0600`",
        "restricted data",
        "never sent to gemini",
        "not stored in the routing receipt database",
        "failures and fallbacks",
        "terminal sol provider, model, and status",
        "metadata-only",
        "sha-256",
        "byte counts",
        "never stores prompt",
        "model output",
        "provider error text",
        "existing receipt rows",
        "raw task text",
        "slack alert",
    ):
        assert contract in doc


def test_guide_explains_daily_assurance_and_failure_only_alerting() -> None:
    doc = _routing_doc().lower()

    for contract in (
        "america/los_angeles",
        "min(5, eligible_count)",
        "five random",
        "one separate",
        "openai-codex/gpt-5.6-sol",
        "postdeployment operational assurance",
        "not a model-evaluation program",
        "no slack api call",
        "empty stdout",
        "one aggregate slack alert",
    ):
        assert contract in doc


def test_guide_rejects_predeployment_evaluation_as_a_launch_gate() -> None:
    doc = _routing_doc().lower()

    for contract in (
        "no shadow program",
        "no class qualification",
        "no benchmark threshold",
        "predeployment comparative evaluation",
    ):
        assert contract in doc


def test_existing_delegation_and_cron_guides_link_to_operator_guide() -> None:
    delegation = (DOCS / "delegation.md").read_text(encoding="utf-8").lower()
    cron = (DOCS / "cron.md").read_text(encoding="utf-8").lower()

    link = "./gemini-delegation-routing.md"
    assert link in delegation
    assert "output-only" in delegation
    assert "disabled by default" in delegation

    assert link in cron
    assert "gemini daily review" in cron
    assert "failure-only" in cron
