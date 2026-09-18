"""t_2550b91f: warn-path redirect/flag-operand phantom suppression (re-land of
dc9df97cc6/dde1565f62 semantics). Pure-unit tests -- no tirith binary, no network."""
import sys

sys.path.insert(0, "/usr/local/lib/hermes-agent")

from tools.tirith_security import (  # noqa: E402
    _is_phantom_package_finding,
    _suppress_phantom_package_findings,
    _redirect_artifact_tokens,
    _flag_operand_tokens,
)


def _inc(pkg):
    return {"rule_id": "analysis_incomplete", "severity": "MEDIUM",
            "description": f"Tirith could not complete every configured runtime "
                           f"threat-intelligence check for package '{pkg}'"}


def _real(pkg):
    return {"rule_id": "analysis_incomplete", "severity": "MEDIUM",
            "description": f"threat-intelligence check for package '{pkg}' failed: deadline"}


def test_fd_phantom_only_warn_becomes_allow():
    cmd = "pip install requests 2>&1"
    findings = [_inc("2>&1"), _inc("2")]
    action, kept = _suppress_phantom_package_findings(cmd, "warn", findings)
    assert action == "allow", (action, kept)
    assert kept == []


def test_redirect_target_phantom_only_warn_becomes_allow():
    cmd = "pip install requests > /tmp/probe.log"
    findings = [_inc("/tmp/probe.log")]
    action, kept = _suppress_phantom_package_findings(cmd, "warn", findings)
    assert action == "allow"
    assert kept == []


def test_flag_operand_phantom_only_warn_becomes_allow():
    cmd = "uv pip install torch --index-strategy unsafe-best-match"
    findings = [_inc("unsafe-best-match")]
    action, kept = _suppress_phantom_package_findings(cmd, "warn", findings)
    assert action == "allow"
    assert kept == []


def test_real_package_warn_is_untouched():
    cmd = "pip install somepkg 2>&1"
    findings = [_real("somepkg")]
    action, kept = _suppress_phantom_package_findings(cmd, "warn", findings)
    assert action == "warn"
    assert kept == findings


def test_mixed_warn_keeps_real_findings_drops_phantoms():
    cmd = "npm ci 2>&1"
    real = _real("left-pad")
    phantom = _inc("2>&1")
    action, kept = _suppress_phantom_package_findings(cmd, "warn", [real, phantom])
    assert action == "warn"
    assert kept == [real]


def test_block_action_never_downgraded():
    cmd = "pip install requests 2>&1"
    findings = [_inc("2>&1")]
    action, kept = _suppress_phantom_package_findings(cmd, "block", findings)
    assert action == "block"
    assert kept == findings


def test_non_incomplete_findings_never_dropped():
    cmd = "pip install requests 2>&1"
    f = {"rule_id": "lookalike_tld", "severity": "MEDIUM", "description": "x.app"}
    action, kept = _suppress_phantom_package_findings(cmd, "warn", [f])
    assert action == "warn"
    assert kept == [f]


def test_phantom_classifier_requires_all_names_artifacts():
    cmd = "pip install requests 2>&1"
    artifacts = _redirect_artifact_tokens(cmd) | _flag_operand_tokens(cmd)
    assert artifacts == {"2", "2>&1"}
    assert _is_phantom_package_finding(_inc("2"), artifacts) is True
    assert _is_phantom_package_finding(_inc("requests"), artifacts) is False
    other_rule = {"rule_id": "lookalike_tld", "severity": "MEDIUM",
                  "description": "threat-intelligence check for package '2'"}
    assert _is_phantom_package_finding(other_rule, artifacts) is False  # wrong rule_id
    assert _is_phantom_package_finding("nope", artifacts) is False
