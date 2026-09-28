"""Tests for tools/project_leak_scan.py — project-specific content detector.

The background skill-review fork writes GENERALIZABLE lessons into the shared
skill library; session instance detail (repo names, user paths, status files,
session ids, internal addresses, branch names) must not be distilled along.
"""

from tools.project_leak_scan import (
    LeakFinding,
    sanitize,
    scan,
    session_tokens,
)

# The six incident-shape tokens: each must be detected, and none may survive
# sanitize() in the content that would be written to a shared skill.
PROJECT = "myproject"
ABS_PATH = "/home/alice/myproject"
STATUS_FILE = "MYPROJECT_STATUS.json"
SESSION_ID = "9f3b2c1d-4e5a-4a7b-8c9d-0e1f2a3b4c5d"
INT_ADDR = "10.20.30.40"
BRANCH = "fix/123627-skill-review-distil"

LEAKY = (
    f"When working in {PROJECT}, check {ABS_PATH} and read {STATUS_FILE} "
    f"(session {SESSION_ID} on {INT_ADDR}, branch {BRANCH}) for state."
)


def test_detects_project_name():
    cats = {f.category for f in scan(LEAKY, project_tokens=(PROJECT,))}
    assert "project_name" in cats


def test_detects_absolute_user_path():
    cats = {f.category for f in scan(LEAKY)}
    assert "absolute_path" in cats


def test_detects_project_status_filename():
    cats = {f.category for f in scan(LEAKY)}
    assert "project_filename" in cats


def test_detects_session_id():
    cats = {f.category for f in scan(LEAKY)}
    assert "session_id" in cats


def test_detects_internal_address():
    cats = {f.category for f in scan(LEAKY)}
    assert "network_address" in cats


def test_detects_branch_name():
    cats = {f.category for f in scan(LEAKY)}
    assert "branch_or_worktree" in cats


def test_sanitize_removes_all_incident_tokens_keeps_lessons():
    lessons = (
        "workers should use separate worktrees. "
        "PASS requires effect verification. "
        "shared project state should be reconciled by a coordinator."
    )
    cleaned = sanitize(LEAKY + " " + lessons, project_tokens=(PROJECT,))
    for token in (PROJECT, ABS_PATH, STATUS_FILE, SESSION_ID, INT_ADDR, BRANCH):
        assert token not in cleaned
    assert lessons in cleaned


def test_domain_vocabulary_not_flagged():
    text = ("Convert STL and OBJ with trimesh before slicing to gcode. "
            "See ../references/mesh-notes.md for the /api/v1/jobs route.")
    assert scan(text) == []


def test_owner_machine_paths_not_flagged():
    text = ("Installed under C:\\Program Files\\Hermes\\bin and configured "
            "in ~/.hermes/skills/local.")
    assert scan(text) == []


def test_hash_and_encoding_names_not_flagged():
    assert scan("Verify the SHA-256 digest; the file is UTF-16.") == []


def test_documented_placeholders_not_flagged():
    text = ("Copy MY_STATUS.json, set YOUR_API_KEY (not EXAMPLE_TOKEN), keep "
            "foo and bar as-is, fill <your-value> and ... as needed.")
    assert scan(text) == []


def test_secret_never_echoed():
    cred = "sk-synth-Abc123Xyz789QrsTuv456"
    body = f"Authenticate with Bearer {cred} on every call."
    findings = [f for f in scan(body) if f.category == "secret"]
    assert findings, "expected the synthetic credential to be detected"
    for f in findings:
        assert cred not in f.redacted()
        assert cred[:8] not in f.redacted()
    cleaned = sanitize(body)
    assert cred not in cleaned
    assert cred[:8] not in cleaned


def test_deterministic_across_runs_and_token_order():
    a = scan(LEAKY, project_tokens=(PROJECT, "myproject-extra"))
    b = scan(LEAKY, project_tokens=("myproject-extra", PROJECT))
    assert [(f.category, f.span) for f in a] == [(f.category, f.span) for f in b]
    again = scan(LEAKY, project_tokens=(PROJECT, "myproject-extra"))
    assert [(f.category, f.span) for f in a] == [(f.category, f.span) for f in again]


def test_session_tokens_derive_repo_name():
    toks = session_tokens(cwd="/repo/myproject")
    assert "myproject" in toks
    # longest-first so alternation order cannot depend on set iteration
    assert list(toks) == sorted(set(toks), key=lambda t: (-len(t), t))


def test_finding_spans_do_not_overlap():
    findings = scan(LEAKY, project_tokens=(PROJECT,))
    spans = sorted(f.span for f in findings)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2
    assert all(isinstance(f, LeakFinding) for f in findings)


def test_concrete_test_values_flagged_placeholders_not():
    flagged = scan('Sign in as test_user="jdoe42" to exercise the flow.')
    assert "concrete_test_value" in {f.category for f in flagged}
    assert scan("Sign in with token=EXAMPLE_TOKEN in tests.") == []
