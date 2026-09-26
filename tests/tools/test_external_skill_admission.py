import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import tools.external_skill_admission as gate


def _candidate_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    return h.hexdigest()


def _release(tmp_path: Path) -> tuple[Path, str]:
    archive = tmp_path / "gate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("src/skill_admission_gate/__main__.py", "# fixture\n")
        zf.writestr("config/local_windows.json", "{}")
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

def _config(mode: str, archive: Path, archive_sha: str, evidence_root: Path) -> dict:
    return {
        "skills": {
            "external_admission_gate": {
                "mode": mode,
                "python_executable": sys.executable,
                "release_archive": str(archive),
                "release_sha256": archive_sha,
                "evidence_root": str(evidence_root),
                "timeout_seconds": 30,
                "scanner_timeout_seconds": 5,
            }
        }
    }


def _fake_run_factory(candidate: Path, evidence_root: Path, decision: str = "BLOCK"):
    expected_hash = _candidate_hash(candidate)

    def _fake_run(cmd, **kwargs):
        evidence_dir = evidence_root / "run-1"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "decision": decision,
            "candidate_sha256": expected_hash,
            "evidence_dir": str(evidence_dir),
        }
        (evidence_dir / "decision.json").write_text(
            json.dumps({**payload, "policy_version": "skill-admission-gate-v0.1"}),
            encoding="utf-8",
        )
        exit_code = {"ALLOW_TO_LAB": 0, "QUARANTINE": 10, "BLOCK": 20}[decision]
        return subprocess.CompletedProcess(cmd, exit_code, stdout=json.dumps(payload), stderr="")

    return _fake_run


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    return skill


def test_off_mode_launches_no_external_process(monkeypatch, tmp_path):
    archive, archive_sha = _release(tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config("off", archive, archive_sha, tmp_path / "e"))
    launched = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: launched.append((a, k)))

    result = gate.run_external_admission(_skill(tmp_path), source_id="repo/demo")

    assert result.mode == "off"
    assert result.executed is False
    assert result.allow_continue is True
    assert launched == []


def test_shadow_runs_pinned_gate_but_never_blocks(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    archive, archive_sha = _release(tmp_path)
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("shadow", archive, archive_sha, evidence_root),
    )
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(skill, evidence_root, "BLOCK"))

    result = gate.run_external_admission(skill, source_id="repo/demo", source_version="v1")

    assert result.mode == "shadow"
    assert result.executed is True
    assert result.decision == "BLOCK"
    assert result.candidate_sha256 == _candidate_hash(skill)
    assert result.allow_continue is True
    assert result.error == ""
    assert Path(result.evidence_dir, "decision.json").is_file()


def test_enforce_would_stop_non_allow_decision(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    archive, archive_sha = _release(tmp_path)
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("enforce", archive, archive_sha, evidence_root),
    )
    monkeypatch.setattr(subprocess, "run", _verified_quarantine_run_factory(skill, evidence_root))

    result = gate.run_external_admission(skill, source_id="repo/demo")
    assert result.mode == "enforce"
    assert result.decision == "QUARANTINE"
    assert result.allow_continue is False


def test_candidate_hash_matches_standalone_manifest_contract(tmp_path):
    skill = _skill(tmp_path)
    assert gate._candidate_hash(skill) == _candidate_hash(skill)


def test_malformed_numeric_config_returns_shadow_error(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    cfg = {"skills": {"external_admission_gate": {
        "mode": "shadow", "timeout_seconds": "not-a-number"
    }}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    result = gate.run_external_admission(skill, source_id="repo/demo")

    assert result.mode == "shadow"
    assert result.decision == "ERROR"
    assert result.allow_continue is True
    assert "config" in result.error.lower()


def test_evidence_dir_must_stay_under_configured_root(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    archive, archive_sha = _release(tmp_path)
    evidence_root = tmp_path / "evidence"
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    expected_hash = _candidate_hash(skill)
    (outside / "decision.json").write_text(json.dumps({
        "decision": "ALLOW_TO_LAB", "candidate_sha256": expected_hash
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("shadow", archive, archive_sha, evidence_root),
    )

    def _fake_run(*_args, **_kwargs):
        payload = {
            "decision": "ALLOW_TO_LAB",
            "evidence_dir": str(outside),
            "candidate_sha256": expected_hash,
        }
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gate.run_external_admission(skill, source_id="repo/demo")

    assert result.decision == "ERROR"
    assert result.allow_continue is True
    assert "evidence" in result.error.lower()


def _verified_quarantine_run_factory(candidate: Path, evidence_root: Path):
    expected_hash = _candidate_hash(candidate)

    def _fake_run(cmd, **kwargs):
        evidence_dir = evidence_root / "run-binding"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        nvidia_raw = evidence_dir / "nvidia_raw.json"
        nvidia_raw.write_text(json.dumps({
            "analysis_completeness": {
                "status": "partial",
                "is_complete": False,
                "ledger_exceptions": [
                    {"reason_code": "reference_unresolved", "path": "tests/example.py"}
                ],
            }
        }), encoding="utf-8")
        scanners = [
            {
                "scanner": "hermes_skills_guard", "version": "skills-guard-v5",
                "complete": True, "execution_ok": True, "findings": [],
                "raw_report_path": str(evidence_dir / "hermes_raw.json"),
                "metadata": {}, "verdict": "safe",
            },
            {
                "scanner": "cisco_ai_defense", "version": "skill-scanner 2.1.0",
                "complete": True, "execution_ok": True, "findings": [],
                "raw_report_path": str(evidence_dir / "cisco_raw.json"),
                "metadata": {}, "verdict": "safe",
            },
            {
                "scanner": "nvidia_skillspector", "version": "2.11.2",
                "complete": False, "execution_ok": True, "findings": [],
                "raw_report_path": str(nvidia_raw),
                "metadata": {"analysis_status": "partial"}, "verdict": "CAUTION",
            },
        ]
        for name in ("hermes_raw.json", "cisco_raw.json"):
            (evidence_dir / name).write_text("{}", encoding="utf-8")
        decision = {
            "decision": "QUARANTINE", "candidate_sha256": expected_hash,
            "evidence_dir": str(evidence_dir),
            "policy_version": "skill-admission-gate-v0.1",
            "reasons": ["nvidia_skillspector:incomplete"],
            "scanners": scanners,
        }
        (evidence_dir / "decision.json").write_text(
            json.dumps(decision), encoding="utf-8"
        )
        payload = {
            "decision": "QUARANTINE",
            "candidate_sha256": expected_hash,
            "evidence_dir": str(evidence_dir),
        }
        return subprocess.CompletedProcess(
            cmd, 10, stdout=json.dumps(payload), stderr=""
        )

    return _fake_run


def test_verified_quarantine_returns_exact_approval_binding(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    archive, archive_sha = _release(tmp_path)
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("enforce", archive, archive_sha, evidence_root),
    )
    monkeypatch.setattr(
        subprocess, "run", _verified_quarantine_run_factory(skill, evidence_root)
    )

    result = gate.run_external_admission(
        skill, source_id="owner/repo/demo", source_version="v1"
    )
    binding = result.approval_binding
    assert binding is not None
    assert binding.decision == "QUARANTINE"
    assert binding.candidate_sha256 == _candidate_hash(skill)
    assert binding.source_id == "owner/repo/demo"
    assert binding.source_version == "v1"
    assert binding.gate_release_sha256 == archive_sha
    assert binding.policy_version == "skill-admission-gate-v0.1"
    assert binding.hermes_guard_version == "skills-guard-v5"
    assert binding.cisco_version == "skill-scanner 2.1.0"
    assert binding.nvidia_version == "2.11.2"
    assert ("nvidia_skillspector", False) in binding.scanner_completeness
    assert (
        "nvidia_skillspector", ("reference_unresolved",)
    ) in binding.scanner_reason_codes
    assert len(binding.evidence_digest) == 64


def test_non_quarantine_results_never_expose_approval_binding(monkeypatch, tmp_path):
    archive, archive_sha = _release(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("off", archive, archive_sha, tmp_path / "evidence"),
    )
    off = gate.run_external_admission(_skill(tmp_path), source_id="repo/demo")
    assert off.approval_binding is None


def test_block_and_error_results_have_no_approval_binding(monkeypatch, tmp_path):
    skill = _skill(tmp_path)
    archive, archive_sha = _release(tmp_path)
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config("shadow", archive, archive_sha, evidence_root),
    )
    monkeypatch.setattr(
        subprocess, "run", _fake_run_factory(skill, evidence_root, "BLOCK")
    )
    blocked = gate.run_external_admission(skill, source_id="repo/demo")
    assert blocked.decision == "BLOCK"
    assert blocked.approval_binding is None

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"skills": {"external_admission_gate": {
            "mode": "shadow", "timeout_seconds": "bad"
        }}},
    )
    errored = gate.run_external_admission(skill, source_id="repo/demo")
    assert errored.decision == "ERROR"
    assert errored.approval_binding is None
