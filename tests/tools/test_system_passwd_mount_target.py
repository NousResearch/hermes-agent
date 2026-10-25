"""Regression for #132440: sandbox mount targets are not critical host reads."""

from __future__ import annotations

from tools.skills_guard import scan_file, scan_skill, should_allow_install


class TestSystemPasswdMountTargetDemotion:
    def test_bwrap_ro_bind_target_is_confirmable_not_quarantined(self, tmp_path):
        skill_dir = tmp_path / "sandbox"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: sandbox\n---\nRun `python contained.py`.\n", encoding="utf-8"
        )
        (skill_dir / "contained.py").write_text(
            'identity_binds = ["--ro-bind", str(identity / "passwd"), "/etc/passwd",\n'
            '                  "--ro-bind", str(identity / "group"), "/etc/group"]\n',
            encoding="utf-8",
        )
        result = scan_skill(skill_dir, source="community")
        passwd = [fi for fi in result.findings if fi.pattern_id == "system_passwd_access"]
        assert passwd, "expected system_passwd_access finding for the mount target"
        assert all(fi.severity == "high" for fi in passwd), [
            (fi.severity, fi.description) for fi in passwd
        ]
        assert result.verdict == "caution"
        assert should_allow_install(result, force=True)[0] is True

    def test_docker_volume_target_is_confirmable(self, tmp_path):
        f = tmp_path / "run.sh"
        f.write_text(
            'docker run -v "$WORKDIR/passwd:/etc/passwd:ro" image\n',
            encoding="utf-8",
        )
        findings = scan_file(f, "run.sh")
        passwd = [fi for fi in findings if fi.pattern_id == "system_passwd_access"]
        assert passwd and all(fi.severity == "high" for fi in passwd)

    def test_host_read_of_passwd_stays_critical(self, tmp_path):
        f = tmp_path / "evil.py"
        f.write_text('users = open("/etc/passwd").read()\n', encoding="utf-8")
        findings = scan_file(f, "evil.py")
        passwd = [fi for fi in findings if fi.pattern_id == "system_passwd_access"]
        assert passwd and all(fi.severity == "critical" for fi in passwd)

    def test_bind_plus_action_on_same_line_stays_critical(self, tmp_path):
        f = tmp_path / "evil.sh"
        f.write_text(
            'bwrap --ro-bind /tmp/p /etc/passwd -- cat /etc/passwd\n',
            encoding="utf-8",
        )
        findings = scan_file(f, "evil.sh")
        passwd = [fi for fi in findings if fi.pattern_id == "system_passwd_access"]
        assert passwd and all(fi.severity == "critical" for fi in passwd)
