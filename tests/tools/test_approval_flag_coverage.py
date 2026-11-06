"""CR-2026-10-21-005: flagger coverage — state-changing commands must flag, read-only must not.

Pre-patch audit (2026-10-21) proved these passed UNFLAGGED under approvals.mode=manual:
plain `git push` (all forms), `docker run --privileged`, `docker exec -u 0/root`,
`systemctl enable`. The read-only controls guard Admin's audit tooling
(grep/stat/sha256sum/find/sed/ls/cat) — they must never flag.
"""
from tools.approval import detect_dangerous_command


class TestFlagCoverageCR005:
    STATE_CHANGING = [
        "git push origin main",
        "git push",
        "git push -u origin main",
        "git -C /srv/repo push origin main",
        "docker run --privileged --pid=host alpine id",
        "docker exec -u 0 unified-api-1 id",
        "docker exec -u root unified-web-1 sh",
        "docker exec --user=root unified-web-1 sh",
        "systemctl enable unified-api",
        "systemctl --now enable unified-api",
    ]

    READ_ONLY = [
        "grep -rn foo /etc/passwd",
        'grep -rn "git push" /usr/local/lib/hermes-agent/tools',
        'grep -n "systemctl enable" approval_detection.py',
        "stat /etc/hosts",
        "sha256sum /etc/hosts",
        "ls -la /root/.hermes",
        "find /root/.hermes/skills -name '*.md'",
        "sed -n '1,5p' /etc/hosts",
        "git log --oneline -5",
        "git status --porcelain",
        "docker ps",
        "systemctl status unified-api",
        "cat /etc/hosts",
    ]

    def test_state_changing_flagged(self):
        for cmd in self.STATE_CHANGING:
            is_dangerous, _key, desc = detect_dangerous_command(cmd)
            assert is_dangerous, f"not flagged (gap reopened): {cmd}"
            assert desc, f"flagged but no description: {cmd}"

    def test_read_only_not_flagged(self):
        for cmd in self.READ_ONLY:
            got = detect_dangerous_command(cmd)
            assert got == (False, None, None), f"read-only wrongly flagged: {cmd} -> {got}"

    def test_force_push_keeps_specific_desc(self):
        _, _, desc = detect_dangerous_command("git push --force origin main")
        assert desc and "force" in desc, f"force push lost specific desc: {desc!r}"

    def test_privileged_false_carveout(self):
        got = detect_dangerous_command("docker run --privileged=false alpine id")
        assert got == (False, None, None), f"=false should not flag: {got}"

    def test_non_root_exec_not_flagged(self):
        for cmd in ("docker exec -u 1000 app id", "docker exec -u appuser app id"):
            got = detect_dangerous_command(cmd)
            assert got == (False, None, None), f"non-root exec wrongly flagged: {cmd} -> {got}"
