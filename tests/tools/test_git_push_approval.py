"""Force-push approval follows shell argument boundaries, not substrings in branch names."""

import pytest

from tools.approval_detection import _approval_key_aliases, detect_dangerous_command


@pytest.mark.parametrize("command", [
    "git push -u origin feat/aios-f-vat-placeholder",
    "git status --porcelain && git add docs/evidence/t_10bcf319/report.md && git diff --cached --check && git commit -m 'docs: 留存 AIOS F 逐條驗收與既有分支衝突證據' && git fetch origin && git rebase origin/main && git diff origin/main --stat && git diff origin/main --check && git status --porcelain && git push -u origin feat/aios-f-vat-placeholder",
    "git push -u origin feat/fix--force-placeholder",
    "git push -u origin 'feat/aios-f-vat-placeholder'",
    "git push --set-upstream origin feat/aios-f-vat-placeholder",
    "git push origin main && printf '%s' '-f'",
    "git push origin main; printf '%s' '--force'",
    "git commit -m 'describe git push -f without executing it'",
    "printf '%s' 'git push --force'",
    "git push --push-option '-f' origin main",
    "git push -oforce=false origin main",
    "git push -- origin --force",
])
def test_branch_names_and_quoted_prose_do_not_request_force_approval(command):
    assert detect_dangerous_command(command) == (False, None, None)


@pytest.mark.parametrize("command", [
    "git push -f origin main",
    "git push --force origin main",
    "git push --force-with-lease origin main",
    "git push --force-with-lease=refs/heads/main:abc origin main",
    "git push --forc origin main",
    "git push '-f' origin main",
    'git push "--force" origin main',
    "git push --fo''rce origin main",
    "git push -uf origin main",
    "git push origin main -f",
    "git status && git push -f origin main",
    "printf '%s' \"$(git push -f origin main)\"",
    "/usr/bin/git push -f origin main",
    "env git push --force origin main",
    "git push${IFS}--force origin main",
    "bash -c 'git push -f origin main'",
])
def test_real_force_flags_remain_behind_the_approval_gate(command):
    dangerous, key, description = detect_dangerous_command(command)
    assert dangerous
    assert key and description
    if description.startswith("git force push"):
        assert r"git\s+push" in _approval_key_aliases(key)
