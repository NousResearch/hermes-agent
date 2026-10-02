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
    "git -C repo push -u origin feat/aios-f-vat-placeholder",
    "git -Crepo push --set-upstream origin feat/fix--force-placeholder",
    "git -C 'repo with spaces' push origin topic+suffix",
    "git -C push status --short",
    "git -c alias.note=push status",
    "git -C repo log --oneline origin +x",
    "printf '%s' 'git -C repo push --force origin +x'",
    "git commit -m 'describe git push origin +x without executing it'",
    "git push +remote main",
    "git push -- +remote main",
    "git push --repo +remote main",
    "git push --repo=+remote main",
    "git push --push-option +x origin main",
    "git push --push-option=+x origin main",
    "git push -o +x origin main",
    "git push -o+x origin main",
    "git push --receive-pack +x origin main",
    "git push --exec=+x origin main",
    "git push origin main && printf '%s' '+x'",
    "git push origin main # +x",
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
    "git push -fu origin main",
    "git -C repo push --force origin main",
    "git -C repo push -f origin main",
    "git -C repo push --force-with-lease origin main",
    "git -C repo push --force-with-lease=refs/heads/main:abc origin main",
    "git -C repo push -uf origin main",
    "git -C repo push -fu origin main",
    "git -Crepo push --force origin main",
    "git -C 'repo with spaces' push -f origin main",
    "git -C parent -C child push --force-with-lease origin main",
    "git -c core.hooksPath=hooks -C repo push --force origin main",
    "git --git-dir=repo/.git --work-tree repo push --force origin main",
    "git push origin +x",
    "git push origin +refs/heads/x:refs/heads/x",
    "git push origin '+refs/heads/x:refs/heads/x'",
    "git push origin main +topic:topic",
    "git push -- origin +x",
    "git push origin -- +x",
    "git push --repo origin +x",
    "git push --repo=origin +x",
    "git -C repo push origin +x",
    "git -Crepo push -u origin +refs/heads/x:refs/heads/x",
    "git push --push-option +data origin +x",
    "git status && git -C repo push origin +x",
    "printf '%s' \"$(git -C repo push origin +x)\"",
    "env /usr/bin/git -C repo push origin +x",
])
def test_real_force_flags_remain_behind_the_approval_gate(command):
    dangerous, key, description = detect_dangerous_command(command)
    assert dangerous
    assert key and description
    if description.startswith("git force push"):
        assert r"git\s+push" in _approval_key_aliases(key)
