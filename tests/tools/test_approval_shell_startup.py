"""Home startup inventory; env and sed grammar belong to sibling #99228."""

import pytest

from tools import approval, approval_detection


NEWLY_GATED = (
    ".zshenv", ".zlogin", ".zlogout", ".bash_aliases", ".bash_login",
    ".bash_logout", ".kshrc", ".cshrc", ".tcshrc", ".login",
)
# #85339's default inventory still omits the last five names above.
ALREADY_GATED = (".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile")
WRITE_VECTORS = (
    "echo x >> {p}", "echo x > {p}", "echo x | tee -a {p}",
    "cp placeholder {p}", "mv placeholder {p}", "install -m600 placeholder {p}",
    "sed -i 's/a/b/' {p}", "sed --in-place 's/a/b/' {p}",
)


@pytest.fixture(params=[approval.detect_dangerous_command, approval_detection.detect_dangerous_command],
                ids=["facade", "detection"])
def detector(request):
    return request.param


@pytest.fixture(params=["/root", "/home/security-test-user"])
def home(request, monkeypatch):
    monkeypatch.setenv("HOME", request.param)
    return request.param


@pytest.fixture(params=["~", "$HOME", "${HOME}"])
def prefix(request):
    return request.param


class TestShellStartupFileParity:
    def test_sibling_startup_files_are_gated(self, detector, home, prefix):
        for name in NEWLY_GATED:
            for vector in WRITE_VECTORS:
                command = vector.format(p=f"{prefix}/{name}")
                dangerous, key, reason = detector(command)
                assert dangerous and key and reason, command

    def test_original_five_still_gated(self, detector, home, prefix):
        for name in ALREADY_GATED:
            for vector in WRITE_VECTORS:
                command = vector.format(p=f"{prefix}/{name}")
                assert detector(command)[0], command

    def test_letter_continuation_near_misses_stay_safe(self, detector, home, prefix):
        # Preserve the inherited word-boundary contract; suffix punctuation
        # remains broad in the pre-existing non-sed patterns, not fixed here.
        for name in (".logins.json", ".zshenvrc-notes", ".kshrcx", ".loginx"):
            for vector in WRITE_VECTORS:
                command = vector.format(p=f"{prefix}/{name}")
                assert detector(command) == (False, None, None), command

    def test_unrelated_dotfiles_and_reads_stay_safe(self, detector, home, prefix):
        for name in (".gitconfig", ".vimrc", ".tmux.conf", ".inputrc"):
            for vector in WRITE_VECTORS:
                command = vector.format(p=f"{prefix}/{name}")
                assert detector(command) == (False, None, None), command
        for command in (
            "echo x >> /opt/app/.bashrc", "echo x >> ./project/.zshenv",
            f"cat {prefix}/.zshenv", f"grep alias {prefix}/.bash_aliases",
            f"sed -n '1,5p' {prefix}/.zshrc",
        ):
            assert detector(command) == (False, None, None), command

    def test_multicomponent_absolute_home(self, detector, monkeypatch):
        home = "/home/security-test-user"
        monkeypatch.setenv("HOME", home)
        for name in NEWLY_GATED + ALREADY_GATED:
            for vector in WRITE_VECTORS:
                assert detector(vector.format(p=f"{home}/{name}"))[0]
                assert detector(vector.format(p=f"/opt/app/{name}x")) == (False, None, None)
