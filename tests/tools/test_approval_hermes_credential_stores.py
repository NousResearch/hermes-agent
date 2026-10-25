"""Hermes credential terminal mutation contracts (#100111).

Command strings are classified, never executed. Safe reads/copy-out here do not
claim that every store is covered by the independent read guard. Reordered sed
options and read-only --posix depend on the shared grammar correction (#99228);
assert them normally so integration cannot silently omit the dependency.
"""

from pathlib import Path

import pytest

from hermes_constants import get_default_hermes_root, get_hermes_home
from tools.approval import detect_dangerous_command


STORES = (
    "mcp-tokens/arbitrary/nested/token", "pairing/arbitrary/nested/device",
    ".anthropic_oauth.json", "mcp-tokens", "pairing",
)
SPELLINGS = ("~/.hermes", "$HOME/.hermes", "${HOME}/.hermes", "$HERMES_HOME", "${HERMES_HOME}", "home-absolute")
LAYOUTS = [("literal", spelling) for spelling in SPELLINGS] + [
    ("custom-root", "absolute"), ("custom-profile", "absolute"),
]


@pytest.mark.parametrize("home", ["/home/credential-test-user", "/root"])
@pytest.mark.parametrize("store", STORES)
@pytest.mark.parametrize("layout,spelling", LAYOUTS)
@pytest.mark.parametrize("vector", [
    "echo x >> {p}", "echo x > {p}", "echo x | tee -a {p}",
    "cp input {p} && true", 'mv input "{p}"; true',
    'install -m600 input "{p}" || true', "sed -i 's/a/b/' {p}",
    "sed -n -i 's/a/b/' {p}", "perl -pi 's/a/b/' {p}",
])
def test_hermes_credential_mutations_require_approval(monkeypatch, tmp_path, home, store, layout, spelling, vector):
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    if layout == "literal":
        prefix = f"{home}/.hermes" if spelling == "home-absolute" else spelling
        paths = [f"{prefix}/{store}"]
    else:
        # Real path getters, with A -> B -> A in one imported detector instance.
        paths = None
    for name in ("install-a", "install-b", "install-a") if paths is None else ("literal",):
        if paths is None:
            root = tmp_path / name / "root-store"
            active = root / "profiles" / "work" if layout == "custom-profile" else root
            monkeypatch.setenv("HERMES_HOME", str(active))
            assert get_hermes_home() == active
            # Scratch can nest beneath the real /root/.hermes. Keep the canonical
            # actual root in that case, not a detached synthetic root.
            native_root = Path(home) / ".hermes"
            expected_root = native_root if root.is_relative_to(native_root) else root
            assert get_default_hermes_root() == expected_root
            current_paths = [f"{base}/{store}" for base in (get_default_hermes_root(), get_hermes_home())]
        else:
            current_paths = paths
        for path in current_paths:
            command = vector.format(p=path)
            dangerous, key, reason = detect_dangerous_command(command)
            assert dangerous and key and reason, command


@pytest.mark.parametrize("home", ["/home/credential-test-user", "/root"])
@pytest.mark.parametrize("control", ["read-copy-out-near-miss", "sed-read-only"])
def test_hermes_credential_controls_stay_safe(monkeypatch, tmp_path, home, control):
    monkeypatch.setenv("HOME", home)
    for name in ("install-a", "install-b", "install-a"):
        active = tmp_path / name / "root-store" / "profiles" / "work"
        monkeypatch.setenv("HERMES_HOME", str(active))
        bases = ("~/.hermes", "$HERMES_HOME", str(get_default_hermes_root()), str(get_hermes_home()))
        for base in bases:
            commands = (
                f"echo x >> {base}/mcp-tokens-backup/token",
                f"echo x > {base}/pairing-notes/device",
                f"cp input {base}/mcp-tokens-backup && true",
                f"echo x > {base}/.anthropic_oauth.json.bak",
                f"echo x > {base}/.anthropic_oauth.json#backup",
                f"echo x > {base}/logs/agent.log",
                f"sed -i 's/a/b/' {base}/state.db",
                f"echo x >> {base}/sessions/state.db",
                f"cp input {base}/skills/mine/SKILL.md",
                f"install input {base}/cache/image.png",
                f"cp {base}/mcp-tokens/token /tmp/token-copy",
                f"cat {base}/pairing/device",
                f"grep -r token {base}/mcp-tokens/token",
                "echo x > /opt/app/mcp-tokens/token",
                "echo x > /opt/app/pairing/device",
            ) if control == "read-copy-out-near-miss" else (
                f"sed --posix 's/a/b/' {base}/mcp-tokens/token",
                f"sed -n 's/a/b/p' {base}/pairing/device",
            )
            for command in commands:
                assert detect_dangerous_command(command) == (False, None, None), command
        # A root/profiles basename alone is not a store: old installations must
        # cease matching after a live root/profile switch (no module reload).
        other = "install-b" if name == "install-a" else "install-a"
        old = tmp_path / other / "root-store" / "profiles" / "work"
        assert detect_dangerous_command(f"echo x > {old}/mcp-tokens/token") == (False, None, None)
