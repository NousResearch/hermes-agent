"""Global package-manager install detection.

A real incident motivated this: an agent asked to install a tool ran
`pip install <pkg>` / `npm install -g <pkg>` directly via the terminal tool
to satisfy the request, with neither action requiring approval -- despite
carrying the same "arbitrary code from a registry" risk class as the
existing `curl|bash` pattern two lines above these in DANGEROUS_PATTERNS.

An earlier version of this change also added patterns for standalone
script execution (`bash x.sh`, `./x`) to close a related gap: a script
downloaded in one tool call and run in the next carries no dangerous
keyword of its own. Review on #108236 (see NousResearch/hermes-agent#108235)
correctly identified that gap as unsolvable with a stateless per-command
regex -- `detect_dangerous_command()` only ever sees one command string at
a time, so it cannot distinguish a downloaded script from this project's
own documented entry points (`./scripts/run_tests.sh`, `./venv/bin/python`,
`bash tests/gateway/run_smoke.sh`, all of which the removed patterns
false-positived on). That half needs cross-call state (tracking files a
preceding curl/wget/write_file call produced) and is left for a follow-up;
this PR keeps only the package-install half, which is a clean, stateless
pattern-list fix.
"""

from tools.approval import detect_dangerous_command


class TestGlobalPackageInstall:
    def test_pip_install_arbitrary_package(self):
        is_dangerous, key, desc = detect_dangerous_command(
            "pip install agent-reach")
        assert is_dangerous is True
        assert key is not None
        assert "pip install" in desc

    def test_pip3_install(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "pip3 install some-package")
        assert is_dangerous is True
        assert "pip install" in desc

    def test_python_dash_m_pip_install(self):
        # "-m pip install" contains "pip install" as a substring, so the existing
        # \bpip3?\s+install\b pattern already matches this invocation style.
        is_dangerous, _, desc = detect_dangerous_command(
            "python -m pip install requests")
        assert is_dangerous is True
        assert "pip install" in desc

    def test_pip_install_upgrade_arbitrary_package(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "pip install --upgrade requests")
        assert is_dangerous is True
        assert "pip install" in desc

    def test_pip_install_editable_remote_repo(self):
        # Editable install of a REMOTE source still pulls arbitrary code; only a
        # local "." target is exempted below.
        is_dangerous, _, desc = detect_dangerous_command(
            "pip install -e git+https://example.com/some/repo.git")
        assert is_dangerous is True
        assert "pip install" in desc

    def test_pip_install_target_non_local_dir(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "pip install --target /opt/payload some-pkg")
        assert is_dangerous is True
        assert "pip install" in desc

    def test_npm_install_dash_g(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "npm install -g mcporter")
        assert is_dangerous is True
        assert "global npm install" in desc

    def test_npm_i_dash_g_short_alias(self):
        is_dangerous, _, desc = detect_dangerous_command("npm i -g some-cli")
        assert is_dangerous is True
        assert "global npm install" in desc

    def test_pipx_install(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "pipx install bilibili-cli")
        assert is_dangerous is True
        assert "pipx install" in desc

    def test_uv_tool_install(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "uv tool install skillevaluator")
        assert is_dangerous is True
        assert "uv tool install" in desc

    def test_uv_pip_install(self):
        is_dangerous, _, desc = detect_dangerous_command("uv pip install pandas")
        assert is_dangerous is True
        assert "pip install" in desc

    # -- negatives: project-scoped / self-contained installs stay routine ----

    def test_pip_install_dash_r_requirements_not_flagged(self):
        assert detect_dangerous_command(
            "pip install -r requirements.txt") == (False, None, None)

    def test_uv_pip_install_dash_r_not_flagged(self):
        assert detect_dangerous_command(
            "uv pip install -r requirements.txt") == (False, None, None)

    def test_pip_install_editable_local_project_not_flagged(self):
        assert detect_dangerous_command("pip install -e .") == (False, None, None)

    def test_pip_install_editable_long_flag_local_not_flagged(self):
        assert detect_dangerous_command(
            "pip install --editable .") == (False, None, None)

    def test_pip_install_target_local_dir_not_flagged(self):
        assert detect_dangerous_command(
            "pip install --target . some-pkg") == (False, None, None)

    def test_pip_install_target_local_relative_dir_not_flagged(self):
        assert detect_dangerous_command(
            "pip install --target ./vendor some-pkg") == (False, None, None)

    def test_pip_install_upgrade_pip_itself_not_flagged(self):
        assert detect_dangerous_command(
            "pip install --upgrade pip") == (False, None, None)

    def test_pip_install_dash_u_pip_itself_not_flagged(self):
        assert detect_dangerous_command("pip install -U pip") == (False, None, None)

    def test_npm_install_no_flag_not_flagged(self):
        assert detect_dangerous_command("npm install") == (False, None, None)

    def test_npm_install_named_local_package_not_flagged(self):
        assert detect_dangerous_command("npm install lodash") == (False, None, None)

    def test_npm_install_save_dev_not_flagged(self):
        assert detect_dangerous_command(
            "npm install --save-dev vitest") == (False, None, None)

    def test_uv_add_not_flagged(self):
        # `uv add` writes to the current project's pyproject.toml/lockfile -- project-scoped,
        # same category as `npm install <pkg>` with no -g, so deliberately not covered here.
        assert detect_dangerous_command("uv add requests") == (False, None, None)
