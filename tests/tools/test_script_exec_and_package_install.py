"""Standalone script execution and global package-manager install detection.

Two real incidents in the same week motivated this: an agent asked to
install a tool (a) split a `curl | bash` installer into a download call
followed by a separate `bash /tmp/x.sh` call in the NEXT turn, and (b)
separately ran `pip install <pkg>` / `npm install -g <pkg>` to satisfy a
request, with neither action requiring approval.

(a) is a real gap in the existing curl|sh pattern: it can only see one
command at a time, and a script downloaded in an earlier call carries no
dangerous keyword when it is later handed to an interpreter or run
directly. `chmod +x x.sh && ./x.sh` on one line is already caught; the
standalone forms (no chmod, or download and exec as separate turns) are
not.

(b) is not a bypass of any existing pattern -- installing new software via
a package manager was simply never covered, even though it carries the
same "arbitrary code from a registry" risk class as the curl|bash pattern
two lines above it in DANGEROUS_PATTERNS.
"""

from tools.approval import detect_dangerous_command


class TestScriptExecStandalone:
    def test_bash_on_downloaded_script(self):
        is_dangerous, key, desc = detect_dangerous_command(
            "bash /tmp/install_parallel.sh")
        assert is_dangerous is True
        assert key is not None
        assert "script file" in desc

    def test_sh_on_script_with_flags(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "sh -e /opt/hermes/install_parallel.sh")
        assert is_dangerous is True
        assert "script file" in desc

    def test_relative_dot_slash_execution(self):
        is_dangerous, _, desc = detect_dangerous_command("./install.sh")
        assert is_dangerous is True
        assert "relative-path file" in desc

    def test_chained_relative_execution_still_caught(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "chmod +x install.sh && ./install.sh")
        assert is_dangerous is True

    # -- negatives ------------------------------------------------------

    def test_bash_dash_c_not_flagged_by_this_rule(self):
        # Already covered elsewhere (script-execution-via-flag); not a .sh file argument.
        is_dangerous, _, desc = detect_dangerous_command("bash -c 'echo hi'")
        if is_dangerous:
            assert "script file" not in desc

    def test_plain_prose_mentioning_sh_not_flagged(self):
        assert detect_dangerous_command(
            "echo 'see setup.sh for details'") == (False, None, None)


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

    # -- negatives: project-scoped installs stay routine -----------------

    def test_pip_install_dash_r_requirements_not_flagged(self):
        assert detect_dangerous_command(
            "pip install -r requirements.txt") == (False, None, None)

    def test_npm_install_no_flag_not_flagged(self):
        assert detect_dangerous_command("npm install") == (False, None, None)

    def test_npm_install_named_local_package_not_flagged(self):
        assert detect_dangerous_command("npm install lodash") == (False, None, None)

    def test_npm_install_save_dev_not_flagged(self):
        assert detect_dangerous_command(
            "npm install --save-dev vitest") == (False, None, None)

    def test_uv_pip_install_dash_r_not_flagged(self):
        assert detect_dangerous_command(
            "uv pip install -r requirements.txt") == (False, None, None)
