"""Project env write contracts through both public detector import surfaces."""

import re

import pytest

from tools import approval, approval_detection


@pytest.fixture(params=[approval.detect_dangerous_command, approval_detection.detect_dangerous_command],
                ids=["facade", "detection"])
def detector(request, monkeypatch):
    # Identity comparisons do not create files or execute the classified commands.
    return request.param


@pytest.fixture(params=["/root", "/home/security-test-user"])
def home(request, monkeypatch):
    monkeypatch.setenv("HOME", request.param)
    return request.param


class TestEnvrcAndProjectEnvInPlaceEdits:
    @pytest.mark.parametrize("ifs", ["${IFS}", "$IFS"])
    @pytest.mark.parametrize("carrier", ["$({body})", "`{body}`", '"$({body})"', '"`{body}`'])
    @pytest.mark.parametrize("target", ["/etc/hosts", "~/.hermes/config.yaml", "~/.bashrc", "app/.envrc"])
    @pytest.mark.parametrize("depth", [1, 8])
    def test_sed_ifs_in_executable_substitution(self, detector, ifs, carrier, target, depth):
        body = f"sed{ifs}-i s/a/b/ {target}"
        for _ in range(depth - 1):
            body = 'printf \'%s\' "$(' + body + ')"'
        command = "printf '%s' " + carrier.format(body=body)
        dangerous, key, reason = detector(command)
        assert dangerous and key and reason, (command, (dangerous, key, reason))

    @pytest.mark.parametrize("body", [
        "sed${IFS}-n s/a/b/ ~/.hermes/config.yaml",
        "sed$IFS-n s/a/b/ /etc/hosts",
        "sed${IFS}-i s/a/b/ README.md",
        "sed$IFS-i s/a/b/ ~/.bashrc.example",
        "sed${IFS}-i s/a/b/ app/.envrc.example",
        "sed${IFS}-i s/a/b/ ~/.hermes/config.yaml.example",
        "sed'${IFS}'-i s/a/b/ ~/.hermes/config.yaml",
        'sed"${IFS}"-i s/a/b/ ~/.hermes/config.yaml',
        "sed'$IFS'-i s/a/b/ ~/.bashrc",
        'sed"$IFS"-i s/a/b/ ~/.bashrc',
        r"sed\${IFS}-i s/a/b/ /etc/hosts",
        r"sed\$IFS-i s/a/b/ app/.envrc",
        "sed${IFS_OTHER}-i s/a/b/ ~/.bashrc",
        "sed$IFS_OTHER-i s/a/b/ app/.envrc",
        "printf '%s' 'sed${IFS}-i s/a/b/ ~/.bashrc'",
        'printf \'%s\' "sed${IFS}-i s/a/b/ app/.envrc"',
        "printf '%s' '$(sed${IFS}-i s/a/b/ /etc/hosts)'",
        "printf '%s' '`sed$IFS-i s/a/b/ ~/.hermes/config.yaml`'",
    ])
    @pytest.mark.parametrize("carrier", ["$({body})", "`{body}`", '"$({body})"', '"`{body}`'])
    def test_sed_ifs_substitution_data_stays_safe(self, detector, body, carrier):
        command = "printf '%s' " + carrier.format(body=body)
        assert detector(command) == (False, None, None), command

    @pytest.mark.parametrize("spelling", ["spaced-posix", "native-drive", "native-unc"])
    def test_sed_home_operands_preserve_word_identity(self, detector, tmp_path, monkeypatch, spelling):
        native_separators = spelling != "spaced-posix"
        # Native path data must have a real drive or UNC root, not a POSIX
        # tmp_path with its slashes replaced (which spells shell escapes).
        if spelling == "native-drive":
            user_home, hermes_home = r"C:\Users\sed-user", r"C:\Hermes\sed-profile"
        elif spelling == "native-unc":
            user_home, hermes_home = r"\\server\share\sed-user", r"\\server\share\sed-profile"
        else:
            user_home = str(tmp_path / "user home")
            hermes_home = str(tmp_path / "hermes home")
        monkeypatch.setenv("HOME", user_home)
        monkeypatch.setenv("HERMES_HOME", hermes_home)
        for path in [user_home + "/.bashrc", hermes_home + "/config.yaml"]:
            if native_separators:
                path = path.replace("/", "\\")
            # These are classifier inputs, not commands executed on this host.
            operand = path if native_separators else f'"{path}"'
            result = detector(f"sed -ni 's/a/b/' {operand}")
            assert result[0] and result[1] and result[2], (operand, result)
            assert detector(f"sed -n 's/a/b/' {operand}") == (False, None, None)
            assert detector(f"sed -i -e {operand} README.md") == (False, None, None)
            assert detector(f"sed -ni 's/a/b/' \"{path}.example\"") == (False, None, None)

    def test_sed_driveless_backslashes_do_not_identify_posix_home(self, detector, tmp_path, monkeypatch):
        user_home = str(tmp_path / "user-home")
        monkeypatch.setenv("HOME", user_home)
        operand = (user_home + "/.bashrc").replace("/", "\\")
        assert detector(f"sed -ni 's/a/b/' {operand}") == (False, None, None)
        assert detector(f"sed -ni 's/a/b/' '{user_home}/.bashrc'")[0]

    def test_sed_inventory_slash_marks_subtree_not_filename_prefix(self, detector, monkeypatch):
        monkeypatch.setenv("HOME", "/home/sed-subtree-user")
        # Extend only this test's inventory, without importing another policy's names.
        inventory = approval_detection._SED_USER_TARGET_RE
        monkeypatch.setattr(approval_detection, "_SED_USER_TARGET_RE", re.compile(
            # Match the actual credential inventory's bare-root alternative:
            # HOME lexical normalization removes a terminal slash.
            rf"(?:{inventory.pattern}|~/\.sed-test-subtree(?:/|$))", inventory.flags,
        ))
        for path, expected in [
            ("~/.sed-test-subtree/", True),
            ("~/.sed-test-subtree", True),
            ("~/.sed-test-subtree/./", True),
            ("~/.sed-test-subtree/token", True),
            ("~/.sed-test-subtree/nested/token with spaces", True),
            ("~/.sed-test-subtree.example/token", False),
            ("~/.sed-test-subtree/../notes/token", False),
            ("~/.bashrc", True),
            ("~/.bashrc.example", False),
            ("~/.bashrc/child", False),
        ]:
            operand = f'"{path}"'
            result = detector(f"sed -ni 's/a/b/' {operand}")
            assert result[0] is expected, (path, result)
            assert bool(result[1]) is expected
            assert detector(f"sed -n 's/a/b/' {operand}") == (False, None, None)
            assert detector(f"sed -i -e {operand} README.md") == (False, None, None)

    @pytest.mark.parametrize("path", [".env", "app/.envrc", "~/.bashrc", "~/.hermes/config.yaml", "/etc/security-test"])
    @pytest.mark.parametrize("flag,expected", [("-i", True), ("--in-place", True), ("--posix", False), ("-n", False)])
    def test_sed_in_place_on_project_env_is_gated(self, detector, home, path, flag, expected):
        result = detector(f"sed {flag} 's/a/b/' {path}")
        assert result[0] is expected, result
        assert bool(result[1]) is expected

    @pytest.mark.parametrize("command,expected", [
        ("sed -n -i 's/a/b/' .env", True),
        ("sed -n --in-place 's/a/b/' .env", True),
        ("sed -e 's/a/b/' -i app/.envrc", True),
        ("sed -i -e 's/a/b/' .env", True),
        ("sed -i -f program.sed .env", True),
        ("sed -i --expression='s/a/b/' .env", True),
        ("sed --file=program.sed --in-place=.bak .env", True),
        ("sed -ne's/a/b/' -i.bak .env", True),
        ("sed -ni 's/a/b/' .env", True),
        ("sed -in 's/a/b/' .env", True),
        ("sed -i -e .env README.md", False),
        ("sed -i -f .env README.md", False),
        ("sed -i --expression=.env README.md", False),
        ("sed -i --file=.env README.md", False),
        ("sed -e's/.env/i/' README.md", False),
        ("sed -finput.sed .env", False),
        ("sed -i -l .env 's/a/b/' README.md", False),
        ("sed -i --line-length .env 's/a/b/' README.md", False),
    ])
    def test_sed_option_arguments_are_not_targets(self, detector, home, command, expected):
        assert detector(command)[0] is expected

    @pytest.mark.parametrize("command,expected", [
        ("sed -i 's/a/b/' .env2>/tmp/out", False),
        ("sed -i 's/a/b/' .env2 2>/tmp/out", False),
        ("sed -i 's/a/b/' README.md < .env", False),
        ("sed -i 's/a/b/' README.md 2>/tmp/out < app/.envrc", False),
        ("sed -i 's/a/b/' README.md <.env", False),
        ("sed -i 's/a/b/' .env < README.md", True),
        ("sed${IFS}-i 's/a/b/' ~/.hermes/config.yaml", True),
        ("echo 'sed${IFS}-i s/a/b/ .env'", False),
        ("env -S \"sed -n -i -e s/a/b/ .env\"", True),
        ("env --split-string='sed --in-place s/a/b/ app/.envrc'", True),
        ("env -S \"sed --posix s/a/b/ .env\"", False),
        ("sudo -- env MODE=placeholder command /usr/bin/sed -n -i 's/a/b/' './.env'", True),
        ("sed -i -e 's/a/b/' -- .env", True),
        ("sed -- -i .env", False),
        ("sed -i 's/a/b/' -- .env", True),
        ("sed -i 's/a/b/' README.md; cat .env", False),
        ("echo \"sed -i s/a/b/ .env\"", False),
        ("sed -i 's/.env/x/' README.md", False),
        ("sed --in-place 's/a/b/' .environment", False),
        ("sed -i 's/a/b/' .envrc.example", False),
        ("sed -i 's/a/b/' notes.envrc.md", False),
        ("sed -i 's/a/b/' .env-sample", False),
        ("sed -i 's/a/b/' README.md # .env", False),
        ("printf '%s' \"$(sed -ni 's/a/b/' .env)\"", True),
        ("sed -i 's/a;|b/' 'app/.envrc'", True),
        ("sed -i 's/a/b/' /srv/app/config.yaml", False),
        ("sed -n '1,5p' .env", False),
        ("sed 's/a/b/' .env > /tmp/out", False),
    ])
    def test_ordinary_sed_and_env_reads_stay_safe(self, detector, home, command, expected):
        assert detector(command)[0] is expected

    @pytest.mark.parametrize("path", ["~/.envrc", "./.envrc", "app/.envrc", ".envrc"])
    @pytest.mark.parametrize("vector", [
        "echo x >> {p}", "echo x > {p}", "echo x | tee -a {p}",
        "cp placeholder {p}", "mv placeholder {p}", "install -m600 placeholder {p}",
        "sed -i 's/a/b/' {p}", "sed --in-place 's/a/b/' {p}",
    ])
    def test_envrc_write_vectors_are_gated(self, detector, home, path, vector):
        dangerous, key, reason = detector(vector.format(p=path))
        assert dangerous and key and reason

    @pytest.mark.parametrize("command", [
        "echo x >> .env", "cp placeholder app/.env", "echo x > ~/.env.local",
        "install -m600 placeholder ./.env.production",
    ])
    def test_env_variants_still_gated(self, detector, home, command):
        assert detector(command)[0]

    @pytest.mark.parametrize("command", [
        "echo x >> ~/.environment", "echo x >> .env-sample",
        "echo x >> .envrc.example", "cp a.txt ~/.envoy/config", "echo x >> notes.envrc.md",
    ])
    def test_near_miss_env_names_stay_safe(self, detector, home, command):
        assert detector(command) == (False, None, None)
