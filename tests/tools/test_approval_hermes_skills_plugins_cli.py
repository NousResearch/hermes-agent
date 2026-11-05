"""Extension installation and source trust require terminal approval."""

import pytest

from tools.approval_detection import detect_dangerous_command


DESC = "install, enable or trust a skill or plugin (code and instructions every session loads)"


@pytest.mark.parametrize("command", [
    "hermes skills install foo --yes",
    "hermes -p work skills install foo",
    "hermes --profile=work skills install foo",
    "hermes skills install foo --force",
    'hermes "skills" install foo',
    "hermes skills 'install' foo",
    "hermes skills install",
    "hermes skills update",
    "hermes skills trust .",
    "hermes skills tap add owner/repo",
    "hermes 'skills' 'tap' 'add' owner/repo",
    "hermes skills snapshot import s.json --force",
    "hermes plugins install x --enable --yes-deps",
    'hermes -p work "plugins" "install" x',
    "hermes plugins enable x",
    "hermes plugins update",
    "hermes plugins pack install p.yaml",
    "hermes 'plugins' 'pack' 'install' p.yaml",
    "hermes plugins trust-update-url x",
    "hermes plugins adopt x",
    "hermes plugins validate . --install-deps",
    "hermes plugins validate --install-deps .",
    'hermes plugins validate . "--install-deps"',
    "hermes plugins validate '--install-deps' .",
    "cd /x && hermes skills install foo",
    # -p/--profile is accepted anywhere before `--`, and a quoted flag word reaches argparse unquoted.
    "hermes skills -p work install foo",
    'hermes "-p" work skills install foo',
    "hermes skills tap --profile work add owner/repo",
    "hermes skills snapshot -p work import s.json",
    "hermes plugins --profile=work enable x",
    "hermes plugins pack -p work install p.yaml",
])
def test_extension_mutation_requires_approval(command):
    assert detect_dangerous_command(command) == (True, DESC, DESC)


@pytest.mark.parametrize("command", [
    "hermes skills search pdf",
    "hermes skills browse",
    "hermes skills inspect foo",
    "hermes skills list",
    "hermes skills check",
    "hermes skills audit foo",
    "hermes skills uninstall foo",
    "hermes skills untrust",
    "hermes skills snapshot export out.json",
    "hermes skills tap list",
    "hermes skills tap remove owner/repo",
    "hermes plugins list",
    "hermes plugins search pdf",
    "hermes plugins browse",
    "hermes plugins check",
    "hermes plugins show x",
    "hermes plugins capabilities x",
    "hermes plugins remove x",
    "hermes plugins uninstall x",
    "hermes plugins disable x",
    "hermes plugins pack show p.yaml",
    "hermes plugins pack export",
    "hermes plugins validate .",
    "hermes skills -p work list",
    'hermes "-p" work plugins disable x',
    "hermes skills install-local foo",
    "hermes plugins install-local x",
    "hermes skills tap add-local owner/repo",
    "hermes skills snapshot import-local s.json",
    "hermes plugins pack install-local p.yaml",
    "hermes plugins trust-update-url-local x",
])
def test_read_only_removal_and_longer_verbs_need_no_extension_approval(command):
    assert detect_dangerous_command(command) == (False, None, None)
