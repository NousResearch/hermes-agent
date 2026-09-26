"""Block-comment prose must not hard-block a plugin; executable lines still do."""
import pytest

from tools.plugin_guard import scan_plugin, should_allow_plugin_install


def scan(tmp_path, source, suffix='.ts'):
    root = tmp_path / 'plugin'
    root.mkdir()
    (root / ('adapter' + suffix)).write_text(source, encoding='utf-8')
    return scan_plugin(root)


@pytest.mark.parametrize('suffix', ['.js', '.ts'])
@pytest.mark.parametrize('source', [
    '/**\n * Refuses /etc/shadow before reading.\n */\nexport const x = 1;',
    '/* Refuses /etc/shadow before reading. */',
    '\ufeff  /*\r\n Refuses /etc/shadow.\r\n */\r\n',
    'const x = 1; /* header\n * Refuses /etc/shadow.\n */ const y = 2;',
    '/* /etc/shadow */ /* /etc/passwd */',
    '#!node /*\n/** Refuses /etc/shadow. */',
    '/**\n * Examples: `template`, /regex/, \\ slash, <!-- and -->.\n * Refuses /etc/shadow.\n */',
])
def test_whole_line_block_prose_stays_visible_and_installable(tmp_path, source, suffix):
    result = scan(tmp_path, source, suffix)
    findings = [f for f in result.findings if f.pattern_id == 'system_passwd_access']
    assert findings
    assert all(f.severity in {'low', 'medium'} for f in findings)
    assert should_allow_plugin_install(result)[0] is True


@pytest.mark.parametrize('source', [
    '/* harmless */ readFileSync("/etc/shadow");',
    'readFileSync("/etc/shadow"); /* harmless */',
    '/* harmless */ readFileSync("/etc/shadow"); /* benign */',
    '/**\n * harmless\n */\nreadFileSync("/etc/shadow");',
    'const marker = "/*";\nreadFileSync("/etc/shadow");\n/* */',
    "const marker = '/*';\nreadFileSync('/etc/shadow');\n/* */",
    '// /*\nreadFileSync("/etc/shadow");\n/* */',
    'const marker = `/*`;\nreadFileSync("/etc/shadow");\n/* */',
    'const marker = `${"/*"}`;\nreadFileSync("/etc/shadow");\n/* */',
    'const marker = `${`nested ${"/*"}`}`;\nreadFileSync("/etc/shadow");\n/* */',
    'const rx = /[/*]/;\nreadFileSync("/etc/shadow");\n/* */',
    'if (yes) /[/*]/.test(value);\nreadFileSync("/etc/shadow");\n/* */',
    'const ratio = 4 / 2;\nreadFileSync("/etc/shadow");\n/* */',
    '/* /etc/shadow',
    'const marker = "broken\n/* /etc/shadow */',
    '/* benign */' + ' ' * 150 + 'readFileSync("/etc/shadow");',
    '--> /*\nreadFileSync("/etc/shadow");\n/* */',
    '#!node /*\nreadFileSync("/etc/shadow");\n/* */',
    '<!-- /*\nreadFileSync("/etc/shadow");\n/* */',
    '/* benign */\u2028readFileSync("/etc/shadow");',
])
def test_block_markers_never_launder_executable_or_uncertain_lines(tmp_path, source):
    result = scan(tmp_path, source)
    assert any(f.pattern_id == 'system_passwd_access' and f.severity == 'critical'
               for f in result.findings)
    assert should_allow_plugin_install(result, force=True)[0] is False


def test_instruction_documents_keep_full_severity(tmp_path):
    result = scan(tmp_path, '/* echo payload > .claude/settings.json */', '.md')
    assert should_allow_plugin_install(result, force=True)[0] is False


def test_agent_facing_comment_keeps_confirmation(tmp_path):
    result = scan(tmp_path, '/* echo payload > .claude/settings.json */')
    assert any(f.pattern_id == 'other_agent_config_mod_shell' and f.severity == 'high'
               for f in result.findings)
    assert should_allow_plugin_install(result)[0] is None
