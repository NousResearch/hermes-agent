"""Every pattern of the injection-scan tables builds on RE2 (the build is lazy; this forces it)."""
import cron.lifecycle_guard
import cron.prompt_scan
import hermes_re
import tools.skills_guard
import tools.threat_patterns


def test_every_scan_pattern_builds():
    tools.threat_patterns.scan_for_threats("x")
    # The cron tables are pattern strings the scanner compiles on first use, with the flag it
    # passes: compile them the same way so build_all() sees them (the list keeps them alive). A matcher entry
    # compiled its patterns at import.
    cron_tables = [
        hermes_re.compile(pattern, hermes_re.IGNORECASE)
        for table in (cron.prompt_scan._CRON_THREAT_PATTERNS, cron.prompt_scan._CRON_EXFIL_COMMAND_PATTERNS)
        for pattern, _pid in table if isinstance(pattern, str)
    ]
    assert cron_tables
    assert hermes_re.build_all(wide=True) == []
