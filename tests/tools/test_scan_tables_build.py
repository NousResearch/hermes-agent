"""Every pattern of the injection-scan tables builds on RE2 (the build is lazy; this forces it)."""
import cron.lifecycle_guard
import hermes_re
import tools.skills_guard
import tools.threat_patterns


def test_every_scan_pattern_builds():
    tools.threat_patterns.scan_for_threats("x")
    assert hermes_re.build_all(wide=True) == []
