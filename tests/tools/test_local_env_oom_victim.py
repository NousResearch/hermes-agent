"""Terminal commands are preferred OOM victims over Hermes (``hermes_platform.oom``)."""

import pytest

@pytest.mark.platforms("linux")
def test_terminal_commands_are_preferred_oom_victims_over_hermes(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = env.execute("sleep 0.3; cat /proc/$$/oom_score_adj")
    finally:
        env.cleanup()
    own = int(open("/proc/self/oom_score_adj", encoding="utf-8").read())
    assert int(result["output"].strip().splitlines()[-1]) == min(1000, own + 500)
