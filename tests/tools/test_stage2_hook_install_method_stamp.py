"""Contract test: the s6-overlay stage2 hook must NOT stamp the install method
into the shared $HERMES_HOME, and must heal a stale 'docker' stamp left there
by older images.

Background (shared-$HERMES_HOME bug)
------------------------------------
$HERMES_HOME (/opt/data) is a DATA volume that users commonly bind-mount from
the host (``~/.hermes:/opt/data``) and sometimes share with a host-side
Desktop/CLI install. Older images wrote ``printf 'docker' > $HERMES_HOME/.install_method``
at boot, which clobbered the host install's own marker — so the host's in-app
updater read 'docker' and refused to run ``hermes update`` ("doesn't apply
inside the Docker container").

The fix scopes the stamp to the install tree (baked at
``/opt/hermes/.install_method`` in the Dockerfile, read first by
``detect_install_method``). stage2 must therefore:

  * NOT write the 'docker' stamp into $HERMES_HOME any more, and
  * proactively remove a stale 'docker' stamp from $HERMES_HOME so homes
    already poisoned by an older image self-heal on the next boot.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def test_stage2_does_not_write_install_method_into_home(stage2_text: str) -> None:
    # No write/tee of the home-scoped install-method stamp anywhere.
    assert not re.search(
        r"(tee|>)\s*\"?\$HERMES_HOME/\.install_method", stage2_text
    ), (
        "stage2 must not stamp $HERMES_HOME/.install_method — that data dir "
        "may be shared with a host install whose marker would be clobbered"
    )


def test_stage2_heals_stale_docker_home_stamp(stage2_text: str) -> None:
    # It must remove a stale 'docker' stamp from $HERMES_HOME so already
    # poisoned shared homes recover.
    assert 'rm -f "$HERMES_HOME/.install_method"' in stage2_text, (
        "stage2 must remove a stale 'docker' stamp from $HERMES_HOME to heal "
        "homes poisoned by older images"
    )
    # The removal must be guarded on the value being 'docker' so we never
    # delete a legitimately-different stamp a user/host install put there.
    assert re.search(r'\[\s*"\$stamped"\s*=\s*"docker"\s*\]', stage2_text), (
        "the stale-stamp removal must be guarded on the value == 'docker'"
    )
