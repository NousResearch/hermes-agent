"""A child spawned FOR another profile never inherits the launch profile's settings, and never loses the host cap.

``strip_launch_profile_env`` dropped a ``TERMINAL_*`` name only when it appeared in the launch ``.env`` or
in ``TERMINAL_CONFIG_ENV_MAP``. A key the launch process got from systemd ``Environment=`` / ``op run``
(``TERMINAL_SCRATCH_DIR``) or from a bridge output outside that map (``TERMINAL_DOCKER_IMAGE_PINNED``)
crossed into ``hermes -p B``: B's worker read A's default image as pinned, so a default-image flip deleted
B's persisted sandbox. The host-wide ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is the exception: it can only be
tightened, so B's child keeps it and B's own ``.env`` may lower it, never raise it.

The same residue came from every non-terminal config.yaml -> env bridge (``HERMES_TIMEZONE``,
``AUXILIARY_<TASK>_*``, ``HERMES_AGENT_TIMEOUT``, media policy): B's child ran with A's aux model and
agent timeout where B's config was silent, and with A's timezone even where it was not, because
``hermes_time`` reads ``HERMES_TIMEZONE`` before config.yaml.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.config import apply_terminal_config_to_env
from tools.environments.local import served_profile_child_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The child's real startup: load its own home's .env (and re-bridge its terminal config), then report.
_PROBE = ("from hermes_cli.env_loader import load_hermes_dotenv; load_hermes_dotenv(); import json,os;"
          "print(json.dumps({k:v for k,v in os.environ.items() if k.upper().startswith('TERMINAL_')}))")


def _terminal_env_seen_by_child(env: dict) -> dict:
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=PROJECT_ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def launch_env(tmp_path, monkeypatch):
    """Launch home A (the process's HERMES_HOME) pins a docker image in config.yaml; its ``.env`` holds
    no terminal setting. The scratch dir and the host memory cap arrive through the process env only
    (systemd ``Environment=``); B's ``.env`` asks for a looser cap than the host allows."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\n", encoding="utf-8")
    (a / "config.yaml").write_text("terminal:\n  docker_image: a-image\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\nTERMINAL_LOCAL_MEMORY_MAX_MB=4096\n", encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("TERMINAL_")}
    env["TERMINAL_SCRATCH_DIR"] = str(tmp_path / "a-scratch")
    env["TERMINAL_LOCAL_MEMORY_MAX_MB"] = "64"
    apply_terminal_config_to_env(env=env)  # A's own startup bridge, as load_hermes_dotenv runs it
    return a, b, env


def test_routed_child_drops_launch_terminal_policy_but_keeps_the_host_memory_cap(launch_env):
    """A -> B -> A through the seam every served-profile spawn uses, observed from inside a real child."""
    a, b, env = launch_env
    launch_policy = {"TERMINAL_SCRATCH_DIR": env["TERMINAL_SCRATCH_DIR"], "TERMINAL_DOCKER_IMAGE": "a-image",
                     "TERMINAL_DOCKER_IMAGE_PINNED": "1"}
    host_cap = {"TERMINAL_LOCAL_MEMORY_MAX_MB": "64"}
    assert (launch_policy | host_cap).items() <= env.items()

    def seen(home):
        return _terminal_env_seen_by_child(served_profile_child_env(base=env, target_home=home,
                                                                    inherit_credentials=True))

    assert (launch_policy | host_cap).items() <= seen(a).items()
    routed = seen(b)
    assert not launch_policy.items() & routed.items(), "B's child inherited the launch profile's terminal policy"
    assert host_cap.items() <= routed.items(), "B's child lost, or loosened, the host memory cap"
    assert (launch_policy | host_cap).items() <= seen(a).items()


_BRIDGED = ("HERMES_TIMEZONE", "AUXILIARY_VISION_MODEL", "HERMES_AGENT_TIMEOUT", "HERMES_MEDIA_DELIVERY_STRICT")

# The launch process: the gateway's config.yaml -> env bridge runs at import, then one child per target
# home is built through served_profile_child_env and reports its env and resolved timezone.
_LAUNCH = """
import json, subprocess, sys
import gateway.run  # noqa: F401
from tools.environments.local import served_profile_child_env
names, homes = json.loads(sys.argv[1]), json.loads(sys.argv[2])
probe = ("import hermes_time,json,os,sys; names=json.loads(sys.argv[1]);"
         "print(json.dumps({'env': {n: os.environ[n] for n in names if n in os.environ},"
         " 'tz': hermes_time.get_timezone_name()}))")
seen = []
for home in homes:
    out = subprocess.run([sys.executable, "-c", probe, json.dumps(names)], capture_output=True, text=True,
                         env=served_profile_child_env(target_home=home, inherit_credentials=True))
    seen.append(json.loads(out.stdout.strip().splitlines()[-1]))
print(json.dumps(seen))
"""


def test_routed_child_drops_the_launch_config_bridge_residue_launch_child_keeps_it(tmp_path):
    """A sets a timezone, a vision model, an agent timeout and strict media delivery in config.yaml;
    B's config.yaml sets only its own timezone."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / "config.yaml").write_text(
        "timezone: Asia/Tokyo\nauxiliary:\n  vision:\n    model: a-vision-model\n"
        "agent:\n  gateway_timeout: 1234\ngateway:\n  strict: true\n", encoding="utf-8")
    (b / "config.yaml").write_text("timezone: Europe/Berlin\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in _BRIDGED}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(a), PYTHONPATH=str(PROJECT_ROOT))
    out = subprocess.run([sys.executable, "-c", _LAUNCH, json.dumps(_BRIDGED), json.dumps([str(a), str(b), str(a)])],
                         env=env, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    launch, routed, launch_again = json.loads(out.stdout.strip().splitlines()[-1])

    bridged = {"HERMES_TIMEZONE": "Asia/Tokyo", "AUXILIARY_VISION_MODEL": "a-vision-model",
               "HERMES_AGENT_TIMEOUT": "1234", "HERMES_MEDIA_DELIVERY_STRICT": "1"}
    for child in (launch, launch_again):
        assert child == {"env": bridged, "tz": "Asia/Tokyo"}
    assert routed == {"env": {}, "tz": "Europe/Berlin"}, "B's child inherited the launch profile's bridged config"
