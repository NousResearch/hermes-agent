"""Fresh-process regression through startup capture and the real child consumer."""
import os
import subprocess
import sys
from pathlib import Path


def test_startup_capture_preserves_user_sites_after_generation_switch(tmp_path):
    home = tmp_path / "home"
    user = tmp_path / "custom" / "lib" / "python3.13" / "site-packages"
    user_dist = tmp_path / "custom" / "dist-packages"
    import hashlib
    install_id = hashlib.sha256(str(Path(__file__).resolve().parents[2]).encode()).hexdigest()[:16]
    old = home / "installs" / install_id / "environments" / "old" / "venv" / "lib" / "python3.14" / "site-packages"
    new = home / "installs" / install_id / "environments" / "new" / "venv" / "lib" / "python3.14" / "site-packages"
    for path in (old, new, user, user_dist):
        path.mkdir(parents=True)
    env = dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=os.pathsep.join(map(str, (old, user, user_dist))))
    code = '''
import os
from pathlib import Path
from tools.environments import local, local_pythonpath
old, user, user_dist, new = map(Path, __import__('sys').argv[1:])
assert old in local._startup_pythonpath_site_packages
# Runtime selection advances after this process captured its startup PYTHONPATH.
local._hermes_site_packages = []
local._hermes_repo_root_aliases = ()
from pm import environments as pe
(new.parents[2] / "pyvenv.cfg").write_text("home = /fake/python\\n")
facts = pe.runtime_facts_path(Path.cwd())
facts.parent.mkdir(parents=True, exist_ok=True)
facts.write_text(__import__('json').dumps({"packages": {"venv": {"environment": str(new.parents[2])}}}))
assert pe.selected_venv(Path.cwd()) == new.parents[2]
child = {"PYTHONPATH": os.pathsep.join(map(str, (old, new, user, user_dist)))}
local_pythonpath._strip_hermes_owned_pythonpath(child)
assert child["PYTHONPATH"] == os.pathsep.join(map(str, (user, user_dist))), child
assert user not in local._startup_pythonpath_site_packages
assert user_dist not in local._startup_pythonpath_site_packages
# Defensive consumer must not trust an unproven startup entry either.
local._startup_pythonpath_site_packages = (old, user, user_dist)
child = {"PYTHONPATH": os.pathsep.join(map(str, (old, user, user_dist)))}
local_pythonpath._strip_hermes_owned_pythonpath(child)
assert child["PYTHONPATH"] == os.pathsep.join(map(str, (user, user_dist))), child
'''
    proc = subprocess.run([sys.executable, "-c", code, *map(str, (old, user, user_dist, new))],
                          cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
