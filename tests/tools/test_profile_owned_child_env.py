"""Reconciliation invariants for profile-owned child credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

@pytest.fixture
def homes(tmp_path):
    source, target, managed = [tmp_path / n for n in ('source', 'target', 'managed')]
    for home in (source, target, managed):
        home.mkdir()
        (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    seed = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path), 'USER': 'synthetic-audit',
            'HERMES_HOME': str(source), 'HERMES_RUNTIME_DIR': str(Path.cwd()),
            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'TZ': 'UTC',
            'PYTEST_CURRENT_TEST': 'reconciliation'}
    with patch.dict(os.environ, seed, clear=True), patch.object(Path, 'home', return_value=tmp_path):
        from agent import secret_scope
        from hermes_constants import pin_process_hermes_home
        from hermes_cli import env_loader, managed_scope
        assert Path(secret_scope.__file__).resolve().is_relative_to(Path.cwd())
        env_loader.reset_secret_source_cache()
        managed_scope.invalidate_managed_cache()
        try:
            yield source, target, managed
        finally:
            secret_scope.set_multiplex_active(False)
            pin_process_hermes_home(None)
            env_loader.reset_secret_source_cache()
            managed_scope.invalidate_managed_cache()


def test_real_child_source_aliases_cannot_gain_target_authority(homes):
    source, target, _ = homes
    (source / '.env').write_text('ACME_LOGIN=fake-alpha\nSHARED_KEY=fake-alpha-shared\n', encoding='utf-8')
    (target / '.env').write_text('SHARED_KEY=fake-beta-shared\n', encoding='utf-8')
    (target / 'config.yaml').write_text('terminal:\n  env_passthrough:\n    - SHARED_KEY\n', encoding='utf-8')
    from agent.secret_scope import (build_profile_secret_scope, set_secret_scope,
                                   reset_secret_scope, set_multiplex_active)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=str(source))
    os.environ.update({'APPTAINERENV_ACME_LOGIN': 'fake-alpha',
                       'SINGULARITYENV_APPTAINERENV_ACME_LOGIN': 'fake-alpha',
                       '_HERMES_FORCE_ACME_LOGIN': 'fake-forced-alpha',
                       'BENIGN_CONTROL': 'keep'})
    set_multiplex_active(True)
    home_token = set_hermes_home_override(target)
    scope_token = set_secret_scope(build_profile_secret_scope(target), profile_home=str(target))
    try:
        from tools.environments.local import _make_run_env
        env = _make_run_env({})
        names = ['ACME_LOGIN', 'APPTAINERENV_ACME_LOGIN',
                 'SINGULARITYENV_APPTAINERENV_ACME_LOGIN', '_HERMES_FORCE_ACME_LOGIN',
                 'SHARED_KEY', 'BENIGN_CONTROL']
        code = 'import os,json; print(json.dumps({n:os.environ.get(n) for n in ' + repr(names) + '}))'
        child = subprocess.run([sys.executable, '-I', '-c', code], env=env,
                               capture_output=True, text=True, timeout=20, check=True)
        observed = json.loads(child.stdout)
        expected = dict.fromkeys(names)
        expected.update(SHARED_KEY='fake-beta-shared', BENIGN_CONTROL='keep')
        assert observed == expected
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
