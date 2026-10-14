"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import hermes_bootstrap

import pytest


@pytest.fixture(autouse=True)
def _stub_managed_chromium(monkeypatch, tmp_path):
    """Keep the PM store's owning-root borrow out of every tui_gateway test.

    Two resolutions reach OUT of the sandboxed HERMES_HOME to this checkout's owning data root,
    which on a source install (``~/​.hermes/hermes-agent``) is the real ``~/.hermes`` — so reading
    the tools store's payload manifest touches the real home and ``home_io_guard`` refuses it as a
    TEST BUG. Both are documented product behaviour (a borrowing launch must not re-download
    tools; the managed Chromium resolves through the same store); CI checks out outside any Hermes
    home and never sees this. Stub both so the suite is green on source installs too:

    * ``pm.environments.owning_home_root`` → None: ``_unstamped_store`` then falls back to
      ``get_default_hermes_root() / "tools"`` — the sandbox, i.e. a fresh install's store. Any
      test that boots a real session/agent (serve exit-flush, profiles listing, subprocess
      encoding) otherwise trips the guard during PM dependency activation.
    * ``hermes_cli.browser_runtime.chromium_executable`` → None: ``display.*`` status snapshots
      probe the managed Chromium through the same store.
    """
    import pm.environments

    monkeypatch.setattr(pm.environments, "owning_home_root", lambda _project_root: None)
    # ``store_root``/``payload_venv`` read the sealed-payload manifest beside the tree the code runs
    # from: on this source install that is the real ``~/.hermes/manifest.json`` (the guard refuses
    # the read attempt even though the file does not exist). A checkout with no payload manifest —
    # CI's case — reads None; stub the same so agent bootstrap in tests stays in the sandbox.
    monkeypatch.setattr(pm.environments, "_payload_manifest", lambda _root: None)
    # ``profiles.list`` scans the wrapper dir (``~/.local/bin``, where the ``hermes -p <profile>``
    # wrappers live) and READS every file in it — on this host that includes the real
    # ``~/.local/bin/node``, which the guard refuses. An empty sandbox bin dir is what tests
    # assume: no wrapper aliases exist there.
    import hermes_cli.profiles as _profiles

    _empty_bin = tmp_path / "wrapper-bin"
    _empty_bin.mkdir(exist_ok=True)
    monkeypatch.setattr(_profiles, "_get_wrapper_dir", lambda: _empty_bin)
    # ``_user_local_bin_entries`` stats the real ``~/.local/bin`` (HOME is deliberately stable in
    # tests) when a terminal env is built for a listed session — another out-of-sandbox probe the
    # guard refuses on a source install. A sandboxed PATH without the developer's user bin dir is
    # what every other isolation layer already assumes.
    from tools.environments import local as _local_env

    monkeypatch.setattr(_local_env, "_user_local_bin_entries", lambda: [])
    from hermes_cli import browser_runtime

    monkeypatch.setattr(browser_runtime, "chromium_executable", lambda **_kwargs: None)
