---
title: Testing
---

# Testing

Hermes uses `pytest` for the Python test suite. The project keeps local test
runs close to CI by combining a canonical shell runner with hermetic pytest
fixtures.

## Recommended command

From a POSIX-like checkout, run:

```bash
scripts/run_tests.sh
```

Pass a path, test node id, or pytest flags to narrow the run:

```bash
scripts/run_tests.sh tests/agent/
scripts/run_tests.sh tests/tools/test_example.py::test_specific_case
scripts/run_tests.sh --tb=long -v
```

The script is intentionally preferred over raw `pytest` because it:

- activates an available repository or installed Hermes virtualenv
- installs `pytest-split` into that venv when missing and possible
- unsets credential-shaped environment variables before pytest starts
- unsets Hermes behavioral/session variables that can leak from a live agent
- pins deterministic runtime values (`TZ=UTC`, `LANG=C.UTF-8`,
  `PYTHONHASHSEED=0`)
- clears `pyproject.toml` pytest `addopts` before applying runner defaults
- pins xdist to 4 workers by default, overrideable with `HERMES_TEST_WORKERS`
- excludes integration and e2e tests from the default run

## Raw pytest

`pyproject.toml` sets these defaults:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks tests requiring external services (API keys, Modal, etc.)",
]
addopts = "-m 'not integration' -n auto"
```

For one-off local debugging, clear those defaults explicitly when needed:

```bash
python -m pytest tests/path/to/test_file.py -o "addopts=" -n 0 -v
```

Use raw pytest when you need a single-process debugger-friendly run, a native
Windows fallback, or a very narrow selector. Use `scripts/run_tests.sh` for
pre-push confidence on POSIX-like systems.

## Test isolation model

`tests/conftest.py` enforces the important invariants:

- credential-shaped environment variables are unset before every test
- `HERMES_HOME` points at a per-test temp directory
- `HOME` is not redirected, so production code should use `get_hermes_home()`
  instead of `Path.home() / ".hermes"`
- session, gateway, terminal, browser, and kanban environment variables are
  cleared before tests run
- deterministic runtime values are set
- shared plugin/singleton state is reset between tests
- a Unix SIGALRM-based timeout catches hung tests on supported platforms

When adding tests, avoid depending on a developer's real `~/.hermes`, real API
keys, running gateway sessions, or local machine timezone.

## Integration and e2e tests

Default local runs skip integration and e2e tests. Mark tests that require live
external services with `@pytest.mark.integration`, and document any required
environment variables or local services near the test.

Run these tests deliberately, not as part of routine quick iteration.

## Cross-platform notes

`scripts/run_tests.sh` expects POSIX virtualenv paths such as `.venv/bin/activate`.
On native Windows, use a Python environment with pytest installed and invoke
pytest directly, often with `-n 0` for debugger-friendly single-process runs.

Add explicit skip guards for platform-specific behavior. Common cases include:

- symlink creation, which may require elevated privileges on Windows
- POSIX file modes such as `0o600`, which NTFS does not enforce the same way
- `signal.SIGALRM`, which is Unix-only
- Windows-specific regression tests, which should skip on non-Windows platforms

When a test monkeypatches `sys.platform`, also patch related `platform` module
functions if the code under test reads both.

## Troubleshooting

### The runner cannot find a virtualenv

Create or install a development environment, then retry:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
scripts/run_tests.sh
```

### Local keys appear to be missing inside tests

That is expected. The test harness deliberately removes credential-shaped
environment variables. Tests that need provider behavior should inject fake
values with pytest fixtures such as `monkeypatch`.

### A test passes alone but fails under xdist

Check for shared global state, shared filesystem paths, real `~/.hermes` usage,
fixed ports, or ordering assumptions. Prefer per-test `tmp_path` and the
sandboxed `HERMES_HOME`.
