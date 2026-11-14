"""A local Chromium that dies on a missing shared library must be told to install the libraries."""

from unittest.mock import patch

from tools import browser_use_cli

LOADER_ERROR = (
    "Chromium exited with code 127 during startup.\n"
    "/home/u/.hermes/tools/chromium-1208/chrome-linux64/chrome: error while loading shared libraries: "
    "libnspr4.so: cannot open shared object file: No such file or directory"
)


def _resolve_with(error):
    with (
        patch("tools.browser_lightpanda.using_lightpanda_engine", return_value=False),
        patch.object(browser_use_cli, "_export_session_cdp", lambda env, info, key, fail, msg: fail(error)),
    ):
        return browser_use_cli._resolve_local_cdp({}, None)


def test_missing_system_library_is_reported_as_a_dependency_problem():
    """The reported failure: the loader names a library, the advice is to reinstall Chromium."""
    err = _resolve_with(RuntimeError(LOADER_ERROR))
    assert err is not None

    assert "system libraries" in err
    assert "reinstalling Chromium will not help" in err
    assert "to (re)install Chromium" not in err
    assert "libnspr4.so" in err


def test_other_launch_failures_keep_the_reinstall_advice():
    """The guard must not swallow the ordinary "Chromium will not start" case."""
    err = _resolve_with(RuntimeError("Chromium exited with code 1 during startup"))
    assert err is not None

    assert "to (re)install Chromium" in err and "will not help" not in err
