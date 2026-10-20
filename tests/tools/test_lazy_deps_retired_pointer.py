"""The retired lazy_deps shims must point at the durable install path (#135131).

``install_specs``/``ensure`` keep raising ``ImportError`` — the retirement is
behaviour — but plugin setups (e.g. the mem0 OSS backend's provider-deps loop)
catch that exception and surface its text as the user's only hint. The message
therefore has to name the command that actually works (``hermes pm install
--extra NAME``) instead of asserting that runtime installation is unavailable,
which left callers printing a bare ``uv pip install <pkg>`` that the next
environment sync prunes.
"""

import pytest

import tools.lazy_deps as lazy_deps


def test_install_specs_names_the_durable_command():
    with pytest.raises(ImportError) as excinfo:
        lazy_deps.install_specs(["psycopg2-binary"])
    message = str(excinfo.value)
    assert "hermes pm install --extra NAME" in message
    assert "is retired" in message
    assert "pruned on the next environment sync" in message


def test_ensure_names_the_durable_command():
    with pytest.raises(ImportError) as excinfo:
        lazy_deps.ensure("browser")
    message = str(excinfo.value)
    assert "hermes pm install" in message
    assert "pruned on the next environment sync" in message
    # Relaunching installs nothing; the old pointer must not survive.
    assert "relaunch" not in message.lower()
