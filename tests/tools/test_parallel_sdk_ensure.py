"""``_ensure_parallel_sdk_installed`` must not fail when the SDK is importable.

The helper's docstring says it "swallows benign ImportError from the lazy_deps
helper itself" and lets the subsequent ``from parallel import ...`` be the real
gate. The exception handlers implemented the opposite: ``lazy_deps.ensure``
signals an unavailable feature with ``FeatureUnavailable``, which subclasses
``RuntimeError`` rather than ``ImportError``, so it fell through to the
``except Exception`` arm and was re-raised as ``ImportError`` — even when the
package was already importable and nothing needed installing.

On a host with ``security.allow_lazy_installs=false`` this made the Parallel
provider unusable regardless of whether ``parallel-web`` was present.
"""

from __future__ import annotations

import sys
import types

import pytest

from plugins.web.parallel.provider import _ensure_parallel_sdk_installed
from tools.lazy_deps import FeatureUnavailable


@pytest.fixture
def fake_parallel_sdk(monkeypatch):
    """Make ``import parallel`` succeed without installing anything."""
    module = types.ModuleType("parallel")

    class Parallel:
        def __init__(self, api_key):
            self.api_key = api_key

    class AsyncParallel:
        def __init__(self, api_key):
            self.api_key = api_key

    module.Parallel = Parallel
    module.AsyncParallel = AsyncParallel
    monkeypatch.setitem(sys.modules, "parallel", module)
    return module


def _deny_lazy_install(*_args, **_kwargs):
    raise FeatureUnavailable(
        "search.parallel",
        ("parallel-web==0.4.2",),
        "lazy installs disabled (security.allow_lazy_installs=false)",
    )


def test_importable_sdk_survives_disabled_lazy_installs(monkeypatch, fake_parallel_sdk):
    """Feature reported unavailable, but the package imports: not an error."""
    monkeypatch.setattr("tools.lazy_deps.ensure", _deny_lazy_install)

    _ensure_parallel_sdk_installed()  # must not raise

    from parallel import Parallel

    assert Parallel(api_key="k").api_key == "k"


def test_missing_sdk_still_reports_the_install_hint(monkeypatch):
    """Genuinely absent package keeps the actionable ImportError."""
    monkeypatch.delitem(sys.modules, "parallel", raising=False)
    monkeypatch.setattr("tools.lazy_deps.ensure", _deny_lazy_install)

    with pytest.raises(ImportError) as excinfo:
        _ensure_parallel_sdk_installed()

    assert "parallel-web" in str(excinfo.value)


def test_unrelated_failure_is_still_surfaced(monkeypatch, fake_parallel_sdk):
    """A non-availability error is a real fault and must not be swallowed."""

    def boom(*_args, **_kwargs):
        raise OSError("disk exploded")

    monkeypatch.setattr("tools.lazy_deps.ensure", boom)

    with pytest.raises(ImportError, match="disk exploded"):
        _ensure_parallel_sdk_installed()
