"""LAZY_DEPS maps a feature name to an extra in pyproject.toml.

Two things need a test here.

The map itself: each value must name an extra that pyproject.toml declares.
Nothing else checks that. A typo makes the feature raise FeatureUnavailable
at first use, on the one machine that enabled that backend.

The reader: extra_specs expands a `hermes-agent[x]` reference, and that code
is ours. A cycle or a silent empty result would each ship a
wrong package set. uv resolves the extras its own way and cannot catch a
fault in our reader.

Nothing here restates a version. pyproject.toml holds the specs, uv.lock
pins them, and `uv lock --check` and `uv audit` read the lockfile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import lazy_deps as ld  # noqa: E402


class TestFeatureExtraMapping:
    def test_every_feature_maps_to_a_declared_extra(self):
        """A value in LAZY_DEPS must name an extra that pyproject declares.

        This is the whole contract of the map. A typo here raises
        FeatureUnavailable at first use of that backend, and only on a
        machine that enabled it.
        """
        declared = set(ld._optional_dependencies())
        missing = {
            feature: extra
            for feature, extra in ld.LAZY_DEPS.items()
            if extra not in declared
        }
        assert not missing, (
            f"LAZY_DEPS names extras that pyproject.toml does not declare: "
            f"{missing}"
        )

    def test_every_feature_resolves_to_at_least_one_spec(self):
        """An extra can exist and still be empty after composition.

        An empty result installs nothing and reports success, so the backend
        stays broken with no error to read.
        """
        empty = [f for f in ld.LAZY_DEPS if not ld.feature_specs(f)]
        assert not empty, f"features that resolve to no packages: {empty}"


class TestExtraComposition:
    """extra_specs expands `hermes-agent[x]`. That expansion is our code."""

    def test_self_references_resolve(self):
        """[messaging] contains [telegram], so its specs must appear."""
        composed = set(ld.extra_specs("messaging"))
        assert set(ld.extra_specs("telegram")) <= composed
        assert not any(s.startswith("hermes-agent[") for s in composed), (
            "a self-reference must be expanded, not passed to pip"
        )

    def test_cycles_terminate(self, monkeypatch):
        """A cycle in the extras must not hang or recurse without end."""
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {
            "a": ("hermes-agent[b]",),
            "b": ("hermes-agent[a]",),
        })
        assert ld.extra_specs("a") == ()

    def test_unknown_extra_resolves_to_nothing(self, monkeypatch):
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {})
        assert ld.extra_specs("nope") == ()

    def test_a_marker_on_a_pin_survives_composition(self, monkeypatch):
        """A composed extra must keep the marker that its leaf pin carries.

        [wake] contains [wake-tflite], whose pin is macOS-only. Dropping
        the marker on the way out installs that package on Linux.
        """
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {
            "big": ("hermes-agent[small]", "pkg-c==3.0"),
            "small": ("pkg-a==1.0; platform_system == 'Darwin'",),
        })
        assert set(ld.extra_specs("big")) == {
            "pkg-a==1.0; platform_system == 'Darwin'",
            "pkg-c==3.0",
        }


class TestAnchorSpec:
    """The anchor identifies an extra. A shared helper cannot do that.

    active_features() seeds its record from the anchors, so a wrong anchor
    turns `hermes update` into an installer for backends the user never
    enabled. The regression this guards: [voice] listed
    `hermes-agent[audio-io]` first, expansion put sounddevice at index 0,
    and one sounddevice install marked every audio feature active.
    """

    def test_the_anchor_is_the_first_direct_pin(self, monkeypatch):
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {
            "voice": ("hermes-agent[shared]", "the-real-engine==1.0"),
            "shared": ("helper==1.0",),
        })
        assert ld._anchor_spec("voice") == "the-real-engine==1.0"

    def test_a_reference_only_extra_recurses(self, monkeypatch):
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {
            "outer": ("hermes-agent[inner]",),
            "inner": ("real-pkg==1.0",),
        })
        assert ld._anchor_spec("outer") == "real-pkg==1.0"

    def test_a_cycle_returns_none(self, monkeypatch):
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {
            "a": ("hermes-agent[b]",),
            "b": ("hermes-agent[a]",),
        })
        assert ld._anchor_spec("a") is None

    def test_every_feature_has_an_anchor(self):
        missing = [
            f for f in ld.LAZY_DEPS if ld._anchor_spec(ld.LAZY_DEPS[f]) is None
        ]
        assert not missing, f"features with no anchor: {missing}"

    def test_anchors_identify_their_extras(self):
        """Two features with different extras must have different anchors.

        Two features that map to one extra (stt.mistral and tts.mistral)
        share an anchor by design: one install serves both. Across extras,
        a shared anchor means presence of one package activates a feature
        the user never enabled.
        """
        anchor_to_extra: dict[str, str] = {}
        for extra in set(ld.LAZY_DEPS.values()):
            anchor = ld._anchor_spec(extra)
            assert anchor is not None
            name = ld._pkg_name_from_spec(anchor)
            other = anchor_to_extra.get(name)
            assert other is None or other == extra, (
                f"extras [{extra}] and [{other}] share the anchor package "
                f"{name!r} — active_features cannot tell them apart. Put a "
                f"distinctive pin first in each extra."
            )
            anchor_to_extra[name] = extra


class TestWheelInstallFallsBackToDistMetadata:
    """A wheel install (Nix) has no pyproject.toml beside the code.

    The extras table must come from the installed dist metadata there.
    Without the fallback, every lazy_deps entry point raised on Nix —
    including ensure() on a feature whose packages were baked into the
    sealed venv via extraDependencyGroups, which must be a no-op.
    """

    @pytest.fixture
    def no_pyproject(self, monkeypatch):
        """Simulate the wheel layout: no project root on disk."""
        monkeypatch.setattr(ld, "_project_root", lambda: None)
        ld._pyproject.cache_clear()
        ld._metadata_optional_dependencies.cache_clear()
        yield
        ld._pyproject.cache_clear()
        ld._metadata_optional_dependencies.cache_clear()

    def test_the_extras_table_comes_from_metadata(self, no_pyproject):
        """The test venv has hermes-agent installed, so the real metadata
        path resolves the real table.

        Compared against the metadata's own Provides-Extra, not against
        LAZY_DEPS: a dev venv's installed metadata lags the checkout when
        pyproject gained an extra since the last sync, and that lag is not
        a fault in the reader. On a real wheel install the two match.
        """
        table = ld._optional_dependencies()
        assert table, "dist metadata produced no extras table"
        from importlib.metadata import metadata

        provided = set(metadata("hermes-agent").get_all("Provides-Extra") or [])
        known = provided & set(ld.LAZY_DEPS.values())
        assert known, "no LAZY_DEPS extra appears in the installed metadata"
        missing = {e for e in known if e not in table}
        assert not missing, (
            f"extras in Provides-Extra but absent from the parsed table: {missing}"
        )

    def test_a_baked_feature_is_a_noop_for_ensure(self, no_pyproject, monkeypatch):
        """The exact Nix regression: deps baked, ensure() must not raise."""
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("nothing to install"),
        )
        ld.ensure("provider.anthropic", prompt=False)  # must not raise
        assert ld.is_available("provider.anthropic") is True

    def test_a_marker_survives_the_metadata_round_trip(self, no_pyproject):
        """setuptools ANDs `extra == "x"` onto a pin's own marker; the
        reader must strip the extra clause and keep the platform half."""
        specs = ld.extra_specs("wake-tflite")
        assert len(specs) == 1
        req = ld._parse_spec(specs[0])
        assert req is not None and req.marker is not None
        assert "platform_system" in str(req.marker)
        assert "extra" not in str(req.marker)

    def test_composition_still_expands(self, no_pyproject):
        """Self-references survive the metadata round trip too."""
        composed = set(ld.extra_specs("messaging"))
        assert set(ld.extra_specs("telegram")) <= composed
        assert not any(s.startswith("hermes-agent[") for s in composed)


class TestEntryPointsNeverRaise:
    """is_available and feature_install_command are used in status paths
    with no try/except (wake_word.py); their contracts are bool / Optional.
    """

    @pytest.fixture
    def specs_unreadable(self, monkeypatch):
        monkeypatch.setattr(ld, "_project_root", lambda: None)
        monkeypatch.setattr(ld, "_metadata_optional_dependencies", lambda: {})
        ld._pyproject.cache_clear()
        yield
        ld._pyproject.cache_clear()

    def test_is_available_returns_false(self, specs_unreadable):
        assert ld.is_available("provider.anthropic") is False

    def test_feature_install_command_returns_none(self, specs_unreadable):
        assert ld.feature_install_command("provider.anthropic") is None
