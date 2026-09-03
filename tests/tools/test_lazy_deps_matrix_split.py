"""``platform.matrix`` must install without pulling ``python-olm`` (#62401).

The Matrix adapter treats E2EE as optional — ``_check_e2ee_deps()`` degrades
to plaintext when olm is absent, and ``mautrix.crypto`` is only imported
behind ``if self._encryption:``. But the lazy-dep group pinned
``mautrix[encryption]``, so *every* Matrix user paid for a ``python-olm``
build regardless of their E2EE answer.

That build cannot succeed on macOS: python-olm ships manylinux wheels only,
libolm was archived upstream and dropped from Homebrew, and its bundled
``list.hh`` fails to compile under Apple Clang 21 (``++other_pos`` on a
``T *const``) — a hard error, not something ``-Wno-*`` can silence.

So the group is split: ``platform.matrix`` is the plaintext runtime and
installs everywhere, ``platform.matrix.e2ee`` carries olm and is gated off
the hosts where it cannot build.

See also #64065 (``hermes update`` spurious refresh failure on macOS) and
#85588 (the 0.21.1 pin bump taking the adapter down on macOS).
"""
import sys

import pytest

import tools.lazy_deps as ld

BASE = "platform.matrix"
E2EE = "platform.matrix.e2ee"


def _names(feature):
    return {ld._pkg_name_from_spec(s) for s in ld.LAZY_DEPS[feature]}


class TestBaseFeatureIsOlmFree:
    """The plaintext group must be installable on a host with no compiler."""

    def test_base_feature_pins_plain_mautrix(self):
        specs = ld.LAZY_DEPS[BASE]
        assert any(s.startswith("mautrix==") for s in specs), (
            f"{BASE} must pin bare mautrix (no extras marker); got {specs}"
        )

    def test_base_feature_has_no_encryption_extra(self):
        offending = [s for s in ld.LAZY_DEPS[BASE] if "[encryption]" in s]
        assert not offending, (
            f"{BASE} must not request the [encryption] extra — it pulls "
            f"python-olm, which cannot build on macOS. Found: {offending}"
        )

    def test_base_feature_does_not_name_python_olm(self):
        assert "python-olm" not in _names(BASE)

    def test_base_feature_keeps_the_plaintext_runtime_deps(self):
        """Splitting olm out must not drop the deps connect() actually needs."""
        required = {"mautrix", "aiosqlite", "asyncpg", "aiohttp-socks", "aiohttp"}
        assert required <= _names(BASE), (
            f"{BASE} lost runtime deps: {sorted(required - _names(BASE))}"
        )


class TestE2EEFeature:
    def test_e2ee_feature_exists(self):
        assert E2EE in ld.LAZY_DEPS

    def test_e2ee_anchor_is_python_olm(self):
        """``active_features()`` keys off ``specs[0]``.

        If the anchor were mautrix, a plaintext Matrix install would mark the
        E2EE group active and ``hermes update`` would retry the impossible olm
        build on every run — exactly the #64065 symptom.
        """
        assert ld._pkg_name_from_spec(ld.LAZY_DEPS[E2EE][0]) == "python-olm"

    def test_e2ee_requests_the_encryption_extra(self):
        """Keep the extra so uv resolves anything upstream adds to it."""
        assert any("[encryption]" in s for s in ld.LAZY_DEPS[E2EE])

    def test_e2ee_lists_the_extra_contents_explicitly(self):
        """``_is_satisfied`` strips ``[extras]``, so the extra alone is not a
        detectable requirement — bare mautrix would satisfy it. The concrete
        packages must be listed so a half-installed E2EE stack is spotted."""
        assert {"python-olm", "pycryptodome", "unpaddedbase64", "base58"} <= _names(E2EE)

    def test_plain_mautrix_does_not_satisfy_the_e2ee_group(self, monkeypatch):
        """The #62401 'under-checking' defect: with only bare mautrix present,
        the E2EE group must still report work to do."""
        monkeypatch.setattr(
            ld, "_is_satisfied",
            lambda spec: ld._pkg_name_from_spec(spec) == "mautrix",
        )
        assert ld.feature_missing(E2EE) != ()

    def test_e2ee_group_matches_mautrix_declared_extra(self):
        """Drift guard: we mirror mautrix's ``[encryption]`` extra by hand, so
        fail loudly if upstream adds a package we would silently skip."""
        pytest.importorskip("mautrix")
        from importlib.metadata import requires

        from packaging.requirements import Requirement

        declared = {
            ld._pkg_name_from_spec(Requirement(raw).name)
            for raw in (requires("mautrix") or ())
            if (r := Requirement(raw)).marker
            and r.marker.evaluate({"extra": "encryption"})
        }
        missing = {d.lower() for d in declared} - {n.lower() for n in _names(E2EE)}
        assert not missing, (
            f"mautrix[encryption] gained {sorted(missing)} — add them to "
            f"LAZY_DEPS[{E2EE!r}] so _is_satisfied can see them"
        )


class TestActiveFeatureAnchoring:
    def test_plaintext_install_does_not_activate_the_e2ee_group(self, monkeypatch):
        installed = {"mautrix", "aiosqlite", "asyncpg", "aiohttp-socks", "aiohttp"}
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) in installed,
        )
        active = ld.active_features()
        assert BASE in active
        assert E2EE not in active, (
            "a plaintext Matrix install must not make hermes update retry the "
            "olm build (#64065)"
        )


class TestPlatformGate:
    @pytest.mark.parametrize("plat", ["win32", "darwin", "linux"])
    def test_base_matrix_is_supported_on_every_host(self, monkeypatch, plat):
        """Plain mautrix is a pure-python wheel — nothing to gate."""
        monkeypatch.setattr(sys, "platform", plat)
        assert ld._unsupported_feature_reason(BASE) is None

    @pytest.mark.parametrize("plat,label", [("win32", "Windows"), ("darwin", "macOS")])
    def test_e2ee_is_gated_where_olm_cannot_build(self, monkeypatch, plat, label):
        monkeypatch.setattr(sys, "platform", plat)
        reason = ld._unsupported_feature_reason(E2EE)
        assert reason, f"E2EE must be gated on {label}"
        # refresh_active_features classifies skips by this prefix.
        assert reason.startswith("unsupported ")
        assert label in reason
        # The message has to leave the user somewhere to go.
        assert "mautrix" in reason

    def test_e2ee_is_allowed_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert ld._unsupported_feature_reason(E2EE) is None

    @pytest.mark.macos_only
    def test_real_macos_host_gates_e2ee_but_not_base(self):
        """Patching ``sys.platform`` only proves the branch string.

        On a real Mac this is the assertion that matters: the gateway can
        install Matrix, and only the impossible olm build is refused.
        """
        assert ld._unsupported_feature_reason(BASE) is None
        assert "unsupported on macOS" in (ld._unsupported_feature_reason(E2EE) or "")
