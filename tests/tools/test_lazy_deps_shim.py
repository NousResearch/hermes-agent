"""The retired install_specs shim answers from import truth, not blanket failure (#135131).

Plugins still call `tools.lazy_deps.install_specs` during agent construction (memory providers
wiring an OSS backend). The old shim raised ImportError unconditionally, so a provider whose
driver was already importable through the paths PM installs into silently reported the
capability as unavailable. Satisfied specs must return; only genuinely missing ones raise."""

import pytest

import tools.lazy_deps as lazy


def test_satisfied_spec_returns_instead_of_raising():
    # pytest is importable inside the test environment by construction.
    lazy.install_specs(["pytest"])


def test_missing_spec_raises_naming_it():
    with pytest.raises(ImportError, match="definitely-not-a-real-package-xyz"):
        lazy.install_specs(["definitely-not-a-real-package-xyz"])


def test_mixed_specs_raise_only_for_the_missing_half():
    with pytest.raises(ImportError, match="missing-pkg-135131") as excinfo:
        lazy.install_specs(["pytest>=7", "missing-pkg-135131>=1,<2"])
    assert "pytest" not in str(excinfo.value)


def test_spec_name_parsing_strips_version_and_extras():
    assert lazy._pkg_name_from_spec("chromadb>=0.4,<1") == "chromadb"
    assert lazy._pkg_name_from_spec("package[extra] >= 1.0") == "package"
