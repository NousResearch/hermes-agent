"""Native Windows system paths remain absolute inside the hermetic runner."""

import os

import pytest


@pytest.mark.windows_only
def test_system_drive_expansion_cannot_create_a_relative_programdata_tree():
    expanded = os.path.expandvars(r"%SystemDrive%\ProgramData")
    assert os.path.isabs(expanded), expanded
    assert "%" not in expanded
