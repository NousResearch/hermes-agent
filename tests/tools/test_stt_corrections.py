"""Personal STT correction/glossary tests."""

from pathlib import Path

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_add_and_apply_stt_correction_replaces_phrase_case_insensitively(hermes_home):
    from tools.stt_corrections import add_correction, apply_stt_corrections, list_corrections

    result = add_correction("gétoueur du christart", "GetHooked starter kit")

    assert result["action"] == "added"
    assert list_corrections()[0]["wrong"] == "gétoueur du christart"
    assert (
        apply_stt_corrections("Le GÉTOUEUR du Christart est lent.")
        == "Le GetHooked starter kit est lent."
    )


def test_parse_stt_correction_args_accepts_arrow_separator(hermes_home):
    from tools.stt_corrections import parse_correction_args

    wrong, right = parse_correction_args("gétoueur du Christart => GetHooked starter kit")

    assert wrong == "gétoueur du Christart"
    assert right == "GetHooked starter kit"


def test_remove_stt_correction_by_one_based_index(hermes_home):
    from tools.stt_corrections import add_correction, list_corrections, remove_correction

    add_correction("mutuelle", "Mutual")
    removed = remove_correction(1)

    assert removed["wrong"] == "mutuelle"
    assert list_corrections() == []
