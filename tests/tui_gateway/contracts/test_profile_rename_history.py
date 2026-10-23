"""Regression for #117696: roster rename history belongs to the wire contract."""

import pytest
from pydantic import ValidationError

from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ProfilesListResult
from tui_gateway.contracts.registry import METHODS, check_result


def test_profile_rename_history_survives_result_validation():
    for names in ([], ["old-bot", "older-bot"]):
        payload = {"profiles": [{"name": "bot", "path": "/profiles/bot", "previous_names": names}]}
        check_result(METHODS["profiles.list"], payload)
        result = ProfilesListResult.model_validate(payload)
        assert result.profiles[0].previous_names == names
    assert ProfilesListResult.model_validate({"profiles": []}).profiles == []
    legacy = ProfilesListResult.model_validate({"profiles": [{"name": "bot", "path": "/profiles/bot"}]})
    assert legacy.profiles[0].previous_names == []


def test_profile_rename_history_keeps_strict_result_validation():
    for extra in ({"previous_names": [None]}, {"previous_names": "old-bot"}, {"unknown_field": []}):
        with pytest.raises(ValidationError):
            ProfilesListResult.model_validate({"profiles": [{"name": "bot", "path": "/profiles/bot", **extra}]})
