"""Friendly display names for delegated subagents (#118081).

The name is presentation metadata only — allocation must be deterministic,
collision-free among live children, and degrade to suffixed reuse once the
pool is exhausted.
"""

from tools.delegate_names import _FRIENDLY_NAMES, assign_display_name


def test_pool_names_are_unique_and_single_token():
    assert len(set(_FRIENDLY_NAMES)) == len(_FRIENDLY_NAMES)
    assert all(not n.startswith(" ") and not n.endswith(" ") for n in _FRIENDLY_NAMES)


def test_first_spawn_gets_first_pool_name():
    assert assign_display_name(None) == _FRIENDLY_NAMES[0]
    assert assign_display_name([]) == _FRIENDLY_NAMES[0]


def test_taken_names_are_skipped_in_order():
    taken = set(_FRIENDLY_NAMES[:3])
    assert assign_display_name(taken) == _FRIENDLY_NAMES[3]


def test_exhausted_pool_reuses_with_suffix():
    taken = set(_FRIENDLY_NAMES)
    assert assign_display_name(taken) == f"{_FRIENDLY_NAMES[0]} 2"
    taken.add(f"{_FRIENDLY_NAMES[0]} 2")
    assert assign_display_name(taken) == f"{_FRIENDLY_NAMES[1]} 2"


def test_falsy_taken_entries_are_ignored():
    # registry records without a display name contribute None/"" — never block allocation
    assert assign_display_name([None, "", _FRIENDLY_NAMES[0]]) == _FRIENDLY_NAMES[1]
