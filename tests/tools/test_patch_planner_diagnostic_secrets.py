"""V4A failure diagnostics must not echo arbitrary hunk hints."""

import pytest

from tools.patch_parser import Hunk, HunkLine, _plan_hunks


@pytest.mark.parametrize("hint", [
    "opaque_api_key_7X4Y5Z6SECRET",
    "Authorization: Bearer syntheticlongtokenABC123",
    "https://user:password@example.invalid/path?api_key=opaqueXYZ123",
])
@pytest.mark.parametrize("addition", [False, True])
def test_hint_not_returned_in_planner_errors(hint, addition):
    lines = [HunkLine('+', 'new')] if addition else [HunkLine('-', 'old'), HunkLine('+', 'new')]
    result = _plan_hunks('old\nold\n', [Hunk(context_hint=hint, lines=lines)])
    assert result.errors
    assert hint not in str(result.errors)
    assert 'hunk' in str(result.errors).lower()


def test_not_found_diagnostic_does_not_echo_nearby_source():
    secret = 'opaqueSecretValue7X4Y5Z6'
    content = f'password = "{secret}"\n'
    hunk = Hunk(context_hint='', lines=[HunkLine('-', 'password = "different"'), HunkLine('+', 'changed')])
    result = _plan_hunks(content, [hunk])
    assert result.errors
    assert secret not in str(result.errors)
    assert 'hunk 1 (no hint) not found — Could not find a match for old_string in the file' in str(result.errors)


def test_ambiguous_addition_does_not_echo_hint():
    secret = 'opaqueHintValue7X4Y5Z6'
    hunk = Hunk(context_hint=secret, lines=[HunkLine('+', 'new')])
    result = _plan_hunks(f'{secret}\n{secret}\n', [hunk])
    assert result.errors
    assert secret not in str(result.errors)
    assert 'ambiguous' in str(result.errors)


def test_fuzzy_ambiguous_matches_do_not_echo_source():
    secret = 'opaqueSecretValue7X4Y5Z6'
    content = f'call(foo,  bar) # {secret}\n' * 2
    hunk = Hunk(lines=[HunkLine('-', 'call(foo, bar)'), HunkLine('+', 'changed')])
    result = _plan_hunks(content, [hunk])
    assert result.errors
    assert secret not in str(result.errors)
    assert 'Could not find a match for old_string in the file' in str(result.errors)
