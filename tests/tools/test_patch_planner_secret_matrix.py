"""Synthetic secrets in source, hunks, hints and backend errors never reach the result.

Every refusal and failure path of the V4A pipeline carries reason codes and counts
only. Each secret is checked verbatim, by prefix and suffix, URL-encoded and base64.
"""
import base64
import urllib.parse
from types import SimpleNamespace

import pytest

from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

_SECRETS = [
    "Authorization: Bearer synthBearer7X4Y5Z6QMM8",
    "https://user:SynthPw0rdZ9@example.invalid/p?api_key=SynthKeyAB12CD",
    "ghp_SynthTokenA1B2C3D4E5F6",
    "X-Api-Key: SynthHeaderKey9988",
]
_STAR = "*" * 3


def _forms(secret):
    return {
        secret,
        secret[:8],
        secret[-6:],
        urllib.parse.quote(secret, safe=""),
        base64.b64encode(secret.encode()).decode(),
    }


class Store:
    def __init__(self, files, read_error=None, write_error=None):
        self.files = dict(files)
        self.read_error = read_error
        self.write_error = write_error
        self.writes = []

    def read_file_raw(self, path):
        if self.read_error:
            return SimpleNamespace(error=self.read_error, content=None, not_found=False)
        if path not in self.files:
            return SimpleNamespace(error="no such file", content=None, not_found=True)
        return SimpleNamespace(error=None, content=self.files[path], not_found=False)

    def write_file(self, path, content, pre_content=None):
        if self.write_error:
            return SimpleNamespace(error=self.write_error)
        self.writes.append(path)
        self.files[path] = content
        return SimpleNamespace(error=None)


def _patch(body):
    return f"{_STAR} Begin Patch\n{_STAR} Update File: f\n{body}{_STAR} End Patch\n"


def _run(secret, case):
    src = f'k = "{secret}"\nk = "{secret}"\nz = 1\n'
    store = Store({"f": src})
    if case == "not_found_near_source":
        body = f'@@\n-k = "other{secret[:4]}"\n+k = "x"\n'
    elif case == "ambiguous_source":
        body = '@@\n-k = "' + secret + '"\n+k = "x"\n'
    elif case == "hint_missing":
        body = f"@@ {secret}-absent @@\n-z = 9\n+z = 2\n"
    elif case == "hint_ambiguous_addition":
        body = f"@@ k = @@\n+added {secret}\n"
    elif case == "hunk_text_not_found":
        body = f"@@\n-nothing {secret}\n+replacement {secret}\n"
    elif case == "read_error":
        store = Store({"f": src}, read_error=f"boom {secret}")
        body = "@@\n-z = 1\n+z = 2\n"
    else:
        store = Store({"f": src}, write_error=f"disk {secret}")
        body = "@@\n-z = 1\n+z = 2\n"
    operations, error = parse_v4a_patch(_patch(body))
    assert error is None, error
    return apply_v4a_operations(operations, store), store


@pytest.mark.parametrize("secret", _SECRETS)
@pytest.mark.parametrize("case", [
    "not_found_near_source", "ambiguous_source", "hint_missing",
    "hint_ambiguous_addition", "hunk_text_not_found", "read_error", "write_error",
])
def test_no_secret_form_reaches_the_result(secret, case):
    result, store = _run(secret, case)
    text = repr(result)
    for form in _forms(secret):
        assert form not in text
    if case not in {"write_error"}:
        assert not result.success
    if case != "write_error":
        assert store.writes == []
