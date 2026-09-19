"""Discord REST v10 permission-overwrite wire-contract tests."""

import pytest

from tools.discord_api.permissions import (
    MAX_SNOWFLAKE,
    PermissionOverwriteError,
    TYPE_MEMBER,
    TYPE_ROLE,
    delete_channel_permission_request,
    set_channel_permission_request,
)

CHANNEL = "123456789012345678"
OVERWRITE = "987654321098765432"


def test_type_constants_match_discord_rest_v10():
    assert TYPE_ROLE == 0
    assert TYPE_MEMBER == 1


def test_set_payload_serializes_bitfields_as_decimal_strings():
    req = set_channel_permission_request(
        CHANNEL,
        OVERWRITE,
        allow=1024,
        deny="0008",
        type_=TYPE_MEMBER,
    )
    assert req == {
        "method": "PUT",
        "path": f"/channels/{CHANNEL}/permissions/{OVERWRITE}",
        "payload": {"allow": "1024", "deny": "8", "type": 1},
    }


def test_defaults_are_wire_strings_and_role_type_zero():
    req = set_channel_permission_request(CHANNEL, OVERWRITE, type_=TYPE_ROLE)
    assert req["payload"] == {"allow": "0", "deny": "0", "type": 0}


def test_accepts_full_unsigned_64_bit_snowflake_range():
    req = set_channel_permission_request(2**63, MAX_SNOWFLAKE, type_=TYPE_ROLE)
    assert req["path"] == f"/channels/{2**63}/permissions/{MAX_SNOWFLAKE}"


@pytest.mark.parametrize("bad", [0, "0", -1, 2**64, "18446744073709551616"])
def test_route_snowflakes_must_be_positive_uint64(bad):
    with pytest.raises(PermissionOverwriteError):
        set_channel_permission_request(bad, OVERWRITE, type_=TYPE_ROLE)
    with pytest.raises(PermissionOverwriteError):
        set_channel_permission_request(CHANNEL, bad, type_=TYPE_ROLE)


@pytest.mark.parametrize("bad", [2, -1, "1", 1.0, True, None])
def test_invalid_type_rejected(bad):
    with pytest.raises(PermissionOverwriteError):
        set_channel_permission_request(CHANNEL, OVERWRITE, type_=bad)


@pytest.mark.parametrize("bad", [-1, "-1", 2**64, True, 1.25, None])
def test_invalid_bitfields_rejected(bad):
    with pytest.raises(PermissionOverwriteError):
        set_channel_permission_request(CHANNEL, OVERWRITE, allow=bad, type_=TYPE_ROLE)
    with pytest.raises(PermissionOverwriteError):
        set_channel_permission_request(CHANNEL, OVERWRITE, deny=bad, type_=TYPE_ROLE)


def test_delete_request_shape():
    req = delete_channel_permission_request(CHANNEL, OVERWRITE)
    assert req == {
        "method": "DELETE",
        "path": f"/channels/{CHANNEL}/permissions/{OVERWRITE}",
        "payload": None,
    }
