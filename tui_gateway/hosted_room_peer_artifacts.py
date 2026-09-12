"""Selected #99159 binary read/ACK methods for an existing root peer client.

Source: 5ee4a941a6b64a084221d7d521ce42480bedc15c. No client executor,
route selection, publication or discard authority is introduced here.
"""
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
from tui_gateway.hosted_room_peer_http import (
    MAX_PEER_ERROR_RESPONSE_BYTES, PeerRunsHTTPError, _PeerResponseTooLarge,
    _PeerResponseDeadlineExceeded, _read_bounded_response, _open_roomlink_url,
    _response_error_code,
)


def _request_url(client, path):
    root = client.base_url + client._profile_prefix
    selected = re.search(r"/p/([^/]+)$", urllib.parse.urlsplit(root).path)
    if selected and urllib.parse.unquote(selected.group(1)) != "default":
        raise PeerRunsHTTPError("named peer output is not supported")
    return root + path


def read_artifact(
    self,
    *,
    run_id: str,
    artifact_id: str,
    grant: str,
) -> bytes:
    path = (
        f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/"
        f"artifacts/{urllib.parse.quote(artifact_id, safe='')}"
    )
    request = urllib.request.Request(
        _request_url(self, path),
        method="GET",
        headers={
            "Authorization": f"HermesRoom {self._require_room_grant(grant)}",
            "User-Agent": "Hermes-RoomLink/1.0",
        },
    )
    deadline = time.monotonic() + self.timeout_seconds
    try:
        with _open_roomlink_url(
            request,
            timeout=self.timeout_seconds,
            reject_redirects=True,
        ) as response:
            data = _read_bounded_response(
                response,
                max_bytes=MAX_ATTACHMENT_BYTES,
                deadline=deadline,
            )
    except _PeerResponseTooLarge as exc:
        raise PeerRunsHTTPError("peer artifact bytes exceed the size limit") from exc
    except _PeerResponseDeadlineExceeded as exc:
        raise PeerRunsHTTPError(
            "peer artifact download exceeded the RoomLink time budget",
            retryable=True,
        ) from exc
    except urllib.error.HTTPError as exc:
        try:
            detail = _read_bounded_response(
                exc,
                max_bytes=MAX_PEER_ERROR_RESPONSE_BYTES,
                deadline=deadline,
            ).decode("utf-8", "replace")[:500]
        except _PeerResponseTooLarge as body_exc:
            raise PeerRunsHTTPError(
                "peer artifact error exceeded the RoomLink size limit",
                status_code=exc.code,
            ) from body_exc
        except _PeerResponseDeadlineExceeded as body_exc:
            raise PeerRunsHTTPError(
                "peer artifact error exceeded the RoomLink time budget",
                retryable=True,
                status_code=exc.code,
            ) from body_exc
        except Exception:
            detail = ""
        raise PeerRunsHTTPError(
            (
                "peer artifact download refused an HTTP redirect"
                if exc.code in {301, 302, 303, 307, 308}
                else f"peer rejected artifact download with HTTP {exc.code}: {detail}"
            ),
            retryable=exc.code in {408, 425, 429} or exc.code >= 500,
            status_code=exc.code,
            error_code=_response_error_code(detail),
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PeerRunsHTTPError(
            f"peer is unreachable: {exc}",
            retryable=True,
        ) from exc
    if not data:
        raise PeerRunsHTTPError("peer artifact bytes are invalid")
    return data


def acknowledge_artifacts(
    self,
    *,
    run_id: str,
    artifact_ids: Sequence[str],
    manifest_digest: str,
    message_event_id: str,
    grant: str,
) -> Mapping[str, Any]:
    _request_url(self, "")
    return self._request(
        f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/artifacts/ack",
        method="POST",
        body={
            "artifact_ids": list(artifact_ids),
            "manifest_digest": manifest_digest,
            "message_event_id": message_event_id,
        },
        room_grant=grant,
        reject_redirects=True,
    )
