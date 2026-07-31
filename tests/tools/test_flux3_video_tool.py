"""Native BFL FLUX 3 tools: gating, transport, media delivery, redaction."""

import asyncio
import base64
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import flux3_video_tool as flux3

GATEWAY = "https://tool-gateway.example.com"
BASE_URL = f"{GATEWAY}/api/bfl"
UPLOAD_PATH = "/api/uploads/bfl"

# The shipped pacing, read before the autouse fixture below rewrites it to
# something the tests can spend in an instant.
_DEFAULT_POLL_BUDGET_SECONDS = flux3._POLL_BUDGET_SECONDS
_DEFAULT_CALL_BACKSTOP_SECONDS = flux3._CALL_BACKSTOP_SECONDS

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _endpoints():
    """Every test runs as if the mount is reachable unless it says otherwise."""
    with patch.object(
        flux3,
        "managed_vendor_endpoints",
        return_value={"origin": GATEWAY, "base_url": BASE_URL, "upload_path": UPLOAD_PATH},
    ):
        yield


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeClient:
    """Captures each request a handler makes.

    A list of responses is served in order, with the last one repeating, so a
    poll that looks twice can be given a job that finishes between looks.
    """

    def __init__(self, response, sink):
        self._responses = list(response) if isinstance(response, list) else [response]
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def request(self, method, url, headers=None, json=None):
        self._sink.append({"method": method, "url": url, "headers": headers or {}, "json": json})
        response = self._responses[min(len(self._sink) - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class _FakeStream:
    """A streaming GET that yields `body` in one chunk."""

    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        yield self._body


@contextmanager
def _fake_download(body, status_code=200):
    """Stub the clip download; yields the list of URLs that were fetched.

    Patched at `create_ssrf_safe_async_client` rather than at httpx, which both
    stubs the transport and asserts the download goes through the SSRF-guarded
    client — the URL is vendor-supplied and fetched from the user's machine.
    """
    from tools import url_safety

    fetched = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url):
            fetched.append(url)
            return _FakeStream(body, status_code)

    with patch.object(url_safety, "create_ssrf_safe_async_client", lambda **_kw: _Client()):
        yield fetched


def _run(coro):
    return asyncio.run(coro)


def _record_sleep(sink):
    async def _sleep(seconds):
        sink.append(seconds)

    return _sleep


def _stepped_clock(look_seconds):
    """A monotonic clock on which every look appears to take `look_seconds`.

    The poll loop reads the clock twice per look — once before the request and
    once after — so advancing on every second read charges a look exactly that
    much budget without spending any real time. Substituted for the module's
    whole ``time`` reference rather than patching ``time.monotonic`` globally,
    which would hand the same jumping clock to the event loop underneath.
    """
    reads = {"n": 0}

    def _monotonic():
        value = (reads["n"] // 2) * look_seconds
        reads["n"] += 1
        return value

    return _monotonic


def _call(handler, args, response, headers=None):
    """Invoke a handler with the transport stubbed; returns (parsed, requests)."""
    sink = []
    import httpx

    with patch.object(
        flux3,
        "managed_gateway_auth_headers",
        return_value=headers if headers is not None else {"Authorization": "Bearer nous-token"},
    ), patch.object(httpx, "AsyncClient", lambda **_kw: _FakeClient(response, sink)):
        raw = _run(handler(args))
    return json.loads(raw), sink


class TestGating:
    def test_hidden_without_a_reachable_mount(self):
        with patch.object(flux3, "managed_vendor_endpoints", return_value=None):
            assert flux3.check_bfl_requirements() is False

    def test_visible_to_any_signed_in_account_whatever_its_entitlement(self):
        # Entitlement is the gateway's ruling, and it states its reason in a
        # refusal the model can act on. Deciding it here as well could only
        # hide the tools from someone the server would have served, so the
        # portal's entitlement view must not be consulted at all.
        with patch.object(flux3, "peek_nous_access_token", return_value="nous-token"), \
                patch(
                    "hermes_cli.nous_account.get_nous_portal_account_info",
                    side_effect=AssertionError("entitlement must not gate visibility"),
                ):
            assert flux3.check_bfl_requirements() is True

    def test_hidden_without_a_nous_credential(self):
        # The gateway takes a Nous bearer and nothing else, so with no token
        # every call could only ever answer "sign in" — six schemas on every
        # API call for something that cannot work.
        with patch.object(flux3, "peek_nous_access_token", return_value=None):
            assert flux3.check_bfl_requirements() is False

    def test_a_profile_sees_a_credential_held_at_the_global_root(self, tmp_path, monkeypatch):
        # A profile that was never logged into separately still calls the
        # gateway with the root login, because the transport's refresh path
        # reads that same global fallback. Probing only the profile's own store
        # would hide the tools from someone whose calls would have worked.
        #
        # Exercised through the real auth store rather than a stub: the
        # fallback is the whole point of the test, and it lives in
        # hermes_cli.auth, not here.
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        root = tmp_path / "root"
        (root / "profiles" / "work").mkdir(parents=True)
        (root / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {"nous": {"access_token": "root-token"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))

        # The profile's own store is empty, so this passes only via the
        # global-root fallback — without which the tools would be hidden.
        assert flux3.peek_nous_access_token() is None
        assert flux3.check_bfl_requirements() is True

    def test_the_credential_probe_never_forces_a_token_refresh(self, monkeypatch):
        # check_fn runs on every CLI start, gateway session and cron tick, so
        # it reads a cached credential rather than sitting on a synchronous
        # OAuth refresh.
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")
        with patch.object(flux3, "read_nous_access_token", side_effect=AssertionError("refreshed")):
            assert flux3.check_bfl_requirements() is True

    def test_fails_closed_when_the_credential_probe_raises(self):
        with patch.object(flux3, "peek_nous_access_token", side_effect=RuntimeError("auth store unreadable")):
            assert flux3.check_bfl_requirements() is False


class TestSubmitTransport:
    def test_text_to_video_posts_the_mode_and_arguments(self):
        response = _FakeResponse(200, {"id": "bfl_job_1", "status": "submitted", "guidance": "Poll bfl_flux3_get_result with id=bfl_job_1"})

        parsed, requests = _call(
            flux3._handle_text_to_video,
            {"prompt": "a lake", "aspect_ratio": "16:9", "duration": 5},
            response,
        )

        assert requests[0]["method"] == "POST"
        assert requests[0]["url"] == f"{BASE_URL}/generations"
        assert requests[0]["json"] == {
            "prompt": "a lake",
            "aspect_ratio": "16:9",
            "duration": 5,
            "mode": "text_to_video",
        }
        assert requests[0]["headers"]["Authorization"] == "Bearer nous-token"
        # The gateway's guidance is the model-facing text, verbatim.
        assert parsed["result"] == "Poll bfl_flux3_get_result with id=bfl_job_1"
        assert parsed["details"]["id"] == "bfl_job_1"

    def test_each_generate_tool_sends_its_own_mode(self):
        for handler, args, mode in [
            (flux3._handle_text_to_video, {"prompt": "a"}, "text_to_video"),
            (flux3._handle_image_to_video, {"prompt": "a", "input_image": "https://x/a.png"}, "image_to_video"),
            (
                flux3._handle_keyframes_to_video,
                {"prompt": "a", "input_images": ["https://x/a.png"], "keyframe_indices": [0]},
                "keyframes_to_video",
            ),
            (flux3._handle_video_continuation, {"prompt": "a", "input_video": "https://x/c.mp4"}, "video_continuation"),
        ]:
            _parsed, requests = _call(handler, args, _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}))
            assert requests[0]["json"]["mode"] == mode

    def test_urls_pass_through_without_an_upload(self):
        # Forwarding a URL is cheaper than downloading and re-uploading it.
        _parsed, requests = _call(
            flux3._handle_image_to_video,
            {"prompt": "a", "input_image": "https://example.com/a.png"},
            _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
        )
        assert requests[0]["json"]["input_image"] == "https://example.com/a.png"

    def test_a_refusal_is_surfaced_as_the_tools_result_text(self):
        # Throttles are designed to be hit: the message is written for the
        # model and must reach it intact, with the machine detail alongside.
        response = _FakeResponse(
            429,
            {
                "error": {
                    "code": "BFL_GENERATION_COOLDOWN",
                    "message": "A new BFL video generation may be started once every 5 minutes. Wait 210 seconds.",
                    "details": {"retryAfterSeconds": 210},
                }
            },
        )

        parsed, _requests = _call(flux3._handle_text_to_video, {"prompt": "a"}, response)

        assert parsed["error"] == "A new BFL video generation may be started once every 5 minutes. Wait 210 seconds."
        assert parsed["details"] == {"retryAfterSeconds": 210}

    def test_a_401_asks_for_a_nous_sign_in(self):
        parsed, _requests = _call(flux3._handle_text_to_video, {"prompt": "a"}, _FakeResponse(401, {"error": {"code": "AUTH_ERROR"}}))

        assert parsed["needs_reauth"] is True
        assert "sign in" in parsed["error"].lower()

    def test_missing_credentials_ask_for_a_sign_in_without_calling_out(self):
        parsed, requests = _call(flux3._handle_text_to_video, {"prompt": "a"}, _FakeResponse(200, {}), headers={})

        assert requests == []
        assert "sign in" in parsed["error"].lower()

    def test_a_transport_failure_reports_the_cause(self):
        parsed, _requests = _call(
            flux3._handle_text_to_video,
            {"prompt": "a"},
            RuntimeError("connect failed"),
        )

        assert "Could not reach the video-generation gateway" in parsed["error"]
        assert "connect failed" in parsed["error"]

    def test_an_unreadable_body_does_not_masquerade_as_success(self):
        parsed, _requests = _call(flux3._handle_text_to_video, {"prompt": "a"}, _FakeResponse(502, None, text="upstream exploded"))

        assert "error" in parsed


@pytest.fixture(autouse=True)
def _no_real_poll_wait(monkeypatch):
    """Pace the in-call poll loop off the test clock, at two looks per call.

    The handler counts its budget rather than reading a clock, so a gap and a
    budget in a fixed ratio give a deterministic number of looks with no fake
    clock: a budget of two gaps spends one wait and takes two looks, which is
    the smallest loop that can still show a job finishing between looks.
    """
    monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 1.0)
    monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 2.0)

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(flux3.asyncio, "sleep", _instant)


class TestPollTransport:
    def test_a_terminal_status_returns_at_once_without_waiting(self):
        response = _FakeResponse(200, {"id": "bfl_job_1", "status": "Error", "guidance": "The job is over."})

        parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, response)

        assert requests[0]["method"] == "GET"
        assert requests[0]["url"] == f"{BASE_URL}/generations/bfl_job_1"
        assert requests[0]["json"] is None
        assert len(requests) == 1
        assert parsed["result"] == "The job is over."

    def test_a_running_job_is_waited_out_inside_the_call(self, monkeypatch):
        # A model has no clock, so telling it to pause produced a burst of polls
        # instead of a paced one. The wait lives here where it cannot be skipped.
        monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 45.0)
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 90.0)
        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})

        slept = []
        with patch.object(flux3.asyncio, "sleep", new=_record_sleep(slept)):
            parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, running)

        assert len(requests) == 2, "should look again after waiting"
        assert sum(slept) == 45.0
        assert parsed["details"]["status"] == "Generating"

    def test_the_loop_keeps_looking_until_its_budget_is_spent(self, monkeypatch):
        # The job endpoint answers at once, so a call that looked a fixed twice
        # spent almost none of the time it was allowed and handed control back
        # to the model four or five times per generation. One call now covers
        # the whole budget, and the model decides to keep waiting once.
        monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 10.0)
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 50.0)
        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})

        parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, running)

        assert len(requests) == 5, "five looks spaced by four ten-second gaps"
        assert parsed["details"]["status"] == "Generating"

    def test_the_wait_is_answerable_to_a_stop(self, monkeypatch):
        # Nothing outside the tool can end a call that has already started —
        # the executor only checks for an interrupt between tools — so /stop
        # has to land inside the wait rather than at the end of it.
        monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 45.0)
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 200.0)
        from tools import interrupt as interrupt_module

        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})
        looks = []

        def _stop_after_one_slice():
            looks.append(True)
            return len(looks) > 1

        slept = []
        with patch.object(interrupt_module, "is_interrupted", _stop_after_one_slice), \
                patch.object(flux3.asyncio, "sleep", new=_record_sleep(slept)):
            parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, running)

        assert slept == [flux3._POLL_WAIT_SLICE_SECONDS], "the rest of the wait is abandoned"
        assert len(requests) == 1, "and so is the second look"
        assert parsed["details"]["status"] == "Generating"

    def test_the_call_returns_as_soon_as_the_job_finishes(self):
        # The point of waiting in here is that the caller gets the result on the
        # wait it was already taking, not one round trip later.
        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Error", "guidance": "That job failed."})

        parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, [running, done])

        assert len(requests) == 2
        assert parsed["result"] == "That job failed."

    def test_a_refusal_without_a_stated_wait_is_returned_immediately(self):
        # A dead job or a bad id has nothing to wait for; sleeping on it would
        # only delay showing the model what to do.
        response = _FakeResponse(429, {"error": {"message": "Too many polls. Wait 30 seconds."}})

        parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, response)

        assert len(requests) == 1
        assert "Too many polls" in parsed["error"]

    def test_a_throttle_is_waited_out_inside_the_call(self, monkeypatch):
        # Handing a throttle back ends the call, and the model it lands on has
        # no clock — it asks again at once, tightening the loop that tripped the
        # limit. The stated wait is taken here instead.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 100.0)
        throttled = _FakeResponse(
            429,
            {"error": {"message": "Too many polls.", "details": {"retryAfterSeconds": 30}}},
        )
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {}, "guidance": "Done."})

        slept = []
        with patch.object(flux3.asyncio, "sleep", new=_record_sleep(slept)):
            parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, [throttled, done])

        assert len(requests) == 2, "the loop survives a throttle"
        assert sum(slept) == 30.0, "and waits exactly as long as it was asked to"
        assert parsed["result"] == "Done."

    def test_a_throttle_never_polls_faster_than_the_loop_s_own_cadence(self, monkeypatch):
        # The gateway's number is a floor on politeness, not a licence to
        # hammer: a small or malformed-but-positive wait must not turn the loop
        # into a tight one against an endpoint that just asked us to slow down.
        monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 10.0)
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 100.0)
        throttled = _FakeResponse(
            429,
            {"error": {"message": "Slow down.", "details": {"retryAfterSeconds": 0.001}}},
        )
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {}, "guidance": "Done."})

        slept = []
        with patch.object(flux3.asyncio, "sleep", new=_record_sleep(slept)):
            _parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, [throttled, done])

        assert len(requests) == 2
        assert sum(slept) == 10.0, "the loop's own gap, not the sliver it was offered"

    def test_a_slow_poll_spends_the_budget_it_actually_took(self, monkeypatch):
        # Counting only the waits would let a gateway that answers slowly run
        # the call far past its budget, leaving the backstop to do the work the
        # budget is supposed to do. A look costs what it takes.
        monkeypatch.setattr(flux3, "_POLL_GAP_SECONDS", 1.0)
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 30.0)
        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})

        monkeypatch.setattr(flux3, "time", SimpleNamespace(monotonic=_stepped_clock(20.0)))

        _parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, running)

        assert len(requests) == 2, "two twenty-second looks exhaust a thirty-second budget"

    def test_a_poll_outwaits_the_gateways_own_poll_budget(self):
        # The gateway bounds one status read at 45s across its retries and
        # regional redirect hops. Giving up before it does turns a slow but
        # healthy poll into a transport error, and an error ends the loop.
        assert flux3._POLL_READ_TIMEOUT_SECONDS > 45.0

    def test_a_blip_costs_a_look_rather_than_the_rest_of_the_call(self, monkeypatch):
        # The generation runs upstream and is unaffected by our failing to ask
        # about it, so one unreachable moment must not throw away the minutes
        # of budget left. Returning it would end the call on an error the model
        # can only answer by polling again at once — the burst the loop exists
        # to prevent.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 100.0)
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {}, "guidance": "Done."})

        parsed, requests = _call(
            flux3._handle_get_result,
            {"id": "bfl_job_1"},
            [RuntimeError("connection reset"), done],
        )

        assert len(requests) == 2, "the loop looked again after the blip"
        assert parsed["result"] == "Done."

    def test_a_gateway_answering_in_html_counts_as_unreachable(self, monkeypatch):
        # What a 502 from an edge in front of the gateway looks like from here:
        # a status code and a page, with no error the model could act on. That
        # is an absent answer, not a refusal, so it is retried like one.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 100.0)
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {}, "guidance": "Done."})

        _parsed, requests = _call(
            flux3._handle_get_result,
            {"id": "bfl_job_1"},
            [_FakeResponse(502, None, text="<html>bad gateway</html>"), done],
        )

        assert len(requests) == 2

    def test_a_gateway_that_stays_down_is_reported_rather_than_retried_out(self, monkeypatch):
        # Tolerance is for blips. A gateway that is genuinely down has to reach
        # the model promptly, not after minutes of a budget spent on a host
        # that is not answering.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 1000.0)

        parsed, requests = _call(
            flux3._handle_get_result,
            {"id": "bfl_job_1"},
            RuntimeError("connection reset"),
        )

        assert len(requests) == flux3._MAX_CONSECUTIVE_TRANSPORT_ERRORS
        assert "Could not reach" in parsed["error"]

    def test_the_tolerance_counts_consecutive_failures_only(self, monkeypatch):
        # A flaky gateway that answers every other look is still usable, so the
        # count has to reset on an answer rather than accumulate over the call.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 1000.0)
        blip = RuntimeError("connection reset")
        running = _FakeResponse(200, {"id": "bfl_job_1", "status": "Generating", "guidance": "Still going."})
        done = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {}, "guidance": "Done."})

        parsed, requests = _call(
            flux3._handle_get_result,
            {"id": "bfl_job_1"},
            [blip, blip, running, blip, blip, done],
        )

        assert len(requests) == 6, "four blips, never three in a row, so the call survives"
        assert parsed["result"] == "Done."

    def test_a_throttle_longer_than_the_budget_is_handed_back(self, monkeypatch):
        # A five-minute generation cooldown cannot be absorbed inside one call,
        # so the model gets the message and the number rather than a call that
        # sits out a wait it can never finish.
        monkeypatch.setattr(flux3, "_POLL_BUDGET_SECONDS", 100.0)
        response = _FakeResponse(
            429,
            {"error": {"message": "Wait 210 seconds.", "details": {"retryAfterSeconds": 210}}},
        )

        parsed, requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, response)

        assert len(requests) == 1
        assert parsed["error"] == "Wait 210 seconds."

    def test_the_backstop_answers_rather_than_letting_the_bridge_kill_the_call(self, monkeypatch):
        # model_tools' async bridge abandons a tool at 300s and reports it as a
        # bare "TimeoutError:" — no job id, nothing to say the generation is
        # still alive and one poll away. Whatever stalls inside, the model is
        # answered from here first.
        monkeypatch.setattr(flux3, "_CALL_BACKSTOP_SECONDS", 0.01)

        async def _never_finishes(*_args, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(flux3, "_poll_until_done", _never_finishes)

        parsed, _requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, _FakeResponse(200, {}))

        assert parsed["details"] == {"id": "bfl_job_1", "status": "Generating"}
        assert "bfl_flux3_get_result" in parsed["result"]
        assert "bfl_job_1" in parsed["result"]

    def test_a_poll_does_not_inherit_the_submit_read_timeout(self):
        # A status GET answers at once. Left on the submit path's patience, one
        # hung poll would spend the whole call's budget by itself — while submit,
        # which really does sit behind an upload and an upstream call, keeps it.
        import httpx

        timeouts = []
        sink = []
        settled = _FakeResponse(200, {"id": "j", "status": "Error", "guidance": "over"})

        def _client(**kwargs):
            timeouts.append(kwargs.get("timeout"))
            return _FakeClient(settled, sink)

        with patch.object(flux3, "managed_gateway_auth_headers", return_value={"Authorization": "Bearer t"}), \
                patch.object(httpx, "AsyncClient", _client):
            _run(flux3._handle_get_result({"id": "j"}))
            _run(flux3._handle_text_to_video({"prompt": "a"}))

        poll_timeout, submit_timeout = timeouts
        assert poll_timeout.read == flux3._POLL_READ_TIMEOUT_SECONDS
        assert submit_timeout.read == flux3._TRANSPORT_READ_TIMEOUT_SECONDS
        assert poll_timeout.read < submit_timeout.read

    def test_the_pacing_stays_clear_of_the_agents_per_tool_ceiling(self):
        # The whole point of the two bounds: a clip finishing on the last look
        # still has to be downloaded inside the backstop, and the backstop has
        # to answer before model_tools' async bridge abandons the tool at 300s.
        assert _DEFAULT_POLL_BUDGET_SECONDS < _DEFAULT_CALL_BACKSTOP_SECONDS
        assert _DEFAULT_CALL_BACKSTOP_SECONDS < 300.0

    def test_download_timeout_never_outlives_the_backstop(self):
        # Near the end of the call, remaining budget after grace is a few
        # seconds. Clamping that up used to schedule a download the outer
        # wait_for then cancelled, answering "still generating" for a Ready job.
        started = time.monotonic() - (
            flux3._CALL_BACKSTOP_SECONDS - flux3._DOWNLOAD_GRACE_SECONDS - 2.0
        )
        assert flux3._download_read_timeout(started) <= 2.0 + 0.5  # clock noise only

    def test_ready_saves_the_clip_and_never_returns_the_signed_url(self, tmp_path):
        # The signed URL is a bearer credential for the clip and it used to be
        # re-keyed into a shell command by hand, dropping characters. Neither
        # can happen if the model never sees it.
        signed = "https://cdn.example/container/flux3-clip.mp4?sig=abc%2Bdef%3D&se=2026"
        response = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {"sample": signed}, "guidance": "Deliver the saved file."})

        with _fake_download(b"x" * (128 * 1024)) as fetched:
            parsed, _requests = _call(
                flux3._handle_get_result,
                {"id": "bfl_job_1", "save_to": str(tmp_path)},
                response,
            )

        saved = tmp_path / "flux3-clip.mp4"
        assert saved.read_bytes() == b"x" * (128 * 1024)
        assert parsed["details"]["saved_path"] == str(saved)
        assert parsed["details"]["result"].get("sample") is None
        assert signed not in json.dumps(parsed)
        # The gateway still owns the delivery wording; the client only supplies
        # the path it cannot know.
        assert parsed["result"].startswith(f"Saved to {saved}.")
        assert "Deliver the saved file." in parsed["result"]
        assert fetched == [signed]

    def test_ready_never_overwrites_an_existing_file(self, tmp_path):
        (tmp_path / "flux3-clip.mp4").write_bytes(b"an earlier clip")
        response = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"}, "guidance": "g"})

        with _fake_download(b"y" * (128 * 1024)):
            parsed, _requests = _call(flux3._handle_get_result, {"id": "bfl_job_1", "save_to": str(tmp_path)}, response)

        assert parsed["details"]["saved_path"] == str(tmp_path / "flux3-clip-2.mp4")
        assert (tmp_path / "flux3-clip.mp4").read_bytes() == b"an earlier clip"

    def test_on_messaging_the_clip_lands_where_the_gateway_may_send_it(self, monkeypatch):
        # A chat user has no filesystem: the attachment is the only way they
        # ever see the clip. Downloads is not a delivery root on a strict
        # gateway, so a clip saved there is dropped on the way out and the
        # reply arrives with nothing attached.
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        # Strict mode also trusts anything written in the last 10 minutes, and
        # a clip we just downloaded is always inside that window. Left on, the
        # assertion below passes from any directory on earth and stops being a
        # statement about where the clip was saved.
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
        response = _FakeResponse(200, {
            "id": "bfl_job_1",
            "status": "Ready",
            "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"},
            "guidance": "Deliver the saved file.",
        })

        with _fake_download(b"x" * (128 * 1024)):
            parsed, _requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, response)

        from gateway.platforms.base import validate_media_delivery_path

        saved = parsed["details"]["saved_path"]
        assert validate_media_delivery_path(saved), "the gateway must be allowed to send it"
        # The exact line to copy, so the path is never retyped from memory.
        assert f"\nMEDIA:{saved}\n" in parsed["result"]

    def test_the_offered_tag_is_one_the_gateway_actually_delivers(self, monkeypatch):
        # The whole point of spelling the line out is that the model pastes it
        # verbatim, so the line has to survive the real extractor. A tag that
        # parses but fails validation is the worst outcome: it is stripped from
        # the reply either way, so the user is shown a message that looks like
        # it simply forgot the attachment.
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        response = _FakeResponse(200, {
            "id": "bfl_job_1",
            "status": "Ready",
            "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"},
            "guidance": "Deliver the saved file.",
        })

        with _fake_download(b"x" * (128 * 1024)):
            parsed, _requests = _call(flux3._handle_get_result, {"id": "bfl_job_1"}, response)

        from gateway.platforms.base import BasePlatformAdapter

        offered = [ln for ln in parsed["result"].splitlines() if ln.startswith("MEDIA:")]
        assert len(offered) == 1, "exactly one line to copy"

        reply = f"Here's the clip.\n\n{offered[0]}\n"
        media, cleaned = BasePlatformAdapter.extract_media(reply)
        assert BasePlatformAdapter.filter_media_delivery_paths(media), "must survive validation"
        assert "MEDIA:" not in cleaned, "the tag is consumed, not shown to the user"

    @pytest.mark.parametrize("platform", ["", "cli", "tui", "desktop"])
    def test_off_messaging_the_clip_stays_a_file_and_no_tag_is_offered(self, tmp_path, monkeypatch, platform):
        # The CLI has no attachment channel and its prompt forbids the tag —
        # emitting one there just prints literal text at the user.
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
        response = _FakeResponse(200, {
            "id": "bfl_job_1",
            "status": "Ready",
            "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"},
            "guidance": "Deliver the saved file.",
        })

        with _fake_download(b"x" * (128 * 1024)):
            parsed, _requests = _call(
                flux3._handle_get_result, {"id": "bfl_job_1", "save_to": str(tmp_path)}, response,
            )

        assert parsed["result"].startswith(f"Saved to {tmp_path / 'flux3-clip.mp4'}.")
        assert "MEDIA:" not in parsed["result"]

    @pytest.mark.parametrize("platform", ["api_server", "webhook", "msgraph_webhook", "local"])
    def test_platforms_without_an_attachment_channel_are_offered_no_tag(self, tmp_path, monkeypatch, platform):
        # These carry a real platform value but no way to attach a file. The
        # API server in particular only inlines *images* as data URLs and
        # leaves every other MEDIA: tag untouched, so offering one here puts
        # the literal text in front of an OpenAI-compatible caller.
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
        response = _FakeResponse(200, {
            "id": "bfl_job_1",
            "status": "Ready",
            "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"},
            "guidance": "Deliver the saved file.",
        })

        with _fake_download(b"x" * (128 * 1024)):
            parsed, _requests = _call(
                flux3._handle_get_result, {"id": "bfl_job_1", "save_to": str(tmp_path)}, response,
            )

        assert "MEDIA:" not in parsed["result"]

    def test_a_cli_session_is_recognised_by_its_source(self, tmp_path, monkeypatch):
        # The CLI, TUI, and desktop leave HERMES_SESSION_PLATFORM empty and
        # identify themselves on HERMES_SESSION_SOURCE instead, so keying only
        # on the platform would miss them.
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "tui")
        response = _FakeResponse(200, {
            "id": "bfl_job_1",
            "status": "Ready",
            "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=a"},
            "guidance": "Deliver the saved file.",
        })

        with _fake_download(b"x" * (128 * 1024)):
            parsed, _requests = _call(
                flux3._handle_get_result, {"id": "bfl_job_1", "save_to": str(tmp_path)}, response,
            )

        assert "MEDIA:" not in parsed["result"]

    def test_a_rejected_download_fails_loudly_and_leaves_no_file(self, tmp_path):
        # The original bug: a bad signature returns an XML error body, curl
        # writes it to the .mp4 and exits 0, and it reads as success. A short
        # body is not a video whatever the status code said.
        response = _FakeResponse(200, {"id": "bfl_job_1", "status": "Ready", "result": {"sample": "https://cdn.example/x/flux3-clip.mp4?sig=bad"}, "guidance": "g"})

        with _fake_download(b"<?xml version='1.0'?><Error>AuthenticationFailed</Error>"):
            parsed, _requests = _call(flux3._handle_get_result, {"id": "bfl_job_1", "save_to": str(tmp_path)}, response)

        assert "saving it failed" in parsed["result"]
        assert "Poll this job again" in parsed["result"]
        # Neither a half-written .part nor a plausible-looking .mp4 survives.
        assert [p.name for p in tmp_path.glob("*.mp4*")] == []
        assert "saved_path" not in parsed["details"]

    def test_poll_requires_an_id(self):
        parsed, requests = _call(flux3._handle_get_result, {}, _FakeResponse(200, {}))

        assert "id is required" in parsed["error"]
        assert requests == []

    def test_poll_url_encodes_the_job_id(self):
        _parsed, requests = _call(
            flux3._handle_get_result,
            {"id": "weird/../id"},
            _FakeResponse(200, {"id": "x", "guidance": "ok"}),
        )
        assert requests[0]["url"] == f"{BASE_URL}/generations/weird%2F..%2Fid"


class TestMediaDelivery:
    def _resolved(self, mime="image/png", data=_PNG):
        return SimpleNamespace(data=data, mime=mime)

    def test_a_local_path_is_uploaded_and_replaced_with_a_reference(self):
        async def fake_uploader(data, mime):
            assert data == _PNG
            assert mime == "image/png"
            return "nous-upload:token-1"

        with patch.object(flux3, "build_managed_media_uploader", return_value=fake_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved()
        ) as resolve:
            _parsed, requests = _call(
                flux3._handle_image_to_video,
                {"prompt": "a", "input_image": "/tmp/frame.png"},
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        assert requests[0]["json"]["input_image"] == "nous-upload:token-1"
        # Images and video ride the same safety pipeline; only the permitted
        # type differs, and an image field must not accept a video.
        assert resolve.call_args.kwargs["permitted"] == ("image",)

    def test_video_fields_permit_video_only(self):
        async def fake_uploader(data, mime):
            return "nous-upload:token-v"

        with patch.object(flux3, "build_managed_media_uploader", return_value=fake_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved("video/mp4", b"\x00\x00\x00\x18ftypmp42")
        ) as resolve:
            _parsed, requests = _call(
                flux3._handle_video_continuation,
                {"prompt": "a", "input_video": "/tmp/clip.mp4"},
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        assert requests[0]["json"]["input_video"] == "nous-upload:token-v"
        assert resolve.call_args.kwargs["permitted"] == ("video",)

    def test_every_keyframe_path_is_uploaded(self):
        uploads = []

        async def fake_uploader(data, mime):
            uploads.append(mime)
            return f"nous-upload:token-{len(uploads)}"

        with patch.object(flux3, "build_managed_media_uploader", return_value=fake_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved()
        ):
            _parsed, requests = _call(
                flux3._handle_keyframes_to_video,
                {"prompt": "a", "input_images": ["/tmp/a.png", "https://x/b.png", "/tmp/c.png"], "keyframe_indices": [0, 24, 48]},
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        # The URL in the middle is forwarded untouched.
        assert requests[0]["json"]["input_images"] == [
            "nous-upload:token-1",
            "https://x/b.png",
            "nous-upload:token-2",
        ]

    def test_a_list_valued_input_image_is_still_uploaded(self):
        # The gateway accepts input_image as a string OR a list, so a list of
        # local paths must not slip past unsanitized — that would send raw
        # filesystem paths to the vendor and disclose the user's directories.
        async def fake_uploader(data, mime):
            return "nous-upload:token-1"

        with patch.object(flux3, "build_managed_media_uploader", return_value=fake_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved()
        ):
            _parsed, requests = _call(
                flux3._handle_image_to_video,
                {"prompt": "a", "input_image": ["/tmp/frame.png"]},
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        assert requests[0]["json"]["input_image"] == ["nous-upload:token-1"]
        assert "/tmp/frame.png" not in json.dumps(requests[0]["json"])

    def test_media_fields_are_sanitized_whatever_the_mode_expects(self):
        # The gateway prefers input_image over input_images, so sanitizing only
        # the field this mode documents would let the other one through.
        uploads = []

        async def fake_uploader(data, mime):
            uploads.append(mime)
            return f"nous-upload:token-{len(uploads)}"

        with patch.object(flux3, "build_managed_media_uploader", return_value=fake_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved()
        ):
            _parsed, requests = _call(
                flux3._handle_keyframes_to_video,
                {
                    "prompt": "a",
                    "input_images": ["https://x/b.png"],
                    "input_image": "/tmp/sneaky.png",
                    "keyframe_indices": [0],
                },
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        body = json.dumps(requests[0]["json"])
        assert "/tmp/sneaky.png" not in body
        assert requests[0]["json"]["input_image"] == "nous-upload:token-1"

    def test_text_to_video_strips_media_fields_instead_of_uploading_them(self):
        # The mode takes no media, so an upload would spend the caller's quota
        # on a value the gateway ignores.
        def _must_not_upload(*_args, **_kwargs):
            raise AssertionError("text-to-video must not upload anything")

        with patch.object(flux3, "build_managed_media_uploader", _must_not_upload):
            _parsed, requests = _call(
                flux3._handle_text_to_video,
                {"prompt": "a", "input_image": "/tmp/frame.png"},
                _FakeResponse(200, {"id": "j", "status": "submitted", "guidance": "ok"}),
            )

        assert "input_image" not in requests[0]["json"]
        assert requests[0]["json"]["mode"] == "text_to_video"

    def test_an_over_long_image_list_is_refused_before_any_upload(self):
        def _must_not_upload(*_args, **_kwargs):
            raise AssertionError("an over-long list must be refused before uploading")

        with patch.object(flux3, "build_managed_media_uploader", _must_not_upload):
            parsed, requests = _call(
                flux3._handle_keyframes_to_video,
                {"prompt": "a", "input_images": [f"/tmp/{i}.png" for i in range(11)], "keyframe_indices": [0]},
                _FakeResponse(200, {}),
            )

        assert "at most 10" in parsed["error"]
        assert requests == []

    def test_an_upload_refusal_becomes_the_tools_error(self):
        async def failing_uploader(data, mime):
            raise RuntimeError("the daily upload budget for this account is exhausted")

        with patch.object(flux3, "build_managed_media_uploader", return_value=failing_uploader), patch(
            "tools.image_source.resolve_image_source", return_value=self._resolved()
        ):
            parsed, requests = _call(
                flux3._handle_image_to_video,
                {"prompt": "a", "input_image": "/tmp/frame.png"},
                _FakeResponse(200, {}),
            )

        assert "daily upload budget" in parsed["error"]
        # A failed upload must not reach the gateway as a bare local path.
        assert requests == []

    def test_an_unreadable_file_is_reported_without_dumping_the_value(self):
        from tools.image_source import SourceNotFound

        with patch.object(flux3, "build_managed_media_uploader", return_value=lambda *a: None), patch(
            "tools.image_source.resolve_image_source", side_effect=SourceNotFound("media file not found", src="/tmp/x.png")
        ):
            parsed, requests = _call(
                flux3._handle_image_to_video,
                # Long, but unmistakably a path (dots and dashes are outside
                # the base64 alphabet, so the payload guard leaves it alone).
                {"prompt": "a", "input_image": "/tmp/" + "a-b." * 2000 + "frame.png"},
                _FakeResponse(200, {}),
            )

        assert "error" in parsed
        # The offending value is truncated: echoing it whole would blow up the
        # model's context on the way to reporting a bad path.
        assert len(parsed["error"]) < 500
        assert requests == []


class TestLocalPathDetection:
    @pytest.mark.parametrize(
        "value",
        ["/tmp/frame.png", "~/Pictures/f.png", "./f.png", "../f.png", "file:///tmp/f.png", r"C:\Users\me\f.png", r"\\nas\share\f.png"],
    )
    def test_rooted_paths_are_read_off_disk(self, value):
        assert flux3._looks_like_local_path(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "frame.png",
            "https://example.com/f.png",
            "nous-upload:eyJhbGciOiJIUzI1NiJ9.e30.sig",
            "C:frame.png",
            # Inline base64 of a JPEG always starts "/9j/" (first byte 0xFF),
            # which must not read as an absolute POSIX path.
            "/9j/4AAQSkZJRgABAQAAAQ" + "A" * 300 + "==",
        ],
    )
    def test_ambiguous_and_remote_values_are_forwarded(self, value):
        assert flux3._looks_like_local_path(value) is False

    def test_a_short_base64_lookalike_path_is_still_a_path(self):
        assert flux3._looks_like_local_path("/tmp/frames/a1") is True


class TestSchemas:
    def test_every_tool_is_registered_under_the_bfl_toolset(self):
        from tools.registry import registry

        for name in [
            "bfl_flux3_text_to_video",
            "bfl_flux3_image_to_video",
            "bfl_flux3_keyframes_to_video",
            "bfl_flux3_video_continuation",
            "bfl_flux3_get_result",
            "bfl_flux3_prompting_guide",
        ]:
            entry = registry.get_entry(name)
            assert entry is not None, f"{name} is not registered"
            assert entry.toolset == "bfl"
            assert entry.check_fn is flux3.check_bfl_requirements

    def test_generate_tools_point_at_the_guide_and_the_poll_tool(self):
        # Descriptions are the only text guaranteed to be in context when a
        # model picks a tool, so the pointers live there.
        for schema in [flux3.TEXT_TO_VIDEO_SCHEMA, flux3.IMAGE_TO_VIDEO_SCHEMA, flux3.KEYFRAMES_TO_VIDEO_SCHEMA, flux3.VIDEO_CONTINUATION_SCHEMA]:
            assert "bfl_flux3_prompting_guide" in schema["description"]
            assert "bfl_flux3_get_result" in schema["description"]

    def test_the_guide_covers_the_methodology_without_pinning_server_policy(self):
        guide = flux3.FLUX3_PROMPTING_GUIDE
        assert "grounding" in guide.lower()
        assert "bfl_flux3_get_result" in guide
        # Waits and limits ship live in the gateway's responses; pinning them
        # here would let the client lie about what the server enforces.
        assert "5 minutes" not in guide
        assert "per minute" not in guide

    def test_the_guide_tool_takes_no_arguments_and_calls_nothing(self):
        assert flux3.PROMPTING_GUIDE_SCHEMA["parameters"]["properties"] == {}
        assert _run(flux3._handle_prompting_guide({})) == flux3.FLUX3_PROMPTING_GUIDE
