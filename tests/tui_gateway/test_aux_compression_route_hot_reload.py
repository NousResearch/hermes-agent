"""Live TUI sessions must re-evaluate the aux-window clamp on an aux route edit.

The compaction trigger is capped at the auxiliary compression model's window
(``check_compression_model_feasibility``, durable ceiling from #114707), so
switching ``auxiliary.compression`` is a threshold change. The TUI live-config
signature ignored the aux route, so a probed session never re-ran the probe:
small→big never lifted the ceiling, and big→small never clamped.

Only the ROUTE keys belong in the signature. ``timeout`` /
``reasoning_effort`` / ``extra_body`` are read per call by auxiliary_client
and must not churn it.
"""

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tui_gateway import server

MAIN_CTX = 272_000
SMALL_AUX_CTX = 128_000


def _cfg(aux_compression: dict | None = None, **compression):
    cfg = {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "context_length": MAIN_CTX,
        },
        "compression": {"threshold": 0.75, **compression},
    }
    if aux_compression is not None:
        cfg["auxiliary"] = {"compression": aux_compression}
    return cfg


def _session(probed=True):
    compressor = ContextCompressor(
        model="gpt-5.6-sol",
        provider="openai-codex",
        threshold_percent=0.75,
        config_context_length=MAIN_CTX,
        quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
        _compression_feasibility_checked=probed,
        _compression_warning=None,
        _aux_compression_context_length_config=None,
        _custom_providers=[],
        status_callback=None,
        _current_main_runtime=lambda: {},
        _emit_diagnostic_status=lambda _m: None,
    )
    return {"agent": agent, "session_key": "sid-aux-route"}, compressor


class TestSignatureTracksTheAuxRoute:
    def test_changing_the_aux_model_changes_the_signature(self):
        before = server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "small-aux"})
        )
        after = server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "big-aux"})
        )
        assert before != after

    def test_changing_the_aux_provider_or_base_url_changes_the_signature(self):
        base = server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "aux"})
        )
        assert base != server._tui_compression_config_signature(
            _cfg({"provider": "openrouter", "model": "aux"})
        )
        assert base != server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "aux", "base_url": "https://example.invalid/v1"})
        )

    def test_changing_the_aux_window_pin_changes_the_signature(self):
        assert server._tui_compression_config_signature(
            _cfg({"provider": "custom", "model": "aux", "context_length": 128_000})
        ) != server._tui_compression_config_signature(_cfg({"provider": "custom", "model": "aux"}))

    def test_removing_the_aux_section_changes_the_signature(self):
        assert server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "aux"})
        ) != server._tui_compression_config_signature(_cfg())

    def test_per_call_aux_knobs_do_not_churn_the_signature(self):
        before = server._tui_compression_config_signature(
            _cfg({"provider": "openai-codex", "model": "aux", "timeout": 300})
        )
        after = server._tui_compression_config_signature(
            _cfg({
                "provider": "openai-codex",
                "model": "aux",
                "timeout": 900,
                "reasoning_effort": "low",
                "extra_body": {"x": 1},
            })
        )
        assert before == after


class _AuxClient:
    base_url = "https://aux.example/v1"
    api_key = "aux-key"


class TestLiveAdoptionReClampsToTheCurrentAuxWindow:
    def _adopt(self, monkeypatch, session, cfg, aux_ctx):
        aux = (cfg.get("auxiliary") or {}).get("compression") or {}
        aux_model = aux.get("model") or "auto-aux"
        calls = []

        def _client(*_a, **_k):
            calls.append(aux_model)
            return _AuxClient(), aux_model

        monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
        monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", _client)
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_task_provider_model",
            lambda *a, **k: (aux.get("provider") or "auto", aux_model, "", "", ""),
        )
        monkeypatch.setattr(
            "agent.model_metadata.get_model_context_length",
            lambda *a, config_context_length=None, **k: config_context_length or aux_ctx,
        )
        server._sync_agent_compression_with_config("sid-aux-route", session)
        return calls

    def test_switching_to_a_small_aux_model_clamps_the_live_trigger(self, monkeypatch):
        session, compressor = _session()
        configured = compressor.threshold_tokens
        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "big-aux"}), MAIN_CTX)
        assert compressor.threshold_tokens == configured

        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "small-aux"}), SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX < configured

    def test_switching_back_to_a_large_aux_model_restores_the_trigger(self, monkeypatch):
        session, compressor = _session()
        configured = compressor.threshold_tokens
        live_agent = session["agent"]
        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "small-aux"}), SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX

        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "big-aux"}), MAIN_CTX)
        assert compressor.threshold_tokens == configured
        # Adopted in place: no rebuild, so the prompt-cache prefix survives.
        assert session["agent"] is live_agent

    def test_never_probed_session_stays_lazy(self, monkeypatch):
        """Same rule as model switch / fallback: only an existing verdict is refreshed eagerly."""
        session, compressor = _session(probed=False)
        configured = compressor.threshold_tokens
        calls = self._adopt(
            monkeypatch, session, _cfg({"provider": "openai-codex", "model": "small-aux"}), SMALL_AUX_CTX
        )
        assert calls == []
        assert compressor.threshold_tokens == configured
        assert session["agent"]._compression_feasibility_checked is False

    def test_removing_the_aux_section_lifts_a_previous_clamp(self, monkeypatch):
        """No auxiliary.compression at all = auto route; a fitting window must still lift the ceiling."""
        session, compressor = _session()
        configured = compressor.threshold_tokens
        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "small-aux"}), SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX

        calls = self._adopt(monkeypatch, session, _cfg(), MAIN_CTX)
        assert calls, "removing the section must re-run the probe"
        assert compressor.threshold_tokens == configured

    def test_aux_window_pin_follows_the_route(self, monkeypatch):
        """A pin for the old aux model must not survive a route switch that drops it."""
        session, compressor = _session()
        configured = compressor.threshold_tokens
        pinned = {"provider": "custom", "model": "small-aux", "context_length": SMALL_AUX_CTX}
        self._adopt(monkeypatch, session, _cfg(pinned), MAIN_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX

        self._adopt(monkeypatch, session, _cfg({"provider": "custom", "model": "big-aux"}), MAIN_CTX)
        assert session["agent"]._aux_compression_context_length_config is None
        assert compressor.threshold_tokens == configured

    def test_aux_below_minimum_keeps_the_live_session(self, monkeypatch, caplog):
        """A hard rejection is fatal at session start, but a live edit only warns."""
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

        session, compressor = _session()
        self._adopt(monkeypatch, session, _cfg({"provider": "openai-codex", "model": "small-aux"}), SMALL_AUX_CTX)
        with caplog.at_level("WARNING"):
            self._adopt(
                monkeypatch, session, _cfg({"provider": "openai-codex", "model": "tiny-aux"}),
                MINIMUM_CONTEXT_LENGTH - 1,
            )
        assert "tiny-aux" in caplog.text
        assert compressor.threshold_tokens == SMALL_AUX_CTX
        # Latch left unset so the compaction-time probe re-raises the rejection.
        assert session["agent"]._compression_feasibility_checked is False

    def test_adoption_never_raises_the_trigger_past_the_aux_window(self, monkeypatch):
        """Raising compression.threshold must not outrun a small aux model."""
        session, compressor = _session()
        small = {"provider": "openai-codex", "model": "small-aux"}
        self._adopt(monkeypatch, session, _cfg(small), SMALL_AUX_CTX)
        self._adopt(monkeypatch, session, _cfg(small, threshold=0.9), SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX
