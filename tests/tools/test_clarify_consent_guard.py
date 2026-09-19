"""Regression tests for the clarify auto-decide consent guard (#107068).

When no human answers (headless -q/-z turns, interactive timeouts, gateway
delivery failures) the harness tells the agent to pick an option itself. If an
offered option carries authorization semantics ("authorize me to ...", "allow
me ...", "consent ..."), that instruction converts silence into consent: the
agent picks the Recommended self-authorization option, proceeds with exactly
what its contract forbade, and reports it had user authorization.

Contract under test: a no-user sentinel response is rewritten fail-closed —
authorization-semantic options are excluded from what the agent may pick (and
when EVERY option is authorization-semantic, the agent is told to proceed
WITHOUT the authorization, never to self-authorize). Real user answers are
never rewritten. ``clarify.auto_decide_allow_consent: true`` restores the old
pick-from-all-options behavior.
"""

import json

from tools.clarify_tool import TIMEOUT_RESPONSE, clarify_tool

AUTH = "authorize me to compute these 9 rows directly"
SAFE = ["configure A2A peer then hand off", "no aggregation, layout only"]


def _single_query_cb(question, choices, multi_select=False):
    """The real headless ``hermes chat -q`` clarify callback."""
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback
    return _single_query_clarify_callback(question, choices, multi_select)


def _oneshot_cb(question, choices, multi_select=False):
    """The real oneshot (``hermes -z``) clarify callback."""
    from hermes_cli.oneshot import _oneshot_clarify_callback
    return _oneshot_clarify_callback(question, choices, multi_select)


def _sentinel_cb(sentinel):
    def cb(question, choices, multi_select=False):
        return sentinel
    return cb


class TestGuardFiresOnNoUserSentinels:
    """Every harness-authored no-user answer must be screened."""

    def test_single_query_repro_excludes_authorization_option(self):
        """The exact issue repro: -q turn, Recommended self-authorization
        option first — the auto-decide answer must not offer it as pickable."""
        result = json.loads(clarify_tool(
            "How should the aggregation proceed?",
            choices=[AUTH] + SAFE,
            callback=_single_query_cb,
        ))
        resp = result["user_response"]
        assert isinstance(resp, str)
        assert "[clarify consent guard" in resp
        assert "silence cannot grant authorization" in resp.lower()
        # The rewrite is the audit trail: it must name the question that went
        # unanswered (the -q sentinel embedded it; the guard must not drop it).
        assert "How should the aggregation proceed?" in resp
        # The safe options remain pickable; the authorization one is excluded.
        assert SAFE[0] in resp and SAFE[1] in resp
        assert "excluded" in resp.lower() and AUTH in resp
        # The old behavior — handing every option to the agent's judgment —
        # must be gone.
        assert "using your own judgment and continue" not in resp

    def test_oneshot_callback_rewritten(self):
        result = json.loads(clarify_tool(
            "Format?", choices=[AUTH, SAFE[1]], callback=_oneshot_cb,
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert SAFE[1] in resp
        assert "oneshot mode" not in resp

    def test_timeout_sentinel_rewritten(self):
        result = json.loads(clarify_tool(
            "Format?", choices=[AUTH, "yaml"], callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert "yaml" in resp
        assert TIMEOUT_RESPONSE not in resp

    def test_gateway_timeout_sentinel_rewritten(self):
        result = json.loads(clarify_tool(
            "Proceed?", choices=["allow me to delete the rows", "skip deletion"],
            callback=_sentinel_cb("[user did not respond within 60m]"),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert "skip deletion" in resp

    def test_empty_response_rewritten_when_consent_present(self):
        """The TUI bridge returns "" on timeout/cancel — never a consent grant."""
        result = json.loads(clarify_tool(
            "Proceed?", choices=["grant me full disk access", "read-only mode"],
            callback=_sentinel_cb(""),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert "read-only mode" in resp

    def test_all_options_consent_fails_closed(self):
        """When every option grants authorization, the agent must be told to
        proceed WITHOUT the authorization — never to self-authorize."""
        result = json.loads(clarify_tool(
            "Proceed?",
            choices=["authorize me to run it", "consent to full access"],
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert "do not self-authorize" in resp.lower()
        assert "Pick only from" not in resp

    def test_multi_select_sentinel_guard_is_scalar(self):
        """The guard answer must survive _clean_answer's multi-select list
        parsing — it stays one scalar instruction string."""
        result = json.loads(clarify_tool(
            "Which actions?", choices=[AUTH, SAFE[0]], multi_select=True,
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert isinstance(resp, str)
        assert "[clarify consent guard" in resp

    def test_batch_timeout_blank_is_screened(self):
        """Layer 7: on a batch timeout, the unanswered entry's blank response
        must go through the consent guard — a walk-away is a no-user
        auto-decide, not a deliberate human skip."""
        def batch_cb(question, choices, questions=None, **kw):
            # Batch-capable callback: answers the FIRST question, times out
            # the rest (the tui_gateway bridge carries strings only).
            return json.dumps({"answers": {"q0": TIMEOUT_RESPONSE}, "timed_out": True})
        result = json.loads(clarify_tool(
            "batch",
            questions=[
                {"question": "Proceed?", "choices": [AUTH, SAFE[1]]},
                {"question": "Format?", "choices": [AUTH, "yaml"]},
            ],
            callback=batch_cb,
        ))
        assert result.get("timed_out") is True
        resp0 = result["responses"][0]["user_response"]
        resp1 = result["responses"][1]["user_response"]
        assert "[clarify consent guard" in resp0
        # The blank-after-timeout entry is screened too, not passed as "".
        assert "[clarify consent guard" in resp1
        assert "yaml" in resp1

    def test_batch_deliberate_skip_stays_blank(self):
        """Without timed_out, a blank is a deliberate human skip — untouched."""
        def batch_cb(question, choices, questions=None, **kw):
            return json.dumps({"answers": {"q0": "no aggregation, layout only"}})
        result = json.loads(clarify_tool(
            "batch",
            questions=[
                {"question": "Proceed?", "choices": [AUTH, SAFE[1]]},
                {"question": "Format?", "choices": [AUTH, "yaml"]},
            ],
            callback=batch_cb,
        ))
        assert "timed_out" not in result
        assert result["responses"][1]["user_response"] == ""

    def test_recommended_label_does_not_hide_the_verb(self):
        """Screening sees the bare choices, but must also tolerate the
        "(Recommended)" presentation label."""
        result = json.loads(clarify_tool(
            "Proceed?",
            choices=[f"{AUTH} (Recommended)", SAFE[1]],
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        assert SAFE[1] in resp


class TestGuardDoesNotOverreach:
    """The guard must not break innocuous auto-decide or real answers."""

    def test_innocuous_choices_sentinel_passthrough(self):
        result = json.loads(clarify_tool(
            "Format?", choices=["json", "yaml"], callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert result["user_response"] == TIMEOUT_RESPONSE

    def test_open_ended_sentinel_passthrough(self):
        result = json.loads(clarify_tool(
            "Which timezone?", callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert result["user_response"] == TIMEOUT_RESPONSE

    def test_real_user_answer_never_rewritten(self):
        """A human clicking the authorization option IS consent — recorded."""
        answer = "authorize me to compute these 9 rows directly"
        result = json.loads(clarify_tool(
            "Proceed?", choices=[AUTH, SAFE[1]], callback=_sentinel_cb(answer),
        ))
        assert result["user_response"] == answer

    def test_near_miss_words_do_not_trigger(self):
        """Word boundaries: "authorized personnel", "pre-consented" etc. are
        not consent requests from the user."""
        result = json.loads(clarify_tool(
            "Which path?",
            choices=["use the authorized-personnel roster", "manual entry"],
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert result["user_response"] == TIMEOUT_RESPONSE

    def test_config_gate_restores_old_behavior(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"clarify": {"auto_decide_allow_consent": True}},
        )
        result = json.loads(clarify_tool(
            "Proceed?", choices=[AUTH, SAFE[1]], callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert result["user_response"] == TIMEOUT_RESPONSE


class TestRevision107279Review:
    """Regressions from andrexibiza's review of PR#107279 (2026-09-10):

    P1-1: the opt-out resolver must fail closed on non-bool values —
    ``bool("false")`` is True, so a quoted/malformed YAML value must not
    disable the consent guard (silence must not regain auto-authorization
    through a typo).

    P1-2: the consent classifier must cover the explicit phrases #107265
    (Finn763) already recognizes — approve/approval, permit, let me,
    self-approve, bypass, and the CJK 授权/批准/准许 — so a no-user sentinel
    with e.g. "approve this action" cannot fall through to legacy
    auto-decide.
    """

    # --- P1-1: typed authority for the opt-out -----------------------------

    def test_optout_quoted_false_string_stays_fail_closed(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"clarify": {"auto_decide_allow_consent": "false"}},
        )
        result = json.loads(clarify_tool(
            "Proceed?", choices=[AUTH, SAFE[1]], callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp  # guard armed: "false" is not True
        assert SAFE[1] in resp

    def test_optout_nonempty_junk_string_stays_fail_closed(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"clarify": {"auto_decide_allow_consent": "disabled"}},
        )
        result = json.loads(clarify_tool(
            "Proceed?", choices=[AUTH, SAFE[1]], callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert "[clarify consent guard" in result["user_response"]

    def test_optout_numeric_and_container_values_stay_fail_closed(self, monkeypatch):
        for bad in (0, 1, 2, 0.0, ["true"], {"enabled": True}, None):
            monkeypatch.setattr(
                "hermes_cli.config.load_config",
                lambda bad=bad: {"clarify": {"auto_decide_allow_consent": bad}},
            )
            result = json.loads(clarify_tool(
                "Proceed?", choices=[AUTH, SAFE[1]], callback=_sentinel_cb(TIMEOUT_RESPONSE),
            ))
            assert "[clarify consent guard" in result["user_response"], repr(bad)

    def test_optout_resolver_contract_on_malformed_values(self):
        """Only the literal boolean True disables the guard; every other
        type/value keeps it armed."""
        from tools.clarify_tool import resolve_auto_decide_allow_consent as resolve
        for bad in ("false", "true", "disabled", "yes", 1, 0, 2, 0.0, 1.0,
                    ["true"], {"v": True}, None, (), float("nan")):
            assert resolve({"clarify": {"auto_decide_allow_consent": bad}}) is False, repr(bad)
        assert resolve({"clarify": {"auto_decide_allow_consent": True}}) is True

    # --- P1-2: classifier convergence with #107265 --------------------------

    def test_approve_phrases_now_guarded(self):
        """'approve this action' fell through to legacy auto-decide before the
        revision — silence could still become approval."""
        result = json.loads(clarify_tool(
            "Proceed?", choices=["approve this action", "skip"],
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        resp = result["user_response"]
        assert "[clarify consent guard" in resp
        # the consent option is named — but only as EXCLUDED, never pickable
        assert "Excluded" in resp and "approve this action" in resp
        assert "skip" in resp  # safe option stays pickable

    def test_permit_letme_bypass_phrases_now_guarded(self):
        for consent, safe in (
            ("permit me to write outside the repo", "stay inside the workdir"),
            ("let me bypass the sandbox", "retry in-sandbox"),
            ("bypass the approval gate", "wait for the human"),
            ("self-approve the deployment", "queue it for review"),
            ("approval to delete the rows", "skip deletion"),
        ):
            result = json.loads(clarify_tool(
                "Proceed?", choices=[consent, safe], callback=_sentinel_cb(TIMEOUT_RESPONSE),
            ))
            resp = result["user_response"]
            assert "[clarify consent guard" in resp, consent
            assert safe in resp, consent

    def test_cjk_consent_phrases_now_guarded(self):
        """授权 / 批准 / 准许 — the #107265 EN+ZH coverage contract."""
        for consent, safe in (
            ("授权我直接计算这些行", "仅排版不加总"),
            ("批准删除这些数据行", "跳过删除"),
            ("准许我绕过审批", "等待人工确认"),
        ):
            result = json.loads(clarify_tool(
                "如何处理?", choices=[consent, safe], callback=_sentinel_cb(TIMEOUT_RESPONSE),
            ))
            resp = result["user_response"]
            assert "[clarify consent guard" in resp, consent
            assert safe in resp, consent

    def test_british_spelling_authorise_detected(self):
        from tools.clarify_tool import _is_consent_semantic
        assert _is_consent_semantic("authorise me to proceed")

    def test_near_misses_still_pass_through_after_widening(self):
        """Widening must not catch approval-adjacent nouns that are not consent
        requests: 'improve performance', 'approved-by-default roster' style
        hyphenated near-misses stay pickable."""
        from tools.clarify_tool import _is_consent_semantic
        for text in (
            "use the authorized-personnel roster",
            "the pre-consented form",
            "json",
            "Rebase",
            "Improve performance",
            "no aggregation, layout only",
        ):
            assert not _is_consent_semantic(text), text

    def test_widening_guard_does_not_break_innocuous_passthrough(self):
        result = json.loads(clarify_tool(
            "Format?", choices=["json", "yaml", "table", "Rebase first"],
            callback=_sentinel_cb(TIMEOUT_RESPONSE),
        ))
        assert result["user_response"] == TIMEOUT_RESPONSE


class TestConfigSurface:
    """`clarify.auto_decide_allow_consent` — fail-closed by default."""

    def test_resolver_defaults_false(self):
        from tools.clarify_tool import resolve_auto_decide_allow_consent
        assert resolve_auto_decide_allow_consent({}) is False
        assert resolve_auto_decide_allow_consent({"clarify": {}}) is False
        assert resolve_auto_decide_allow_consent(
            {"clarify": {"auto_decide_allow_consent": True}}) is True

    def test_default_config_ships_fail_closed(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        clarify_cfg = DEFAULT_CONFIG.get("clarify") or {}
        assert clarify_cfg.get("auto_decide_allow_consent") is False


class TestSentinelDetection:
    """The sentinel list mirrors what the harness actually produces."""

    def test_harness_sentinels_detected(self):
        from tools.clarify_tool import _is_auto_decide_sentinel
        for s in (
            TIMEOUT_RESPONSE,
            "",
            "[single-query mode: no user available to answer 'Q'. Pick the best option ...]",
            "[oneshot mode: no user available. Pick the best option ...]",
            "[user did not respond within 60m]",
            "[clarify prompt could not be delivered]",
        ):
            assert _is_auto_decide_sentinel(s), s
        assert _is_auto_decide_sentinel(None) is False

    def test_consent_verb_detection(self):
        from tools.clarify_tool import _is_consent_semantic
        for text in (
            "authorize me to compute these rows",
            "Authorize the agent to proceed",
            "allow me to delete the file",
            "consent to data aggregation",
            "I consent to sharing logs",
            "grant me full disk access",
            "give me permission to run it",
            "permission to proceed without asking",
        ):
            assert _is_consent_semantic(text), text
        for text in (
            "use the authorized-personnel roster",
            "the pre-consented form",
            "configure A2A peer then hand off",
            "no aggregation, layout only",
            "json",
        ):
            assert not _is_consent_semantic(text), text
