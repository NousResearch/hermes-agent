from __future__ import annotations

import json
import http.client
import os
from pathlib import Path

from tools.task_engine_contracts import (
    CANONICAL_STAGES,
    CONTROLLER_ACCEPTANCE,
    ENGINE_DECISION,
    ENGINE_RESEARCH,
    ENGINE_RESEARCH_DECISION,
    FINAL_CONTROLLER,
    GEMINI_HIGH,
    GEMINI_PRO_HIGH,
    GEMMA431B,
    LLAMA70B,
    NEMOTRON120B,
    PIPELINE_BLOCKED,
    PIPELINE_COMPLETE,
    PIPELINE_INCOMPLETE,
    QWEN72B,
    R1_32B,
    StageSpec,
    build_engine_contract,
    detect_task_engine_mode,
    planned_outputs,
    render_final_markdown,
    validate_pipeline,
)
from tools.task_engine_executors import (
    GEMMA431B_ACTUAL_MODEL_DEFAULT,
    LLAMA70B_ACTUAL_MODEL_DEFAULT,
    LocalTaskEngineExecutor,
    NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
    QWEN72B_ACTUAL_MODEL_DEFAULT,
    R1_ACTUAL_MODEL_DEFAULT,
    resolve_gemma431b_omlx_model_alias,
    resolve_llama70b_omlx_model_alias,
    resolve_agy_model_alias,
    resolve_nemotron120b_omlx_model_alias,
    resolve_qwen72b_omlx_model_alias,
    resolve_r1_omlx_model_alias,
    _final_controller_packet_from_artifacts,
    run_decision_final_smoke,
    run_agy_preflight,
    run_omlx_preflight,
    run_research_decision_alternative_generator_smoke,
    run_research_decision_convergence_smoke,
    run_research_decision_evidence_judge_smoke,
    run_research_decision_external_calibration_smoke,
    run_research_decision_final_controller_smoke,
    run_research_decision_intelligence_smoke,
    run_research_decision_insight_harvester_smoke,
    run_research_decision_l1_l10_smoke,
    run_research_decision_l1_l11_smoke,
    run_research_decision_l1_l12_smoke,
    run_research_decision_l1_l13_smoke,
    run_research_decision_l1_l14_smoke,
    run_research_decision_l1_l15_smoke,
    run_research_decision_l1_l16_smoke,
    run_research_decision_l1_l7_smoke,
    run_research_decision_l1_l8_smoke,
    run_research_decision_l1_l9_smoke,
    run_research_decision_premise_auditor_smoke,
    run_research_decision_structure_mapper_smoke,
    run_research_decision_supplementary_search_smoke,
    run_research_l2_5_codex_handoff_smoke,
    run_research_l1_l3_smoke,
    run_research_l1_l4_smoke,
    run_research_l1_l5_smoke,
    run_research_l3_synthesis_smoke,
    run_research_l4_gemini_audit_smoke,
    run_research_l5_acceptance_smoke,
    run_research_l1_l2_smoke,
)
from tools.research_pipeline_runner import research_pipeline_runner
from tools.registry import registry
from tools.task_mode_runtime import TaskModeRuntime, TaskModeState
from tools.task_engine_runner import (
    DIRECT_LEGACY_RESEARCH_DECISION_FULL,
    LEGACY_RESEARCH_DECISION_BANNED_TERMS,
    TERMINOLOGY_LEAKAGE,
    apply_legacy_research_decision_term_guard,
    audit_legacy_research_decision_terms,
    task_engine_runner,
)
from toolsets import resolve_toolset
from work import nightly_task_engine_runner as nightly_runner
from work import stress_task_engine_adhd as stress_runner


ADHD_PROMPT = """这是一个研究决策任务。

ADHD 儿童最新的研究进展和治疗方案；
与我儿子情况相匹配的梳理；
一些建议，以及长期发展的路线。
我想知道是否要主动干预，要主动干预到什么程度？
"""


def _complete_research_evidence_packet_text() -> str:
    return "\n".join(
        [
            "research_evidence_packet",
            "verdict: ACCEPTED",
            "accepted: true",
            "checked_stages: [L1_gemini_search, L2_ddgs_supplement, L2_5_codex_evidence_organizer, L3_r1_synthesis, L4_gemini_audit]",
            "missing_or_invalid_artifacts: []",
            "critical_defects: []",
            "noncritical_defects: []",
            "verification_required: []",
            "evidence_gaps: [long_horizon_transfer_gap]",
            "handoff_caveats: []",
            "audit_summary: L4 audit accepted the compact evidence packet.",
            "evidence_packet_ready_for_decision: true",
            "",
            "## evidence_strength",
            "Evidence strength: strong for stable current evidence, medium for mechanism transfer, weak for individual long-horizon forecasts.",
            "",
            "## claim_table",
            "- claim_id: C1",
            "  claim_text: Verified source-backed claim about current evidence and stable mechanism boundaries.",
            "  epistemic_tier: evidence_supported",
            "  evidence_strength: medium",
            "  source_anchors: S1 (full_text_verified; peer_reviewed_source; https://example.test/source)",
            "  applicability_boundary: Applies within the source population and measurement context.",
            "  counter_signal_or_failure_condition: Contradictory source passages or failed transfer should downgrade it.",
            "  evidence_gap: long_horizon_transfer_gap",
            "  decision_use: can_support_decision",
            "  notes: Fixture claim table row for L5 contract tests.",
            "",
            "## controversy",
            "Controversy: applicability depends on context, population differences, measurement choices, and whether mechanisms transfer to the target scenario.",
            "",
            "## evidence_gap",
            "Evidence gap: direct individual longitudinal evidence and exact future-environment evidence remain unavailable for this decision context.",
            "",
            "## evidence_supported",
            "Evidence supported: current research artifacts and audited synthesis support bounded claims about stable mechanisms and observed evidence.",
            "",
            "## reasonable_inference",
            "Reasonable inference: evidence may be connected to the decision through explicit mechanism chains and stated uncertainty boundaries.",
            "",
            "## foresight_hypothesis",
            "Foresight hypothesis: future-facing claims are conditional hypotheses, not settled facts, and require counter-signals and failure conditions.",
            "",
            "scope: acceptance gate plus compact evidence packet; no raw artifact dump and no user-facing advice.",
        ]
    )


COMPLETE_EXTERNAL_CALIBRATION_FIXTURE = (
    "calibration_scope\n" + "Scope text. " * 80
    + "claim_strength_table\n| Claim | Strength | Notes |\n| --- | --- | --- |\n"
    "| A | supported | grounded in packet |\n"
    "| B | plausible | bounded inference |\n"
    "| C | speculative | needs confirmation |\n"
    "| D | contradicted | conflict noted |\n"
    + "over_inference_checks\n" + "No overreach. " * 60
    + "contradiction_checks\n" + "Contradictions are labeled. " * 60
    + "calibration_verdict\nverdict: calibrated for final controller handoff.\n"
    "handoff_notes_for_final_controller\nUse calibrated claims only.\n"
)


COMPLETE_EXTERNAL_CALIBRATION_MINIMUM_FIELDS = "\n".join(
    [
        "calibration_verdict",
        "Verdict: plausible and supported in bounded parts; speculative future-facing claims must stay conditional.",
        "",
        "agreement_points",
        "Supported agreement points: convergence correctly preserves structural inversion, mechanism chains, and counter-signals.",
        "",
        "disagreement_or_risk_points",
        "Plausible risk points: some claims may overstate transfer from current ADHD evidence to future AI environments.",
        "",
        "missing_considerations",
        "Missing considerations: direct longitudinal evidence, school context variation, individual developmental differences, and tool-quality drift.",
        "",
        "final_adjustment_recommendation",
        "Final adjustment recommendation: keep the convergence conclusion but label long-horizon individual predictions as speculative.",
    ]
)


def test_research_decision_schema_has_exact_16_stage_order():
    names = [stage.stage_name for stage in CANONICAL_STAGES[ENGINE_RESEARCH_DECISION]]
    assert names == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "convergence_report",
        "external_calibration",
        "final_controller_report",
    ]


def test_auto_detection_keeps_ordinary_chat_out_of_engine_modes():
    assert detect_task_engine_mode(ADHD_PROMPT) == ENGINE_RESEARCH_DECISION
    assert detect_task_engine_mode("帮我改一下这个按钮的颜色") is None

    result = json.loads(task_engine_runner(query="帮我改一下这个按钮的颜色", mode="AUTO"))
    assert result["status"] == "not_applicable"
    assert result["ordinary_chat_model_replaced"] is False


def test_legacy_research_runner_blocks_heavy_task_modes():
    result = json.loads(research_pipeline_runner(ADHD_PROMPT))
    assert result["pipeline_status"] == "PIPELINE_BLOCKED"
    assert result["required_tool"] == "task_engine_runner"
    assert result["detected_mode"] == ENGINE_RESEARCH_DECISION


def test_task_engine_runner_is_available_for_real_entrypoint():
    definitions = registry.get_definitions({"task_engine_runner"}, quiet=True)

    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == "task_engine_runner"


def test_hermes_cli_toolset_exposes_task_engine_but_not_legacy_runner():
    tools = set(resolve_toolset("hermes-cli"))

    assert "task_engine_runner" in tools
    assert "research_pipeline_runner" not in tools


def test_research_decision_entry_uses_task_engine_runner_without_route_card_loop(tmp_path: Path):
    result = json.loads(
        task_engine_runner(
            query=ADHD_PROMPT,
            mode="AUTO",
            action="dry-run",
            base_dir=str(tmp_path / "entry"),
        )
    )

    assert detect_task_engine_mode(ADHD_PROMPT) == ENGINE_RESEARCH_DECISION
    assert result["status"] == "ok"
    assert result["mode"] == ENGINE_RESEARCH_DECISION
    assert result["plan"]["stage_count"] == 16
    serialized = json.dumps(result, ensure_ascii=False)
    assert "ROUTE_CARD_NOT_CONFIRMED" not in serialized
    assert "route_card" not in serialized.lower()
    assert "deepseek-v4-flash" not in serialized.lower()
    assert "final_controller_report" in serialized


def test_research_decision_full_is_archived_by_default(tmp_path: Path):
    result = json.loads(
        task_engine_runner(
            query=ADHD_PROMPT,
            mode=ENGINE_RESEARCH_DECISION,
            action="full",
            base_dir=str(tmp_path / "archived"),
        )
    )

    assert result["status"] == "blocked"
    assert result["pipeline_status"] == "PIPELINE_BLOCKED"
    assert result["blocked_reason"] == DIRECT_LEGACY_RESEARCH_DECISION_FULL
    assert result["entrypoint_guard"] == DIRECT_LEGACY_RESEARCH_DECISION_FULL
    assert result["mode"] == ENGINE_RESEARCH_DECISION
    assert result["two_step_recommendation"] == [
        "RESEARCH full -> research_evidence_packet.md",
        "DECISION full with research_packet_path=<path to research_evidence_packet.md>",
    ]
    assert result["allow_override"]["function_arg"] == "allow_archived_research_decision=True"
    serialized = json.dumps(result, ensure_ascii=False)
    assert not audit_legacy_research_decision_terms(serialized)


def test_research_decision_correct_two_step_wording_has_no_banned_terms():
    payload = {
        "status": "PASS",
        "accepted_as": "S02 two-step E2E validation",
        "path": "RESEARCH full + DECISION full",
        "stage_count": 16,
        "source": "current-run validation",
    }

    assert not audit_legacy_research_decision_terms(payload)


def test_research_decision_stage_count_16_alone_is_allowed():
    assert not audit_legacy_research_decision_terms({"stage_count": 16, "message": "stage_count: 16"})


def test_research_decision_banned_terms_are_blocked_outside_audit_context():
    payload = {"status": "PASS", "summary": "RESEARCH_DECISION 16-stage smoke completed"}

    violations = audit_legacy_research_decision_terms(payload)
    guarded = apply_legacy_research_decision_term_guard(payload)

    assert violations[0]["term"] == "RESEARCH_DECISION 16-stage smoke"
    assert guarded["status"] == TERMINOLOGY_LEAKAGE
    assert guarded["pipeline_status"] == "PIPELINE_BLOCKED"
    assert guarded["blocked_reason"] == TERMINOLOGY_LEAKAGE


def test_research_decision_banned_terms_allowed_only_in_audit_context():
    payload = {"legacy_term_audit": {"quoted": list(LEGACY_RESEARCH_DECISION_BANNED_TERMS)}}

    assert not audit_legacy_research_decision_terms(payload)
    assert not audit_legacy_research_decision_terms(
        {"summary": "direct RESEARCH_DECISION full"},
        context="banned_term_check",
    )


def test_research_decision_dry_run_remains_available_when_archived(tmp_path: Path):
    result = json.loads(
        task_engine_runner(
            query=ADHD_PROMPT,
            mode=ENGINE_RESEARCH_DECISION,
            action="dry-run",
            base_dir=str(tmp_path / "dry"),
        )
    )

    assert result["status"] == "ok"
    assert result["mode"] == ENGINE_RESEARCH_DECISION
    assert result["plan"]["stage_count"] == 16


def test_task_mode_gate_allows_task_engine_and_blocks_legacy_paths():
    runtime = TaskModeRuntime()
    runtime._state = TaskModeState.INIT_LOCKED

    assert runtime.preflight("task_engine_runner") is None
    legacy_block = runtime.preflight("research_pipeline_runner")
    web_block = runtime.preflight("web_search")
    assert legacy_block and "task_engine_runner" in legacy_block
    assert web_block and "task_engine_runner" in web_block

    runtime._state = TaskModeState.ROUTE_CARD_SUBMITTED
    assert runtime.preflight("task_engine_runner") is None
    route_card_legacy_block = runtime.preflight("research_pipeline_runner")
    assert route_card_legacy_block and "task_engine_runner" in route_card_legacy_block


def test_ordinary_chat_does_not_trigger_task_engine_runner():
    result = json.loads(task_engine_runner(query="帮我改一下这个按钮的颜色", mode="AUTO"))

    assert detect_task_engine_mode("帮我改一下这个按钮的颜色") is None
    assert result["status"] == "not_applicable"
    assert result["ordinary_chat_model_replaced"] is False


def test_contract_is_72b_first_but_not_global_chat_replacement():
    contract = build_engine_contract(ENGINE_DECISION, "是否应该主动干预")
    assert contract["controller"]["model"] == "Qwen72B"
    assert contract["controller"]["must_not_replace_ordinary_chat"] is True
    assert contract["schema"]["ordinary_chat_model_replaced"] is False


def test_dry_run_generates_canonical_stage_record_plans_for_all_modes(tmp_path: Path):
    expected_counts = {
        ENGINE_RESEARCH: 6,
        ENGINE_DECISION: 10,
        ENGINE_RESEARCH_DECISION: 16,
    }
    for mode, expected_count in expected_counts.items():
        result = json.loads(
            task_engine_runner(
                query=ADHD_PROMPT,
                mode=mode,
                action="dry-run",
                base_dir=str(tmp_path / mode),
            )
        )
        stages = result["plan"]["stages"]
        assert result["status"] == "ok"
        assert result["plan"]["model_calls_made"] is False
        assert len(stages) == expected_count
        assert [stage["stage_name"] for stage in stages] == [
            spec.stage_name for spec in CANONICAL_STAGES[mode]
        ]
        assert [stage["model"] for stage in stages] == [
            spec.model for spec in CANONICAL_STAGES[mode]
        ]
        assert all(stage["created_in_current_run"] is False for stage in stages)
        assert all(stage["valid_for_pipeline"] is False for stage in stages)


def test_structure_mapper_canonical_binding_and_output(tmp_path: Path):
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][8]
    evidence_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    premise_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][10]
    alternative_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][11]
    insight_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][12]
    convergence_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][13]
    calibration_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    final_stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][15]

    assert stage.stage_name == "structure_mapper"
    assert stage.owner == QWEN72B
    assert stage.model == QWEN72B
    assert stage.required_outputs == ("structure_mapper.md",)
    assert evidence_stage.stage_name == "evidence_judge"
    assert evidence_stage.owner == NEMOTRON120B
    assert evidence_stage.model == NEMOTRON120B
    assert evidence_stage.required_outputs == ("evidence_judge.md",)
    assert premise_stage.stage_name == "premise_auditor"
    assert premise_stage.owner == LLAMA70B
    assert premise_stage.model == LLAMA70B
    assert premise_stage.required_outputs == ("premise_auditor.md",)
    assert alternative_stage.stage_name == "alternative_generator"
    assert alternative_stage.owner == GEMMA431B
    assert alternative_stage.model == GEMMA431B
    assert alternative_stage.required_outputs == ("alternative_generator.md",)
    assert insight_stage.stage_name == "insight_harvester"
    assert insight_stage.owner == GEMMA431B
    assert insight_stage.model == GEMMA431B
    assert insight_stage.required_outputs == ("insight_harvester.md",)
    assert convergence_stage.stage_name == "convergence_report"
    assert convergence_stage.owner == R1_32B
    assert convergence_stage.model == R1_32B
    assert convergence_stage.required_outputs == ("convergence_report.md",)
    assert calibration_stage.stage_name == "external_calibration"
    assert calibration_stage.owner == "GPT Bridge or Gemini/agy"
    assert calibration_stage.model == "GPT Bridge or Gemini/agy"
    assert calibration_stage.required_outputs == ("external_calibration.md",)
    assert final_stage.stage_name == "final_controller_report"
    assert final_stage.owner == "Controller"
    assert final_stage.model == FINAL_CONTROLLER
    assert final_stage.required_outputs == ("final_decision_report.md",)

    research = json.loads(
        task_engine_runner(
            query=ADHD_PROMPT,
            mode=ENGINE_RESEARCH,
            action="dry-run",
            base_dir=str(tmp_path / "research"),
        )
    )
    l1 = research["plan"]["stages"][0]
    l2 = research["plan"]["stages"][1]
    handoff = research["plan"]["stages"][2]
    l4 = research["plan"]["stages"][4]
    assert l1["owner"] == GEMINI_HIGH
    assert l1["model"] == GEMINI_HIGH
    assert l1["executor_model"] == GEMINI_HIGH
    assert l4["stage_name"] == "L4_gemini_audit"
    assert l4["owner"] == GEMINI_PRO_HIGH
    assert l4["model"] == GEMINI_PRO_HIGH
    assert l4["executor_model"] == GEMINI_PRO_HIGH
    assert l1["artifact_path"].endswith("L1_gemini_search/source_candidates.json")
    assert l2["artifact_path"].endswith("L2_ddgs_supplement/ddgs_gap_sources.json")
    assert research["plan"]["stages"][5]["artifact_path"].endswith(
        "L5_deepseek_acceptance/research_evidence_packet.md"
    )
    assert "evidence_runner_001.request.json" in "\n".join(handoff["outputs"].values())

    research_decision = json.loads(
        task_engine_runner(
            query=ADHD_PROMPT,
            mode=ENGINE_RESEARCH_DECISION,
            action="dry-run",
            base_dir=str(tmp_path / "research_decision"),
        )
    )
    rd_l4 = research_decision["plan"]["stages"][4]
    assert rd_l4["stage_name"] == "L4_gemini_audit"
    assert rd_l4["owner"] == GEMINI_PRO_HIGH
    assert rd_l4["model"] == GEMINI_PRO_HIGH


def test_simulated_run_validates_and_renders_all_modes(tmp_path: Path):
    for mode in (ENGINE_RESEARCH, ENGINE_DECISION, ENGINE_RESEARCH_DECISION):
        result = json.loads(
            task_engine_runner(
                query=ADHD_PROMPT,
                mode=mode,
                action="simulated-run",
                base_dir=str(tmp_path / mode),
            )
        )
        assert result["status"] == "ok"
        assert result["pipeline_status"] == PIPELINE_COMPLETE
        assert result["validation"]["valid"] is True
        assert result["validation"]["stage_count"] == len(CANONICAL_STAGES[mode])
        assert "pipeline_status=PIPELINE_COMPLETE" in result["markdown"]
        if mode == ENGINE_RESEARCH:
            assert "research_evidence_packet" in result["markdown"]
            assert "accepted: true" in result["markdown"]
        else:
            assert "FINAL CONTROLLER BODY" in result["markdown"]


def test_complete_research_decision_run_validates_and_renders_only_final(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH_DECISION)
    validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=tmp_path)

    assert validation["valid"] is True
    assert validation["pipeline_status"] == PIPELINE_COMPLETE
    assert validation["divergence_unique_model_count"] == 4

    markdown = render_final_markdown(ENGINE_RESEARCH_DECISION, run, validation, base_dir=tmp_path)
    assert "entered_engine_run_pipeline=true" in markdown
    assert "pipeline_status=PIPELINE_COMPLETE" in markdown
    assert "pipeline_validation.valid=true" in markdown
    assert "delegation_used=false" in markdown
    assert "FINAL CONTROLLER BODY" in markdown
    assert "R1 CONVERGENCE BODY" not in markdown
    assert "Compact Pipeline Trace:" in markdown


def test_wrong_divergence_model_fails_closed(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_DECISION)
    for stage in run["stages"]:
        if stage["stage_name"] == "structure_mapper":
            stage["model"] = "R1-32B"
            break

    validation = validate_pipeline(ENGINE_DECISION, run, base_dir=tmp_path)

    assert validation["valid"] is False
    assert validation["pipeline_status"] == PIPELINE_BLOCKED
    assert any("structure_mapper:model_mismatch" in error for error in validation["errors"])
    assert any("structure_mapper:r1_forbidden_here" in error for error in validation["errors"])


def test_codex_handoff_requires_request_json_and_outputs(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH)
    request_json = tmp_path / "L2_5_codex_evidence_organizer" / "evidence_runner_001.request.json"
    request_json.unlink()

    validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=tmp_path)

    assert validation["valid"] is False
    assert any(
        "L2_5_codex_evidence_organizer:required_output_missing:evidence_runner_*.request.json"
        in error
        for error in validation["errors"]
    )


def test_legacy_or_missing_artifacts_fail_closed(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH)
    run["stages"][0]["created_in_current_run"] = False
    run["stages"][1]["legacy_contaminated"] = True
    Path(run["stages"][3]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("created_in_current_run_not_true" in error for error in validation["errors"])
    assert any("legacy_contaminated_not_false" in error for error in validation["errors"])
    assert any("artifact_path_not_found" in error for error in validation["errors"])


def test_final_renderer_blocks_pseudo_tool_chain_leakage(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH_DECISION)
    final_artifact = tmp_path / "final_controller_report" / "final_decision_report.md"
    final_artifact.write_text("FINAL CONTROLLER BODY\nweb_search should not leak", encoding="utf-8")

    markdown = render_final_markdown(ENGINE_RESEARCH_DECISION, run, base_dir=tmp_path)

    assert "pipeline_status=PIPELINE_BLOCKED" in markdown
    assert "final_markdown_forbidden_tokens:web_search" in markdown


def test_l1_l2_smoke_can_complete_subset_but_full_pipeline_stays_incomplete(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            assert stage.stage_name == "L1_gemini_search"
            assert model == "Gemini 3.5 Flash (High)"
            return {"source_candidates": [{"title": "fake", "url": "https://example.test"}]}

        def run_ddgs(self, stage, queries):
            assert stage.stage_name == "L2_ddgs_supplement"
            assert queries
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs", "snippet": "ok"}]

    result = run_research_l1_l2_smoke(
        ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    assert [stage["stage_name"] for stage in result["run"]["stages"]] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
    ]
    assert result["run"]["stages"][0]["owner"] == "Gemini 3.5 Flash (High)"
    assert result["run"]["stages"][0]["executor_model"] == "Gemini 3.5 Flash (High)"
    assert all(stage["created_in_current_run"] is True for stage in result["run"]["stages"])
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:L2_5_codex_evidence_organizer" in error for error in result["full_pipeline_validation"]["errors"])


def test_l2_ddgs_uses_l1_targeted_queries_instead_of_full_user_question(tmp_path: Path):
    captured = {}

    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {
                "source_candidates": [
                    {
                        "query_or_url": "AI hardware companion retention Rabbit Plaud 2026",
                        "why_relevant": "Targets retention evidence for standalone AI hardware companions.",
                    },
                    {
                        "source": "Search: consumer AI hardware funding trend 2026",
                        "why_relevant": "Targets investment signal evidence.",
                    },
                    {
                        "source_or_query": "Rabbit R1 Humane AI Pin customer return rates 2026",
                        "why_relevant": "Targets customer satisfaction evidence.",
                    },
                    {
                        "target": "Ray-Ban Meta smart glasses adoption case studies 2026",
                        "why_relevant": "Targets wearable adoption evidence.",
                    },
                ]
            }

        def run_ddgs(self, stage, queries):
            captured["queries"] = list(queries)
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs", "snippet": "ok"}]

    question = "请用 RESEARCH 和 DECISION 管线评估：2026年前后小样本观察到的独立硬件 AI 伴侣设备使用趋势是否真的在上升？"
    result = run_research_l1_l2_smoke(question, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert captured["queries"][0] == "AI hardware companion retention Rabbit Plaud 2026"
    assert captured["queries"][1] == "consumer AI hardware funding trend 2026"
    assert captured["queries"][2] == "Rabbit R1 Humane AI Pin customer return rates 2026"
    assert captured["queries"][3] == "Ray-Ban Meta smart glasses adoption case studies 2026"
    assert question[:120] not in captured["queries"]


def test_l1_l2_smoke_fails_closed_on_first_real_adapter_error(tmp_path: Path):
    class FailingExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            raise RuntimeError("agy unavailable")

    result = run_research_l1_l2_smoke(
        ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FailingExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["pipeline_status"] == PIPELINE_BLOCKED
    assert result["blocked_stage"] == "L1_gemini_search"
    assert result["run"]["stages"][0]["created_in_current_run"] is False
    assert result["run"]["stages"][0]["valid_for_pipeline"] is False


def test_ddgs_startpage_timeout_does_not_block_when_duckduckgo_has_results(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_search(query, *, backend, timeout_s, max_results):
        calls.append((backend, timeout_s, max_results))
        if backend == "startpage":
            raise TimeoutError("startpage timed out")
        if backend == "duckduckgo":
            return [{"title": "CDC", "href": "https://www.cdc.gov/adhd/", "body": "parent training"}]
        return []

    monkeypatch.setenv("HERMES_DDGS_BACKENDS", "startpage,duckduckgo")
    monkeypatch.setenv("HERMES_DDGS_QUERY_TIMEOUT_S", "3")
    monkeypatch.setattr(executors, "_ddgs_search_once", fake_search)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][1]
    hits = LocalTaskEngineExecutor().run_ddgs(stage, ["ADHD parent training children"])

    assert hits == [
        {
            "query": "ADHD parent training children",
            "title": "CDC",
            "url": "https://www.cdc.gov/adhd/",
            "snippet": "parent training",
        }
    ]
    assert ("startpage", 3, 5) in calls
    assert ("duckduckgo", 3, 5) in calls


def test_ddgs_blocks_only_after_all_allowed_backends_have_no_fresh_results(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.setenv("HERMES_DDGS_BACKENDS", "duckduckgo,yahoo")
    monkeypatch.setattr(executors, "_ddgs_search_once", lambda *args, **kwargs: [])

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][1]
    try:
        LocalTaskEngineExecutor().run_ddgs(stage, ["ADHD parent training children"])
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("DDGS should fail closed when all allowed backends are empty")

    assert "DDGS returned no fresh hits" in message
    assert "backends=duckduckgo,yahoo" in message


def test_ddgs_default_backend_list_uses_duckduckgo_brave_yahoo(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.delenv("HERMES_DDGS_BACKENDS", raising=False)

    assert executors._ddgs_backend_list() == ["duckduckgo", "brave", "yahoo"]
    assert "startpage" not in executors._ddgs_backend_list()
    assert "web_search" not in executors._ddgs_backend_list()
    assert "generic_web_search" not in executors._ddgs_backend_list()


def test_ddgs_rejects_web_search_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_DDGS_BACKENDS", "duckduckgo,web_search")

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][1]
    try:
        LocalTaskEngineExecutor().run_ddgs(stage, ["ADHD parent training children"])
    except RuntimeError as exc:
        assert "cannot include web_search fallback" in str(exc)
    else:
        raise AssertionError("web_search must not be accepted as a DDGS fallback")


def test_ddgs_uses_broad_first_adhd_queries():
    import tools.task_engine_executors as executors

    queries = executors._ddgs_queries(ADHD_PROMPT)

    assert queries[0] == "ADHD parent training children"
    assert all(len(query) < 120 for query in queries)


def test_agy_alias_resolver_rejects_forbidden_ccpa(monkeypatch):
    monkeypatch.setenv("HERMES_AGY_GEMINI_HIGH_MODEL", "CCPA")
    try:
        resolve_agy_model_alias(GEMINI_HIGH)
    except RuntimeError as exc:
        assert "forbidden CCPA" in str(exc)
    else:
        raise AssertionError("CCPA alias should be rejected")

    monkeypatch.delenv("HERMES_AGY_GEMINI_HIGH_MODEL", raising=False)
    monkeypatch.setenv("HERMES_AGY_GEMINI_PRO_HIGH_MODEL", "CCPA")
    try:
        resolve_agy_model_alias(GEMINI_PRO_HIGH)
    except RuntimeError as exc:
        assert "forbidden CCPA" in str(exc)
    else:
        raise AssertionError("CCPA alias should be rejected for Pro High")


def test_agy_alias_resolver_supports_flash_and_pro_high(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HERMES_AGY_GEMINI_HIGH_MODEL", raising=False)
    monkeypatch.delenv("HERMES_AGY_GEMINI_PRO_HIGH_MODEL", raising=False)
    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))

    assert resolve_agy_model_alias(GEMINI_HIGH) == GEMINI_HIGH
    assert resolve_agy_model_alias(GEMINI_PRO_HIGH) == GEMINI_PRO_HIGH


def test_agy_adapter_matches_legacy_call_shape(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "forced failure"})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setenv("GEMINI_DIR", ".gemini")
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()
    try:
        executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("forced AGY failure should raise")

    assert len(calls) == 2
    assert calls[0][0] == ["/opt/homebrew/bin/agy", "models"]
    command, kwargs = calls[1]
    assert command[:2] == ["/opt/homebrew/bin/agy", "--log-file"]
    assert command[2].startswith("/private/tmp/agy-")
    assert command[3:8] == ["--model", GEMINI_HIGH, "-p", "prompt", "--print-timeout"]
    assert command[8] == "600s"
    assert kwargs["timeout"] == 630
    assert Path(kwargs["cwd"]).is_absolute()
    assert ".hermes" not in Path(kwargs["cwd"]).parts
    assert kwargs["env"]["GEMINI_DIR"] == str(Path.home() / ".gemini")
    assert "canonical_model='Gemini 3.5 Flash (High)'" in message
    assert "actual_model='Gemini 3.5 Flash (High)'" in message
    assert "log_file='/private/tmp/agy-" in message
    assert "agy_cwd=" in message
    assert "elapsed_seconds=" in message
    assert "command=" in message
    assert "forced failure" in message

    monkeypatch.setenv("HERMES_AGY_GEMINI_HIGH_MODEL", GEMINI_HIGH)
    monkeypatch.setenv("HERMES_AGY_GEMINI_PRO_HIGH_MODEL", GEMINI_PRO_HIGH)
    assert resolve_agy_model_alias(GEMINI_HIGH) == GEMINI_HIGH
    assert resolve_agy_model_alias(GEMINI_PRO_HIGH) == GEMINI_PRO_HIGH


def test_agy_timeout_is_layered_by_stage():
    import tools.task_engine_executors as executors

    assert executors._agy_timeout_for_stage(CANONICAL_STAGES[ENGINE_RESEARCH][0]) == 600
    assert executors._agy_timeout_for_stage(CANONICAL_STAGES[ENGINE_RESEARCH][4]) == 600
    assert executors._agy_timeout_for_stage(CANONICAL_STAGES[ENGINE_DECISION][0]) == 360
    assert executors._agy_timeout_for_stage(CANONICAL_STAGES[ENGINE_DECISION][1]) == 240
    external_stage = CANONICAL_STAGES[ENGINE_DECISION][8]
    assert external_stage.stage_name == "external_calibration"
    assert executors._agy_timeout_for_stage(external_stage) == 600


def test_decision_agy_stage_wrapper_timeout_allows_provider_timeout_diagnostic():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][0]
    assert stage.stage_name == "intelligence_layer"
    assert executors._decision_stage_timeout_s(stage) > executors._agy_timeout_for_stage(stage) + 30


def test_intelligence_layer_prompt_is_text_only_and_path_free(tmp_path: Path):
    import tools.task_engine_executors as executors

    packet = tmp_path / "research_evidence_packet.md"
    packet.write_text(_complete_research_evidence_packet_text(), encoding="utf-8")

    prompt = executors._decision_intelligence_prompt(
        "Should we shift sales motion?",
        base_dir=tmp_path / "run",
        research_packet_path=packet,
    )

    assert "TEXT ONLY CONTRACT" in prompt
    assert "Do not call tools" in prompt
    assert "write files" in prompt
    assert "Current run root" not in prompt
    assert "research_packet_path:" not in prompt
    assert str(tmp_path) not in prompt
    assert "research_packet_digest:" in prompt


def test_agy_stage_timeout_diagnostic_classifies_invalid_artifact_tool_call(tmp_path: Path):
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][0]
    log_file = tmp_path / "agy.log"
    log_file.write_text(
        "model output error: invalid tool call error (invalid_args) "
        "/tmp/out/decision/intelligence_layer/intelligence_layer_report.md "
        "is not a valid artifact path; artifacts must be in /Users/example/.gemini/brain",
        encoding="utf-8",
    )
    executor = LocalTaskEngineExecutor()
    executor.last_agy_diagnostics[stage.stage_name] = {
        "prompt_chars": 4117,
        "log_file": str(log_file),
        "error_type": "in_progress",
    }

    diagnostic = executors._classify_agy_stage_timeout(
        stage,
        executor=executor,
        started=executors.time.time() - 1,
        timeout_s=executors._decision_stage_timeout_s(stage),
    )

    assert diagnostic["blocked_reason"] == "provider_tool_call_invalid_artifact_path"
    assert diagnostic["invalid_artifact_tool_call"] is True
    assert diagnostic["provider_timeout"] is False
    assert diagnostic["wrapper_timeout"] is False
    assert diagnostic["oversized_payload"] is False
    assert diagnostic["parse_or_postprocessing_timeout"] is False
    assert diagnostic["successful_completion"] is False

    diagnostic_path = executors._write_agy_stage_diagnostic(stage, diagnostic, base_dir=tmp_path)
    persisted = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert persisted["blocked_reason"] == "provider_tool_call_invalid_artifact_path"


def test_agy_printmode_timeout_after_auth_success_classification():
    import tools.task_engine_executors as executors

    log = """
E log.go] error getting token source: You are not logged into Antigravity.
I auth.go] ChainedAuth: authenticated via keyring (effective: keyring)
I server_oauth.go] OAuth: authenticated successfully as user@example.test
I model_resolver.go] Resolving model Gemini 3.5 Flash (High)
I model_config_manager.go] Propagating selected model override to backend: label="Gemini 3.5 Flash (High)"
I http_helpers.go] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse
E printmode.go] Print mode: timed out after 1195 polls (printed=121)
stdout: Error: timed out waiting for response
"""

    assert executors._agy_timeout_response(log) is True
    assert executors._agy_printmode_timeout_after_auth_success(log, GEMINI_HIGH) is True
    assert (
        executors._agy_timeout_blocker_reason(log, GEMINI_HIGH, attempt=1)
        == executors.AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS
    )


def test_agy_print_timeout_takes_priority_over_keychain_false_negative():
    import tools.task_engine_executors as executors

    log = """
You are not logged into Antigravity.
OAuth: authenticated successfully
stdout: Error: timed out waiting for response
E printmode.go] Print mode: timed out after 1195 polls
"""

    assert executors._agy_timeout_response(log) is True
    assert executors._agy_keychain_false_negative(log) is False
    assert executors._agy_timeout_blocker_reason(log, GEMINI_HIGH, attempt=1) == executors.AGY_TIMEOUT_BLOCKED


def test_agy_print_timeout_auth_uncertain_classification():
    import tools.task_engine_executors as executors

    log = """
You are not logged into Antigravity.
E printmode.go] Print mode: timed out after 1195 polls
stdout: Error: timed out waiting for response
"""

    assert executors._agy_printmode_timeout_auth_uncertain(log) is True
    assert (
        executors._agy_timeout_blocker_reason(log, GEMINI_HIGH, attempt=1)
        == executors.AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN
    )


def test_agy_subprocess_cwd_avoids_hidden_hermes(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.setenv("HERMES_AGY_CWD", "/Users/xqdwww/.hermes/hermes-agent")

    cwd = Path(executors._agy_subprocess_cwd())

    assert cwd.is_absolute()
    assert ".hermes" not in cwd.parts


def test_agy_subprocess_env_absolutizes_gemini_dir(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.setenv("GEMINI_DIR", ".gemini")

    env = executors._agy_subprocess_env()

    assert env["GEMINI_DIR"] == str(Path.home() / ".gemini")
    assert executors._agy_gemini_dir_is_absolute(env) is True


def test_agy_keychain_false_negative_classification():
    import tools.task_engine_executors as executors

    false_negative = "\n".join(
        [
            "You are not logged into Antigravity.",
            "OAuth: authenticated successfully",
        ]
    )
    truly_logged_out = "You are not logged into Antigravity."

    assert executors._classify_agy_preflight_block(false_negative, "") == "AGY_KEYCHAIN_TIMEOUT_FALSE_NEGATIVE"
    assert executors._classify_agy_preflight_block("", truly_logged_out) == "AGY_AUTH_REQUIRES_USER"


def test_agy_location_unsupported_classification():
    import tools.task_engine_executors as executors

    log = (
        "agent executor error: FAILED_PRECONDITION (code 400): "
        "User location is not supported for the API use."
    )

    assert executors._agy_location_unsupported(log) is True
    assert executors._classify_agy_preflight_block("", log) == executors.AGY_LOCATION_UNSUPPORTED


def test_agy_preflight_retries_keychain_false_negative_once(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "You are not logged into Antigravity.\nOAuth: authenticated successfully",
                },
            )()
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    result = run_agy_preflight()

    assert result["status"] == "AGY_OK"
    assert len(calls) == 2
    assert all(command == ["/opt/homebrew/bin/agy", "models"] for command in calls)
    assert all("auth" not in command and "login" not in command for command in calls)


def test_agy_preflight_true_logged_out_runs_bare_refresh_then_requires_manual_login(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "You are not logged into Antigravity.",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_reason"] == executors.REQUIRES_MANUAL_AGY_LOGIN
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]
    assert executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(calls[2])


def test_agy_preflight_models_empty_runs_bare_refresh_and_print_mode_passes(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 0, "stdout": "Antigravity CLI\nxqdwww@gmail.com\n", "stderr": ""})()
        Path(command[2]).write_text("Resolving model Gemini 3.1 Pro (High)\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": executors.AGY_INTERNAL_PREFLIGHT_SENTINEL, "stderr": ""})()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "AGY_OK"
    assert result["auth_refresh"]["status"] == "AGY_AUTH_REFRESH_OK"
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]
    assert executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(calls[2])


def test_agy_preflight_location_unsupported_refreshes_then_fails_closed(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []
    location_error = "FAILED_PRECONDITION (code 400): User location is not supported for the API use."

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": location_error})()
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 0, "stdout": "Antigravity CLI\nxqdwww@gmail.com\n", "stderr": ""})()
        Path(command[2]).write_text(location_error, encoding="utf-8")
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": location_error})()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_reason"] == executors.AGY_LOCATION_UNSUPPORTED
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]
    assert executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(calls[2])


def test_run_agy_gemini_retries_keychain_false_negative_then_succeeds(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        long_calls = [call for call in calls if call[-1] != "models"]
        log_file = Path(command[2])
        if len(long_calls) == 1:
            log_file.write_text("You are not logged into Antigravity.\nauthenticated via keyring\n", encoding="utf-8")
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": ""})()
        log_file.write_text("Resolving model Gemini 3.5 Flash (High)\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "fresh AGY output", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()

    assert executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH) == "fresh AGY output"
    assert len(calls) == 3
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert len([call for call in calls if call[-1] != "models"]) == 2
    assert all("auth" not in command and "login" not in command for command in calls)


def test_run_agy_gemini_retries_keychain_false_negative_only_once(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        Path(command[2]).write_text(
            "You are not logged into Antigravity.\nsilent auth succeeded\n",
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()
    try:
        executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("AGY retry failure should block")

    assert len(calls) == 3
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert "AGY_KEYCHAIN_TIMEOUT_FALSE_NEGATIVE" in message
    assert "token" not in message.lower()


def test_run_agy_gemini_retries_timeout_response_then_succeeds(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        long_calls = [call for call in calls if call[-1] != "models"]
        Path(command[2]).write_text("Resolving model Gemini 3.5 Flash (High)\n", encoding="utf-8")
        if len(long_calls) == 1:
            return type("Result", (), {"returncode": 0, "stdout": "Error: timed out waiting for response\n", "stderr": ""})()
        return type("Result", (), {"returncode": 0, "stdout": "fresh AGY output", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: sleeps.append(seconds))

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()

    assert executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH) == "fresh AGY output"
    assert len(calls) == 3
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert sleeps == [executors.AGY_KEYCHAIN_RETRY_SLEEP_S]
    assert all("auth" not in command and "login" not in command for command in calls)


def test_run_agy_gemini_blocks_after_repeated_timeout_response_without_valid_artifact(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        Path(command[2]).write_text("Resolving model Gemini 3.5 Flash (High)\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "Error: timed out waiting for response\n", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()

    try:
        executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("repeated AGY timeout response should block")

    assert len(calls) == 3
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert "L1_gemini_search: AGY_CALL_BLOCKED" in message
    assert "reason=AGY_TIMEOUT_BLOCKED" in message
    assert "actual_model='Gemini 3.5 Flash (High)'" in message
    assert "log_file='/private/tmp/agy-" in message
    assert "stdout: Error: timed out waiting for response" in message
    assert "token" not in message.lower()
    assert not list(tmp_path.rglob("*.md"))


def test_run_agy_gemini_timeout_expired_classifies_auth_uncertain_print_timeout(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 0, "stdout": "Antigravity CLI\nxqdwww@gmail.com\n", "stderr": ""})()
        Path(command[2]).write_text(
            "You are not logged into Antigravity.\nE printmode.go] Print mode: timed out after 1195 polls\n",
            encoding="utf-8",
        )
        raise executors.subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output="Error: timed out waiting for response\n",
            stderr="",
        )

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()

    try:
        executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("AGY print timeout should block")

    assert len(calls) >= 4
    assert ["--print-timeout", "600s"] == calls[1][-2:]
    assert calls[2] == ["/opt/homebrew/bin/agy"]
    assert any(executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(call) for call in calls)
    assert f"reason={executors.REQUIRES_MANUAL_AGY_LOGIN}" in message
    assert "AGY_KEYCHAIN_TIMEOUT_FALSE_NEGATIVE" not in message
    assert "token" not in message.lower()


def test_run_agy_gemini_runs_preflight_once_per_executor(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        Path(command[2]).write_text("Resolving model\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "fresh AGY output", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    executor = LocalTaskEngineExecutor()
    assert executor.run_agy_gemini(CANONICAL_STAGES[ENGINE_RESEARCH][0], "prompt 1", GEMINI_HIGH) == "fresh AGY output"
    assert executor.run_agy_gemini(CANONICAL_STAGES[ENGINE_RESEARCH][4], "prompt 2", GEMINI_PRO_HIGH) == "fresh AGY output"

    assert [call for call in calls if call[-1] == "models"] == [["/opt/homebrew/bin/agy", "models"]]
    assert len([call for call in calls if call[-1] != "models"]) == 2


def test_run_agy_gemini_blocks_before_long_prompt_when_preflight_fails(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 1, "stdout": "Open browser and enter authorization code ABC123", "stderr": ""})()
        if executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(command):
            Path(command[2]).write_text("not authenticated\n", encoding="utf-8")
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "You are not logged into Antigravity."})()
        assert command[-1] == "models"
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "You are not logged into Antigravity."})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    executor = LocalTaskEngineExecutor()
    try:
        executor.run_agy_gemini(CANONICAL_STAGES[ENGINE_RESEARCH][0], "long prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("preflight failure should block before AGY long prompt")

    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]
    assert "L1_gemini_search: AGY_PREFLIGHT_BLOCKED" in message
    assert executors.REQUIRES_MANUAL_AGY_LOGIN in message


def test_run_agy_gemini_empty_print_response_refreshes_and_retries(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 0, "stdout": "Antigravity CLI\nxqdwww@gmail.com\n", "stderr": ""})()
        Path(command[2]).write_text("Resolving model Gemini 3.5 Flash (High)\n", encoding="utf-8")
        if executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(command):
            return type("Result", (), {"returncode": 0, "stdout": executors.AGY_INTERNAL_PREFLIGHT_SENTINEL, "stderr": ""})()
        original_print_calls = [
            call for call in calls
            if call[-1] != "models" and call != ["/opt/homebrew/bin/agy"] and executors.AGY_INTERNAL_PREFLIGHT_SENTINEL not in " ".join(call)
        ]
        if len(original_print_calls) == 1:
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("Result", (), {"returncode": 0, "stdout": "fresh AGY output", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()

    assert executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH) == "fresh AGY output"
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert ["/opt/homebrew/bin/agy"] in calls
    assert any(executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(call) for call in calls)


def test_run_agy_gemini_empty_sentinel_after_refresh_fails_closed(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "models":
            return type("Result", (), {"returncode": 0, "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\n", "stderr": ""})()
        if command == ["/opt/homebrew/bin/agy"]:
            return type("Result", (), {"returncode": 0, "stdout": "Antigravity CLI\nxqdwww@gmail.com\n", "stderr": ""})()
        Path(command[2]).write_text("Resolving model Gemini 3.5 Flash (High)\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("HERMES_AGY_MODEL_ALIAS_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][0]
    executor = LocalTaskEngineExecutor()
    try:
        executor.run_agy_gemini(stage, "prompt", GEMINI_HIGH)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty sentinel after refresh should block")

    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert ["/opt/homebrew/bin/agy"] in calls
    assert any(executors.AGY_INTERNAL_PREFLIGHT_SENTINEL in " ".join(call) for call in calls)
    assert f"reason={executors.AGY_PRINT_MODE_EMPTY_RESPONSE}" in message


def test_l2_5_codex_handoff_smoke_writes_protocol_files(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {"source_candidates": [{"title": "fake"}]}

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

    l1_l2 = run_research_l1_l2_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    assert l1_l2["status"] == "ok"

    result = run_research_l2_5_codex_handoff_smoke(
        l1_l2["run"],
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
    ]
    handoff_dir = tmp_path / "L2_5_codex_evidence_organizer"
    for name in (
        "source_candidates.json",
        "ddgs_gap_sources.json",
        "evidence_runner_001.request.md",
        "evidence_runner_001.request.json",
        "sources.csv",
        "evidence.csv",
        "claims.md",
        "gaps.md",
    ):
        assert (handoff_dir / name).exists()
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:L3_r1_synthesis" in error for error in result["full_pipeline_validation"]["errors"])


def test_l1_l3_smoke_writes_r1_synthesis_and_stops_before_l4(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {"source_candidates": [{"title": "fake"}]}

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            assert stage.stage_name == "L3_r1_synthesis"
            assert stage.owner == R1_32B
            assert model == R1_32B
            assert "L1_gemini_search" in prompt
            assert "L2_ddgs_supplement" in prompt
            assert "L2_5_codex_evidence_organizer" in prompt
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l3_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
    ]
    l3 = stages[-1]
    assert l3["owner"] == R1_32B
    assert l3["model"] == R1_32B
    assert l3["executor_model"] == R1_ACTUAL_MODEL_DEFAULT
    assert l3["artifact_path"].endswith("L3_r1_synthesis/r1_synthesis.md")
    assert l3["created_in_current_run"] is True
    assert l3["legacy_contaminated"] is False
    assert l3["valid_for_pipeline"] is True
    assert Path(l3["artifact_path"]).read_text(encoding="utf-8") == "R1 synthesis body"
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:L4_gemini_audit" in error for error in result["full_pipeline_validation"]["errors"])


def test_l3_requires_fresh_l1_l2_l2_5_artifacts(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {"source_candidates": [{"title": "fake"}]}

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            return "should not run"

    l1_l2 = run_research_l1_l2_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    l2_5 = run_research_l2_5_codex_handoff_smoke(l1_l2["run"], base_dir=tmp_path, executor=FakeExecutor())
    l2_5["run"]["stages"][0]["created_in_current_run"] = False

    result = run_research_l3_synthesis_smoke(l2_5["run"], base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "L3_r1_synthesis"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "not created_in_current_run" in result["run"]["stages"][-1]["error"]


def test_l3_rejects_legacy_artifacts(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {"source_candidates": [{"title": "fake"}]}

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            return "should not run"

    l1_l2 = run_research_l1_l2_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    l2_5 = run_research_l2_5_codex_handoff_smoke(l1_l2["run"], base_dir=tmp_path, executor=FakeExecutor())
    l2_5["run"]["stages"][1]["legacy_contaminated"] = True

    result = run_research_l3_synthesis_smoke(l2_5["run"], base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "L3_r1_synthesis"
    assert "legacy contaminated" in result["run"]["stages"][-1]["error"]


def test_l3_r1_model_mismatch_is_blocked_before_omlx_call():
    stage = CANONICAL_STAGES[ENGINE_RESEARCH][3]
    try:
        LocalTaskEngineExecutor().run_omlx_model(stage, "Qwen72B", "prompt")
    except RuntimeError as exc:
        assert "R1 model binding mismatch" in str(exc)
    else:
        raise AssertionError("L3 must reject non-R1 model bindings")


def test_l3_omlx_uses_actual_r1_model_and_legacy_admin_path(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        return {"choices": [{"message": {"content": "R1 actual output"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][3]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, R1_32B, "prompt")

    assert output == "R1 actual output"
    assert resolve_r1_omlx_model_alias(R1_32B) == R1_ACTUAL_MODEL_DEFAULT
    assert ("load_model", R1_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", R1_32B) not in calls
    assert any(call[0] == "chat" and call[1] == R1_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", R1_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["L3_r1_synthesis"] == R1_ACTUAL_MODEL_DEFAULT


def test_convergence_report_omlx_uses_actual_r1_model_and_legacy_admin_path(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        return {"choices": [{"message": {"content": "divergence_role_summary\nconvergence_decision_framework"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][13]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, R1_32B, "prompt")

    assert "convergence_decision_framework" in output
    assert resolve_r1_omlx_model_alias(R1_32B) == R1_ACTUAL_MODEL_DEFAULT
    assert ("load_model", R1_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", R1_32B) not in calls
    assert any(call[0] == "chat" and call[1] == R1_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", R1_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["convergence_report"] == R1_ACTUAL_MODEL_DEFAULT


def test_l3_omlx_auth_missing_blocks_without_key_leak(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")
    stage = CANONICAL_STAGES[ENGINE_RESEARCH][3]

    try:
        LocalTaskEngineExecutor().run_omlx_model(stage, R1_32B, "prompt")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing OMLX auth should block L3")

    assert "OMLX_AUTH_BLOCKED" in message
    assert "omlx-" not in message


def test_structure_mapper_omlx_uses_actual_qwen72b_model_and_legacy_admin_path(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        return {"choices": [{"message": {"content": "problem_axes\nactor_map\ndecision_questions"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_QWEN72B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][8]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, QWEN72B, "prompt")

    assert "problem_axes" in output
    assert resolve_qwen72b_omlx_model_alias(QWEN72B) == QWEN72B_ACTUAL_MODEL_DEFAULT
    assert ("load_model", QWEN72B_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", QWEN72B) not in calls
    assert any(call[0] == "chat" and call[1] == QWEN72B_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", QWEN72B_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["structure_mapper"] == QWEN72B_ACTUAL_MODEL_DEFAULT


def test_structure_mapper_rejects_forbidden_actual_model_alias(monkeypatch):
    monkeypatch.setenv("HERMES_OMLX_QWEN72B_MODEL", "Qwen2.5-9B-Instruct-mlx")

    try:
        resolve_qwen72b_omlx_model_alias(QWEN72B)
    except RuntimeError as exc:
        assert "forbidden Qwen72B actual model alias" in str(exc)
    else:
        raise AssertionError("structure_mapper must reject 9B actual model aliases")

    monkeypatch.setenv("HERMES_OMLX_QWEN72B_MODEL", R1_ACTUAL_MODEL_DEFAULT)
    try:
        resolve_qwen72b_omlx_model_alias(QWEN72B)
    except RuntimeError as exc:
        assert "forbidden Qwen72B actual model alias" in str(exc)
    else:
        raise AssertionError("structure_mapper must reject R1 actual model aliases")


def test_evidence_judge_omlx_uses_actual_nemotron_and_retries_incomplete_read(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        if len([call for call in calls if call[0] == "chat"]) == 1:
            raise http.client.IncompleteRead(b"partial")
        return {"choices": [{"message": {"content": "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_NEMOTRON120B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, NEMOTRON120B, "prompt")

    assert "evidence_quality_map" in output
    assert resolve_nemotron120b_omlx_model_alias(NEMOTRON120B) == NEMOTRON120B_ACTUAL_MODEL_DEFAULT
    assert ("load_model", NEMOTRON120B_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", NEMOTRON120B) not in calls
    assert len([call for call in calls if call[0] == "chat"]) == 2
    assert any(call[0] == "chat" and call[1] == NEMOTRON120B_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", NEMOTRON120B_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["evidence_judge"] == NEMOTRON120B_ACTUAL_MODEL_DEFAULT


def test_evidence_judge_rejects_forbidden_actual_model_alias(monkeypatch):
    for bad in ["Qwen2.5-72B-Instruct-mlx", "Qwen2.5-9B-Instruct-mlx", R1_ACTUAL_MODEL_DEFAULT, "DeepSeek Controller"]:
        monkeypatch.setenv("HERMES_OMLX_NEMOTRON120B_MODEL", bad)
        try:
            resolve_nemotron120b_omlx_model_alias(NEMOTRON120B)
        except RuntimeError as exc:
            assert "forbidden Nemotron-120B actual model alias" in str(exc)
        else:
            raise AssertionError(f"evidence_judge must reject forbidden actual model alias: {bad}")


def test_evidence_judge_omlx_retries_empty_content_with_reload(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        if len([call for call in calls if call[0] == "chat"]) == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return {"choices": [{"message": {"content": "evidence_quality_map\nstrength_by_claim"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_NEMOTRON120B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, NEMOTRON120B, "prompt")

    assert "evidence_quality_map" in output
    assert len([call for call in calls if call[0] == "chat"]) == 2
    assert all(call[-1] == 1536 for call in calls if call[0] == "chat")
    assert len([call for call in calls if call[0] == "load_model"]) == 2
    assert calls.count(("unload_all",)) == 2
    assert executor.last_omlx_diagnostics["evidence_judge"]["attempt"] == "first"


def test_omlx_empty_content_diagnostic_classifies_error_object():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    data = {
        "error": {
            "message": "oMLX prefill memory guard rejected this prompt",
            "code": "prefill_memory_exceeded",
        },
        "type": "server_error",
    }

    diagnostic = executors._omlx_empty_content_diagnostic(
        stage,
        NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
        data,
        attempt="final",
    )

    assert diagnostic["empty_content"] is True
    assert diagnostic["empty_content_kind"] == "response_error_object"
    assert diagnostic["choices_type"] == "missing"
    assert diagnostic["choices_len"] == 0
    assert diagnostic["error_type"] == "server_error"
    assert "prefill_memory_exceeded" in diagnostic["error_summary"]
    assert diagnostic["blocked_reason"] == "OMLX_PREFILL_MEMORY_GUARD_BLOCKED"
    assert diagnostic["raw_error_code"] == "prefill_memory_exceeded"


def test_evidence_judge_default_token_budget_can_fit_required_sections(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.delenv("HERMES_OMLX_EVIDENCE_JUDGE_MAX_TOKENS", raising=False)
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]

    assert executors._omlx_max_tokens_for_stage(stage) == 1536


def test_evidence_judge_prefill_memory_guard_diagnostic_records_request_context(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            self.loaded = {"Qwen2.5-72B-Instruct-abliterated-mlx-4Bit"}

        def login(self):
            return True

        def get_models(self):
            ids = [
                "Qwen2.5-72B-Instruct-abliterated-mlx-4Bit",
                NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
            ]
            return [{"id": model_id, "loaded": model_id in self.loaded} for model_id in ids]

        def unload_all(self):
            calls.append(("unload_all", sorted(self.loaded)))
            self.loaded.clear()

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            self.loaded.add(model_id)
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            self.loaded.discard(model_id)
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, len(messages), max_tokens))
        raise RuntimeError(
            'OMLX chat HTTP 400: {"error":{"message":"oMLX prefill memory guard rejected this prompt",'
            '"code":"prefill_memory_exceeded","omlx_code":"prefill_memory_exceeded"}}'
        )

    monkeypatch.setenv("OMLX_API_KEY", "secret-test-key")
    monkeypatch.delenv("HERMES_OMLX_EVIDENCE_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    executor = LocalTaskEngineExecutor()
    prompt = "judge evidence without storing full prompt secret-test-key"

    try:
        executor.run_omlx_model(stage, NEMOTRON120B, prompt)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("prefill memory guard must block evidence_judge")

    diagnostic = executor.last_omlx_diagnostics["evidence_judge"]
    diagnostic_text = json.dumps(diagnostic, ensure_ascii=False, default=str)

    assert "OMLX_PREFILL_MEMORY_GUARD_BLOCKED" in message
    assert diagnostic["blocked_reason"] == "OMLX_PREFILL_MEMORY_GUARD_BLOCKED"
    assert diagnostic["prompt_chars"] == len(prompt)
    assert diagnostic["prompt_estimated_tokens"] >= 1
    assert diagnostic["message_count"] == 1
    assert diagnostic["system_message_chars"] == 0
    assert diagnostic["user_message_chars"] == len(prompt)
    assert diagnostic["max_tokens"] == 1536
    assert diagnostic["temperature"] == 0
    assert diagnostic["stream"] is False
    assert diagnostic["actual_model"] == NEMOTRON120B_ACTUAL_MODEL_DEFAULT
    assert diagnostic["endpoint"].endswith("/v1/chat/completions")
    assert diagnostic["loaded_models_before_unload"] == ["Qwen2.5-72B-Instruct-abliterated-mlx-4Bit"]
    assert diagnostic["loaded_models_after_unload"] == []
    assert diagnostic["loaded_models_after_load"] == [NEMOTRON120B_ACTUAL_MODEL_DEFAULT]
    assert diagnostic["compact_mode_used"] is False
    assert diagnostic["compact_budget"] is None
    assert diagnostic["retry_attempt"] == "final"
    assert diagnostic["raw_error_code"] == "prefill_memory_exceeded"
    assert "raw_error_summary" in diagnostic
    assert "prompt_hash" in diagnostic
    assert prompt not in diagnostic_text
    assert "secret-test-key" not in diagnostic_text
    assert any(call == ("chat", NEMOTRON120B_ACTUAL_MODEL_DEFAULT, 1, 1536) for call in calls)


def test_omlx_empty_content_diagnostic_keeps_non_prefill_classification():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    diagnostic = executors._omlx_empty_content_diagnostic(
        stage,
        NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
        {"choices": [{"message": {"content": ""}}]},
        attempt="final",
        request_context={
            "prompt_chars": 12,
            "max_tokens": 512,
            "loaded_models_before_unload": [],
            "loaded_models_after_unload": [],
            "loaded_models_after_load": [NEMOTRON120B_ACTUAL_MODEL_DEFAULT],
            "compact_mode_used": False,
        },
    )

    assert diagnostic["blocked_reason"] == "OMLX_EMPTY_CONTENT_BLOCKED"
    assert diagnostic["empty_content_kind"] == "empty_content_string"
    assert diagnostic["raw_error_code"] == ""
    assert diagnostic["prompt_chars"] == 12


def test_omlx_idle_status_is_ready_for_loaded_checks():
    import tools.task_engine_executors as executors

    class FakeAdmin:
        def get_models(self):
            return [{"id": NEMOTRON120B_ACTUAL_MODEL_DEFAULT, "status": "idle"}]

    admin = executors._OmlxAdmin.__new__(executors._OmlxAdmin)
    admin.get_models = FakeAdmin().get_models

    assert admin.is_model_loaded(NEMOTRON120B_ACTUAL_MODEL_DEFAULT) is True
    assert executors._loaded_omlx_model_ids(admin) == [NEMOTRON120B_ACTUAL_MODEL_DEFAULT]
    assert executors._omlx_observed_model_status(admin, NEMOTRON120B_ACTUAL_MODEL_DEFAULT) == "idle"


def test_decision_omlx_stage_timeout_writes_idle_inference_not_sent_diagnostic(monkeypatch, tmp_path: Path):
    import time
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][3]
    executor = LocalTaskEngineExecutor()
    executor.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
    executor.last_omlx_diagnostics[stage.stage_name] = {
        "stage_name": stage.stage_name,
        "model": stage.model,
        "actual_model": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
        "call_site": "test_call_site",
        "admin_load_requested": True,
        "admin_load_returned": False,
        "observed_model_status": "idle",
        "inference_request_sent": False,
        "inference_response_received": False,
    }
    monkeypatch.setenv("HERMES_DECISION_EVIDENCE_JUDGE_TIMEOUT_S", "1")

    try:
        executors._run_decision_stage_with_timeout(
            stage,
            base_dir=tmp_path / "sample_01" / "decision_run",
            executor=executor,
            operation=lambda: time.sleep(2),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("stage timeout should block")

    diagnostic_path = tmp_path / "sample_01" / "decision_run" / "evidence_judge" / "evidence_judge.diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert "inference_not_sent" in message
    assert diagnostic["sample_id"] == "sample_01"
    assert diagnostic["stage_name"] == "evidence_judge"
    assert diagnostic["model"] == stage.model
    assert diagnostic["timeout_seconds"] == 1
    assert diagnostic["error_type"] == "stage_timeout"
    assert diagnostic["call_site"] == "test_call_site"
    assert diagnostic["admin_load_requested"] is True
    assert diagnostic["admin_load_returned"] is False
    assert diagnostic["observed_model_status"] == "idle"
    assert diagnostic["inference_request_sent"] is False
    assert diagnostic["inference_response_received"] is False
    assert diagnostic["blocked_reason"] == "inference_not_sent"
    assert diagnostic["blocked_reason"] != "model_load_failed"


def test_omlx_response_read_interruption_saves_partial_content(tmp_path: Path):
    import tools.task_engine_executors as executors

    class PartialResponse:
        def __init__(self):
            self.calls = 0

        def read(self, _size):
            self.calls += 1
            if self.calls == 1:
                return b'{"choices":[{"message":{"content":"partial evidence'
            raise KeyboardInterrupt()

    partial_path = tmp_path / "evidence_judge" / "evidence_judge.partial.md"

    try:
        executors._read_omlx_response_bytes(PartialResponse(), partial_content_path=partial_path, started=0)
    except executors._OmlxPartialResponseError as exc:
        error = exc
    else:
        raise AssertionError("partial response read should raise diagnostic error")

    assert error.blocked_reason == "response_read_interrupted"
    assert error.partial_content_chars > 0
    assert error.partial_content_path == str(partial_path)
    assert partial_path.exists()
    assert "partial evidence" in partial_path.read_text(encoding="utf-8")


def test_evidence_judge_empty_content_blocks_and_writes_diagnostic(tmp_path: Path):
    class EmptyEvidenceExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                self.last_omlx_diagnostics[stage.stage_name] = {
                    "stage_name": stage.stage_name,
                    "actual_model": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                    "attempt": "final",
                    "choices_len": 1,
                    "content_length": 0,
                    "empty_content": True,
                }
                raise RuntimeError("evidence_judge: OMLX_EMPTY_CONTENT_BLOCKED: Nemotron-120B returned empty content")
            raise AssertionError("Only evidence_judge should run in this targeted smoke")

    base = tmp_path
    stage_names = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
    ]
    prior_stages = []
    specs = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION]
    for idx, name in enumerate(stage_names):
        spec = specs[idx]
        stage_dir = base / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        if name == "L2_5_codex_evidence_organizer":
            artifact_path = stage_dir
            outputs = {}
            for required in spec.required_outputs:
                path = stage_dir / required
                path.write_text("fresh", encoding="utf-8")
                outputs[required] = str(path)
        else:
            output_name = spec.required_outputs[0] if spec.required_outputs != ("artifact_path",) else "report.md"
            artifact_path = stage_dir / output_name
            if name == "L5_deepseek_acceptance":
                artifact_path.write_text(_complete_research_evidence_packet_text(), encoding="utf-8")
            else:
                artifact_path.write_text("fresh", encoding="utf-8")
            outputs = {output_name: str(artifact_path)} if spec.required_outputs != ("artifact_path",) else {}
        record = {
            "stage_name": name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact_path),
            "outputs": outputs,
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
        }
        if name == "L5_deepseek_acceptance":
            record["status"] = "accepted"
        prior_stages.append(record)

    result = run_research_decision_evidence_judge_smoke(
        {"mode": ENGINE_RESEARCH_DECISION, "stages": prior_stages},
        query=ADHD_PROMPT,
        base_dir=base,
        executor=EmptyEvidenceExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    error = result["run"]["stages"][-1]["error"]
    assert "OMLX_EMPTY_CONTENT_BLOCKED" in error
    diagnostic = base / "evidence_judge" / "evidence_judge.diagnostic.json"
    assert diagnostic.exists()
    assert "diagnostic_artifact=" in error
    data = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert data["empty_content"] is True
    assert data["actual_model"] == NEMOTRON120B_ACTUAL_MODEL_DEFAULT


def test_premise_auditor_omlx_uses_actual_llama70b_and_retries_incomplete_read(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        if len([call for call in calls if call[0] == "chat"]) == 1:
            raise http.client.IncompleteRead(b"partial")
        return {"choices": [{"message": {"content": "implicit_premises\npremise_risks\ncounterexamples"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_LLAMA70B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)
    monkeypatch.setattr(executors.time, "sleep", lambda seconds: None)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][10]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, LLAMA70B, "prompt")

    assert "implicit_premises" in output
    assert resolve_llama70b_omlx_model_alias(LLAMA70B) == LLAMA70B_ACTUAL_MODEL_DEFAULT
    assert ("load_model", LLAMA70B_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", LLAMA70B) not in calls
    assert len([call for call in calls if call[0] == "chat"]) == 2
    assert any(call[0] == "chat" and call[1] == LLAMA70B_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", LLAMA70B_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["premise_auditor"] == LLAMA70B_ACTUAL_MODEL_DEFAULT


def test_premise_auditor_rejects_forbidden_actual_model_alias(monkeypatch):
    for bad in [
        QWEN72B_ACTUAL_MODEL_DEFAULT,
        NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
        "Qwen2.5-9B-Instruct-mlx",
        R1_ACTUAL_MODEL_DEFAULT,
        "DeepSeek Controller",
    ]:
        monkeypatch.setenv("HERMES_OMLX_LLAMA70B_MODEL", bad)
        try:
            resolve_llama70b_omlx_model_alias(LLAMA70B)
        except RuntimeError as exc:
            assert "forbidden Llama70B actual model alias" in str(exc)
        else:
            raise AssertionError(f"premise_auditor must reject forbidden actual model alias: {bad}")


def test_alternative_generator_omlx_uses_actual_gemma431b_model(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        return {"choices": [{"message": {"content": "mutually_exclusive_alternatives\nintervention_intensity_paths"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_GEMMA431B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][11]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, GEMMA431B, "prompt")

    assert "mutually_exclusive_alternatives" in output
    assert resolve_gemma431b_omlx_model_alias(GEMMA431B) == GEMMA431B_ACTUAL_MODEL_DEFAULT
    assert ("load_model", GEMMA431B_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", GEMMA431B) not in calls
    assert any(call[0] == "chat" and call[1] == GEMMA431B_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_all",) in calls
    assert ("unload_model", GEMMA431B_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["alternative_generator"] == GEMMA431B_ACTUAL_MODEL_DEFAULT


def test_alternative_generator_rejects_forbidden_actual_model_alias(monkeypatch):
    for bad in [
        LLAMA70B_ACTUAL_MODEL_DEFAULT,
        QWEN72B_ACTUAL_MODEL_DEFAULT,
        NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
        "Qwen2.5-9B-Instruct-mlx",
        R1_ACTUAL_MODEL_DEFAULT,
        "DeepSeek Controller",
    ]:
        monkeypatch.setenv("HERMES_OMLX_GEMMA431B_MODEL", bad)
        try:
            resolve_gemma431b_omlx_model_alias(GEMMA431B)
        except RuntimeError as exc:
            assert "forbidden Gemma-4-31B actual model alias" in str(exc)
        else:
            raise AssertionError(f"alternative_generator must reject forbidden actual model alias: {bad}")


def test_insight_harvester_omlx_uses_same_actual_gemma431b_model(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            calls.append(("admin_init", base_url, bool(api_key)))

        def login(self):
            calls.append(("login",))
            return True

        def unload_all(self):
            calls.append(("unload_all",))

        def load_model(self, model_id):
            calls.append(("load_model", model_id))
            return {"success": True}

        def unload_model(self, model_id):
            calls.append(("unload_model", model_id))
            return {"success": True}

    def fake_chat(model, messages, *, api_key, timeout, max_tokens, chat_template_kwargs=None):
        calls.append(("chat", model, bool(api_key), timeout, max_tokens))
        return {"choices": [{"message": {"content": "cross_model_insights\nconflicts_and_tensions"}}]}

    monkeypatch.setenv("OMLX_API_KEY", "omlx-test-key")
    monkeypatch.delenv("HERMES_OMLX_GEMMA431B_MODEL", raising=False)
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    monkeypatch.setattr(executors, "_omlx_chat_completion", fake_chat)

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][12]
    executor = LocalTaskEngineExecutor()
    output = executor.run_omlx_model(stage, GEMMA431B, "prompt")

    assert "cross_model_insights" in output
    assert resolve_gemma431b_omlx_model_alias(GEMMA431B) == GEMMA431B_ACTUAL_MODEL_DEFAULT
    assert ("load_model", GEMMA431B_ACTUAL_MODEL_DEFAULT) in calls
    assert ("load_model", GEMMA431B) not in calls
    assert any(call[0] == "chat" and call[1] == GEMMA431B_ACTUAL_MODEL_DEFAULT for call in calls)
    assert ("unload_model", GEMMA431B_ACTUAL_MODEL_DEFAULT) in calls
    assert executor.last_executor_models["insight_harvester"] == GEMMA431B_ACTUAL_MODEL_DEFAULT


def test_l3_artifact_missing_blocks_validation(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return {"source_candidates": [{"title": "fake"}]}

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l3_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    Path(result["run"]["stages"][-1]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH, result["run"], base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("L3_r1_synthesis:artifact_path_not_found" in error for error in validation["errors"])


def test_l1_l4_smoke_writes_gemini_audit_and_stops_before_l5(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                assert model == GEMINI_HIGH
                return {"source_candidates": [{"title": "fake"}]}
            assert stage.stage_name == "L4_gemini_audit"
            assert stage.owner == GEMINI_PRO_HIGH
            assert stage.model == GEMINI_PRO_HIGH
            assert model == GEMINI_PRO_HIGH
            assert model != GEMINI_HIGH
            assert "L3_r1_synthesis" in prompt
            assert "Gemini 3.1 Pro (High)" in prompt
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "Gemini audit body"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l4_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
    ]
    l4 = stages[-1]
    assert l4["owner"] == GEMINI_PRO_HIGH
    assert l4["model"] == GEMINI_PRO_HIGH
    assert l4["executor_model"] == GEMINI_PRO_HIGH
    assert l4["executor_model"] != GEMINI_HIGH
    assert l4["artifact_path"].endswith("L4_gemini_audit/gemini_audit_report.md")
    assert l4["created_in_current_run"] is True
    assert l4["legacy_contaminated"] is False
    assert l4["valid_for_pipeline"] is True
    assert Path(l4["artifact_path"]).read_text(encoding="utf-8") == "Gemini audit body"
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:L5_deepseek_acceptance" in error for error in result["full_pipeline_validation"]["errors"])


def test_l4_cannot_run_when_l3_missing(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return "should not run"

    result = run_research_l4_gemini_audit_smoke({"stages": []}, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "L4_gemini_audit"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1/L2/L2_5/L3 stages" in result["run"]["stages"][-1]["error"]


def test_l4_rejects_legacy_l3_artifact(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {
                    "source_candidates": [
                        {
                            "candidate": "CDC ADHD parent training behavior therapy guidance",
                            "evidence_type": "Guideline",
                            "coverage_axis": "authoritative_guideline_or_consensus",
                            "why_relevant": "ADHD parent training and behavior therapy guidance is relevant to child intervention decisions.",
                        },
                        {
                            "candidate": "AAP ADHD clinical practice guideline school supports",
                            "evidence_type": "Guideline",
                            "coverage_axis": "intervention_or_practice",
                            "why_relevant": "ADHD school supports and clinical guidance inform accommodations and intervention intensity.",
                        },
                        {
                            "candidate": "CLAS ADHD inattentive children parent training organization skills",
                            "evidence_type": "Intervention Study",
                            "coverage_axis": "empirical_study_or_RCT",
                            "why_relevant": "ADHD inattentive children may benefit from parent training and organization skills interventions.",
                        },
                        {
                            "candidate": "ADHD executive function children mind wandering cognitive disengagement",
                            "evidence_type": "Mechanism Review",
                            "coverage_axis": "mechanism_or_theory",
                            "why_relevant": "ADHD executive function and mind wandering mechanisms shape long-term support needs.",
                        },
                    ]
                }
            return "should not run"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    l1_l3 = run_research_l1_l3_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    l1_l3["run"]["stages"][3]["legacy_contaminated"] = True
    result = run_research_l4_gemini_audit_smoke(l1_l3["run"], base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "L4_gemini_audit"
    assert "legacy contaminated" in result["run"]["stages"][-1]["error"]


def test_l4_artifact_missing_blocks_validation(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {
                    "source_candidates": [
                        {
                            "candidate": "https://guideline.example.test/adhd-parent-training",
                            "why_relevant": "ADHD parent training evidence covers active intervention degree and monitoring outcomes.",
                            "coverage_axis": "intervention evidence",
                        },
                        {
                            "candidate": "https://review.example.test/adhd-treatment-review",
                            "why_relevant": "ADHD treatment review evidence discusses behavioral support, medication boundaries, and uncertainty.",
                            "coverage_axis": "comparison evidence",
                        },
                        {
                            "candidate": "https://outcomes.example.test/adhd-long-term",
                            "why_relevant": "ADHD long-term outcome evidence covers follow-up, functional impact, and evidence gaps.",
                            "coverage_axis": "outcome evidence",
                        },
                    ]
                }
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "Gemini audit body"

        def run_ddgs(self, stage, queries):
            return [
                {
                    "query": queries[0],
                    "title": "ADHD intervention evidence review",
                    "url": "https://example.test/ddgs-review",
                    "snippet": "ADHD children intervention evidence reports parent training outcomes, monitoring limits, and active treatment degree.",
                },
                {
                    "query": queries[0],
                    "title": "ADHD treatment outcomes and uncertainty",
                    "url": "https://example.test/ddgs-outcomes",
                    "snippet": "ADHD treatment outcome evidence discusses behavioral intervention, medication boundaries, and uncertainty.",
                },
                {
                    "query": queries[0],
                    "title": "ADHD family intervention follow-up",
                    "url": "https://example.test/ddgs-family",
                    "snippet": "ADHD family intervention evidence covers parent training, school coordination, follow-up, and evidence gaps.",
                },
            ]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l4_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    Path(result["run"]["stages"][-1]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH, result["run"], base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("L4_gemini_audit:artifact_path_not_found" in error for error in validation["errors"])


def test_l5_cannot_run_when_l4_missing(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_controller_acceptance(self, stage, packet):
            return "should not run"

    result = run_research_l5_acceptance_smoke({"stages": []}, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "L5_deepseek_acceptance"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1/L2/L2_5/L3/L4 stages" in result["run"]["stages"][-1]["error"]


def test_l1_l5_smoke_accepts_research_packet_and_stops_before_decision(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {
                    "source_candidates": [
                        {
                            "candidate": "https://parent-training.example.test/adhd-parent-training",
                            "why_relevant": "ADHD intervention evidence for parent training covers population, treatment intensity, and outcome tracking.",
                            "coverage_axis": "intervention evidence",
                        },
                        {
                            "candidate": "https://guideline.example.test/adhd-medication-guideline",
                            "why_relevant": "ADHD treatment guideline evidence compares behavioral intervention, medication, and monitoring outcomes.",
                            "coverage_axis": "comparison evidence",
                        },
                    ]
                }
            assert stage.stage_name == "L4_gemini_audit"
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "Gemini audit body"

        def run_ddgs(self, stage, queries):
            return [
                {
                    "query": queries[0],
                    "title": "ADHD intervention evidence review",
                    "url": "https://review.example.test/ddgs-review",
                    "snippet": "ADHD children intervention evidence reports parent training outcomes and decision-relevant limits.",
                },
                {
                    "query": queries[0],
                    "title": "ADHD long term treatment outcomes",
                    "url": "https://outcomes.example.test/ddgs-outcomes",
                    "snippet": "ADHD treatment outcome evidence discusses active intervention degree, monitoring, and uncertainty.",
                },
            ]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == PIPELINE_COMPLETE
    assert result["full_pipeline_validation"]["valid"] is True
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
    ]
    l5 = stages[-1]
    assert l5["owner"] == CONTROLLER_ACCEPTANCE
    assert l5["model"] == CONTROLLER_ACCEPTANCE
    assert l5["executor_model"] == CONTROLLER_ACCEPTANCE
    assert l5["artifact_path"].endswith("L5_deepseek_acceptance/research_evidence_packet.md")
    assert l5["created_in_current_run"] is True
    assert l5["legacy_contaminated"] is False
    assert l5["valid_for_pipeline"] is True
    packet = Path(l5["artifact_path"]).read_text(encoding="utf-8")
    assert "verdict: ACCEPTED_WITH_DEFECTS" in packet
    assert "accepted: true" in packet
    assert "L4_gemini_audit" in packet
    assert "evidence_packet_ready_for_decision: conditional" in packet
    assert "requires_full_text_verification" in packet
    assert "final_controller_report" not in packet
    assert "final report" not in packet.lower()
    assert "最终建议" not in packet


def test_l5_accepted_research_decision_still_incomplete(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {
                    "source_candidates": [
                        {
                            "candidate": "https://guideline.example.test/adhd-parent-training",
                            "why_relevant": "ADHD parent training evidence covers active intervention degree and monitoring outcomes.",
                            "coverage_axis": "intervention evidence",
                        },
                        {
                            "candidate": "https://review.example.test/adhd-treatment-review",
                            "why_relevant": "ADHD treatment review evidence discusses behavioral support, medication boundaries, and uncertainty.",
                            "coverage_axis": "comparison evidence",
                        },
                        {
                            "candidate": "https://outcomes.example.test/adhd-long-term",
                            "why_relevant": "ADHD long-term outcome evidence covers follow-up, functional impact, and evidence gaps.",
                            "coverage_axis": "outcome evidence",
                        },
                    ]
                }
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return _complete_research_evidence_packet_text()

        def run_ddgs(self, stage, queries):
            return [
                {
                    "query": queries[0],
                    "title": "ADHD intervention evidence review",
                    "url": "https://example.test/ddgs-review",
                    "snippet": "ADHD children intervention evidence reports parent training outcomes, monitoring limits, and active treatment degree.",
                },
                {
                    "query": queries[0],
                    "title": "ADHD treatment outcomes and uncertainty",
                    "url": "https://example.test/ddgs-outcomes",
                    "snippet": "ADHD treatment outcome evidence discusses behavioral intervention, medication boundaries, and uncertainty.",
                },
                {
                    "query": queries[0],
                    "title": "ADHD family intervention follow-up",
                    "url": "https://example.test/ddgs-family",
                    "snippet": "ADHD family intervention evidence covers parent training, school coordination, follow-up, and evidence gaps.",
                },
            ]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    validation = validate_pipeline(ENGINE_RESEARCH_DECISION, result["run"], base_dir=tmp_path)

    assert result["pipeline_status"] == PIPELINE_COMPLETE
    assert validation["valid"] is False
    assert validation["pipeline_status"] == PIPELINE_BLOCKED
    assert any("missing_stage:intelligence_layer" in error for error in validation["errors"])


def test_l5_rejected_does_not_complete(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "verdict: REJECTED\naccepted: false\ninsufficient evidence"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    l1_l4 = run_research_l1_l4_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_l5_acceptance_smoke(l1_l4["run"], base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["pipeline_status"] == PIPELINE_BLOCKED
    assert result["blocked_stage"] == "L5_deepseek_acceptance"
    l5 = result["run"]["stages"][-1]
    assert l5["status"] == "rejected"
    assert l5["valid_for_pipeline"] is False
    assert result["full_pipeline_validation"]["valid"] is False


def test_l5_artifact_missing_blocks_validation(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "Gemini audit body"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    Path(result["run"]["stages"][-1]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH, result["run"], base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("L5_deepseek_acceptance:artifact_path_not_found" in error for error in validation["errors"])


def test_l5_output_does_not_contain_final_report_terms(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "Gemini audit body"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    packet = Path(result["run"]["stages"][-1]["artifact_path"]).read_text(encoding="utf-8")

    assert "final_controller_report" not in packet
    assert "FINAL CONTROLLER BODY" not in packet
    assert "final report" not in packet.lower()
    assert "最终建议" not in packet
    assert "长期发展的路线" not in packet


def test_internal_profiles_detect_foresight_mechanism_without_new_mode():
    import tools.task_engine_executors as executors

    profiles = executors._task_engine_profiles_from_query(
        "这是一个研究决策任务。未来10年 AI 降低知识获取成本后，ADHD 儿童优势/缺陷如何结构性反转？"
    )

    assert executors.PROFILE_FORESIGHT_MECHANISM in profiles
    assert len(CANONICAL_STAGES[ENGINE_RESEARCH]) == 6
    assert len(CANONICAL_STAGES[ENGINE_DECISION]) == 10
    assert len(CANONICAL_STAGES[ENGINE_RESEARCH_DECISION]) == 16
    assert CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9].stage_name == "evidence_judge"
    assert CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9].model == NEMOTRON120B
    assert CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][13].stage_name == "convergence_report"
    assert CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][13].model == R1_32B


def test_internal_profiles_keep_medical_review_evidence_grounded():
    import tools.task_engine_executors as executors

    profiles = executors._task_engine_profiles_from_query("ADHD 儿童最新医学研究进展、治疗方案和指南证据综述")

    assert profiles == [executors.PROFILE_EVIDENCE_GROUNDED]
    assert executors.PROFILE_FORESIGHT_MECHANISM not in profiles


def test_foresight_profile_guidance_enters_l1_prompt_only_for_foresight():
    import tools.task_engine_executors as executors

    foresight = executors._gemini_search_prompt(
        "未来10年 AI 降低知识获取成本后，ADHD 儿童优势/缺陷如何结构性反转？"
    )
    medical = executors._gemini_search_prompt("ADHD 儿童最新医学研究进展、治疗方案和指南证据综述")

    assert "evidence_support" in foresight
    assert "reasonable_inference" in foresight
    assert "foresight_hypothesis" in foresight
    assert "mechanism_chain" in foresight
    assert "uncertainty_boundary" in foresight
    assert "counterexample_or_failure" in foresight
    assert "Return source candidates as concise JSON-compatible notes" in foresight
    assert "max 8 source_candidates" in foresight
    assert "coverage_axes" in foresight
    assert "authoritative_guideline_or_consensus" in foresight
    assert "systematic_review_or_meta_analysis" in foresight
    assert "empirical_study_or_RCT" in foresight
    assert "mechanism_or_theory" in foresight
    assert "intervention_or_practice" in foresight
    assert "local_or_contextual_source" in foresight
    assert "controversy_or_counterevidence" in foresight
    assert "recent_update" in foresight
    assert "evidence_type, coverage_axis, and why_relevant" in foresight
    assert "known_gaps_for_L2" in foresight
    assert "do not present L1 inference as evidence" in foresight
    assert "Do not write long analysis" in foresight
    assert "final-style prose" in foresight
    assert "Return strict JSON only" not in foresight
    assert "max 6 source_candidates" not in foresight
    assert "foresight_hypothesis" not in medical
    assert "mechanism_chain" not in medical


def test_foresight_profile_guidance_enters_l3_and_l4_prompts(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = []
    for name in ("L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer"):
        stage_dir = tmp_path / name
        stage_dir.mkdir()
        artifact = stage_dir / ("report.md" if name != "L2_5_codex_evidence_organizer" else "claims.md")
        artifact.write_text("fresh evidence", encoding="utf-8")
        stages.append({
            "stage_name": name,
            "artifact_path": str(artifact),
            "outputs": {},
        })
    l3_prompt = executors._r1_synthesis_prompt_from_artifacts(
        stages,
        base_dir=tmp_path,
        query="未来10年 AI 结构性反转 机制推理",
    )
    assert "evidence_support" in l3_prompt
    assert "reasonable_inference" in l3_prompt
    assert "foresight_hypothesis" in l3_prompt
    assert "counterexample_or_failure" in l3_prompt

    l3_dir = tmp_path / "L3_r1_synthesis"
    l3_dir.mkdir()
    l3_artifact = l3_dir / "r1_synthesis.md"
    l3_artifact.write_text("fresh synthesis", encoding="utf-8")
    l4_prompt = executors._gemini_audit_prompt_from_artifacts(
        stages + [{"stage_name": "L3_r1_synthesis", "artifact_path": str(l3_artifact), "outputs": {}}],
        base_dir=tmp_path,
        query="未来10年 AI 结构性反转 机制推理",
    )
    assert "Audit whether L3 explicitly separates evidence_support" in l4_prompt
    assert "uncertainty_boundary" in l4_prompt


def test_decision_prompt_can_accept_research_packet_path_without_research_stages(tmp_path: Path):
    import tools.task_engine_executors as executors

    packet = tmp_path / "research_evidence_packet.md"
    packet.write_text(
        "verdict: ACCEPTED\naccepted: true\nclaim: research packet evidence summary\nexecutor_model: hidden-metadata\n",
        encoding="utf-8",
    )

    prompt = executors._decision_intelligence_prompt(
        "这是一个决策任务。请基于研究包判断。",
        base_dir=tmp_path / "decision",
        research_packet_path=packet,
    )

    assert "optional_research_evidence_packet_context" in prompt
    assert "TEXT ONLY CONTRACT" in prompt
    assert "research_packet_path:" not in prompt
    assert "Current run root" not in prompt
    assert str(tmp_path) not in prompt
    assert "research packet evidence summary" in prompt
    assert "research_packet_digest:" in prompt
    assert "do not dump raw research packet into the final report" in prompt
    assert "do not run or invent RESEARCH L1-L5 artifacts" in prompt
    assert "L1_gemini_search" not in prompt
    assert "L5_deepseek_acceptance" not in prompt
    assert "executor_model:" not in prompt


def test_decision_research_packet_context_prioritizes_fixed_section_digest(tmp_path: Path):
    import tools.task_engine_executors as executors

    packet = tmp_path / "research_evidence_packet.md"
    packet.write_text(
        "verdict: ACCEPTED\naccepted: true\nraw preface should not be the digest\n\n"
        "## evidence_strength\nStrong evidence section for DECISION use.\n\n"
        "## controversy\nControversy section for DECISION use.\n\n"
        "## evidence_gap\nEvidence gap section for DECISION use.\n\n"
        "## evidence_supported\nEvidence supported section for DECISION use.\n\n"
        "## reasonable_inference\nReasonable inference section for DECISION use.\n\n"
        "## foresight_hypothesis\nForesight hypothesis section for DECISION use.\n",
        encoding="utf-8",
    )

    context = executors._decision_research_packet_context(packet)

    assert "research_packet_digest:" in context
    assert "## evidence_strength" in context
    assert "## reasonable_inference" in context
    assert "## foresight_hypothesis" in context
    assert "raw preface should not be the digest" not in context


def test_l5_packet_records_foresight_acceptance_requirements(tmp_path: Path):
    import tools.task_engine_executors as executors

    records = []
    for name in ("L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer", "L3_r1_synthesis", "L4_gemini_audit"):
        stage_dir = tmp_path / name
        stage_dir.mkdir()
        artifact = stage_dir / "artifact.md"
        artifact.write_text("证据 evidence。合理推断。前瞻假设。机制链条。uncertainty_boundary。counterexample_or_failure。", encoding="utf-8")
        records.append({"stage_name": name, "artifact_path": str(artifact), "outputs": {}})

    packet = executors._research_acceptance_packet_from_artifacts(
        records,
        base_dir=tmp_path,
        query="未来10年 AI 结构性反转 机制推理",
    )

    assert executors.PROFILE_FORESIGHT_MECHANISM in packet["research_packet_profile"]
    joined = "\n".join(packet["profile_acceptance_requirements"])
    assert "evidence / inference / hypothesis distinction" in joined
    assert "mechanism_chain" in joined
    assert "uncertainty_boundary" in joined
    assert "counterexample_or_failure" in joined
    requirement_map = packet["artifact_summaries"]["_foresight_requirement_map"]
    assert "uncertainty_boundary: detected" in requirement_map
    assert "counterexample_or_failure: detected" in requirement_map


def test_l2_5_handoff_only_stub_is_invalid(tmp_path: Path):
    import tools.task_engine_executors as executors

    stage_dir = tmp_path / "L2_5_codex_evidence_organizer"
    stage_dir.mkdir()
    payload = json.dumps(
        {
            "handoff_protocol": "Hermes-Codex evidence organizer smoke",
            "inputs": {"source_candidates.json": "x", "ddgs_gap_sources.json": "y"},
            "outputs": ["sources.csv", "evidence.csv", "claims.md", "gaps.md"],
        },
        indent=2,
    )
    for name in ("sources.csv", "evidence.csv", "claims.md", "gaps.md"):
        (stage_dir / name).write_text(payload, encoding="utf-8")

    analysis = executors.analyze_l2_5_evidence_organizer(tmp_path)

    assert analysis["l2_5_valid"] is False
    assert analysis["l2_5_stub_detected"] is True
    assert analysis["upstream_critical_defect"] is True
    assert analysis["issues"] == ["l2_5_extraction_missing"]
    assert "L2_5_codex_evidence_organizer/sources.csv" in analysis["missing_or_invalid_artifacts"]


def test_l2_5_stub_marker_does_not_match_url_path_substrings():
    import tools.task_engine_executors as executors

    assert (
        executors._contains_l2_5_stub_marker(
            "https://www.gartner.com/en/articles/hype-cycle-for-emerging-technologies"
        )
        is False
    )
    assert executors._contains_l2_5_stub_marker("n/a") is True


def test_l2_5_prefers_explicit_original_question_over_l2_query(tmp_path: Path):
    import tools.task_engine_executors as executors

    original_question = "是否值得投资独立硬件 AI 伴侣设备，要求区分趋势信号、噪音和证据缺口？"
    l1 = tmp_path / "source_candidates.json"
    l2 = tmp_path / "ddgs_gap_sources.json"
    l1.write_text(
        json.dumps(
            {
                "source_candidates": [
                    {
                        "source": "https://example.test/humane-shutdown",
                        "evidence_type": "news",
                        "coverage_axis": "recent_update",
                        "why_relevant": "Confirms HP acquisition and Humane AI Pin shutdown.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    l2.write_text(
        json.dumps(
            [
                {
                    "query": "Confirms HP acquisition and Humane AI Pin shutdown recent_update",
                    "title": "Humane AI Pin shutdown",
                    "url": "https://example.test/humane",
                    "snippet": "HP acquired assets and the device was shut down.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs = executors.build_l2_5_evidence_organizer_outputs(
        {
            "source_candidates.json": str(l1),
            "ddgs_gap_sources.json": str(l2),
            "original_question": original_question,
        }
    )
    request = json.loads(outputs["evidence_runner_*.request.json"])

    assert request["user_question_anchor"] == original_question


def test_l2_5_real_extraction_is_valid(tmp_path: Path):
    import tools.task_engine_executors as executors

    l1 = tmp_path / "source_candidates.json"
    l2 = tmp_path / "ddgs_gap_sources.json"
    l1.write_text(
        json.dumps(
            {
                "source_candidates": [
                    {
                        "candidate": "https://debezium.io/documentation/reference/stable/connectors/postgresql.html",
                        "evidence_type": "Documentation",
                        "coverage_axis": "authoritative_guideline_or_consensus",
                        "why_relevant": "Specs for WAL-based PostgreSQL CDC replication and transactional impact during migration.",
                    },
                    {
                        "candidate": "https://iceberg.apache.org/docs/latest/",
                        "evidence_type": "Standard Specification",
                        "coverage_axis": "mechanism_or_theory",
                        "why_relevant": "Defines lakehouse table transactions and schema evolution over object storage.",
                    },
                    {
                        "candidate": "Query: PostgreSQL RLS multi tenant SaaS analytics lakehouse migration",
                        "evidence_type": "Search Query",
                        "coverage_axis": "local_or_contextual_source",
                        "why_relevant": "Targets tenant isolation and row-level security constraints for SaaS analytics migration.",
                    },
                    {
                        "candidate": "Query: streaming feature pipeline object storage SaaS analytics",
                        "evidence_type": "Search Query",
                        "coverage_axis": "intervention_or_practice",
                        "why_relevant": "Targets event-driven streaming feature pipeline tradeoffs for analytical workloads.",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    l2.write_text(
        json.dumps(
            [
                {
                    "query": "B2B SaaS PostgreSQL cron ETL CDC lakehouse object storage streaming pipeline",
                    "title": "Lakehouse architecture for SaaS analytics",
                    "url": "https://example.test/lakehouse",
                    "snippet": "Lakehouse object storage can isolate analytical workloads while adding governance and operational complexity.",
                },
                {
                    "query": "B2B SaaS PostgreSQL CDC Debezium lakehouse",
                    "title": "CDC migration risk",
                    "url": "https://example.test/cdc",
                    "snippet": "CDC reduces batch ETL delay but requires schema-change handling, replay strategy, and observability.",
                },
                {
                    "query": "multi tenant SaaS RLS lakehouse analytics",
                    "title": "Tenant isolation in analytics",
                    "url": "https://example.test/rls",
                    "snippet": "Multi-tenant SaaS analytics migration must preserve RLS, access controls, and customer-specific data boundaries.",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outputs = executors.build_l2_5_evidence_organizer_outputs(
        {"source_candidates.json": str(l1), "ddgs_gap_sources.json": str(l2)}
    )
    stage_dir = tmp_path / "L2_5_codex_evidence_organizer"
    stage_dir.mkdir()
    for name in ("sources.csv", "evidence.csv", "claims.md", "gaps.md"):
        (stage_dir / name).write_text(outputs[name], encoding="utf-8")

    analysis = executors.analyze_l2_5_evidence_organizer(tmp_path)

    assert analysis["l2_5_valid"] is True
    assert analysis["l2_5_stub_detected"] is False
    assert analysis["insufficient_sources"] is False
    assert analysis["source_rows"] >= 3
    assert analysis["evidence_rows"] >= 3
    assert analysis["claim_count"] >= 4
    assert analysis["gap_count"] >= 3


def _write_l2_5_inputs_and_analyze(tmp_path: Path, query: str, l1_items: list[dict], l2_items: list[dict]):
    import tools.task_engine_executors as executors

    l1 = tmp_path / "source_candidates.json"
    l2 = tmp_path / "ddgs_gap_sources.json"
    l1.write_text(json.dumps({"source_candidates": l1_items}, ensure_ascii=False), encoding="utf-8")
    l2.write_text(json.dumps([{**item, "query": query} for item in l2_items], ensure_ascii=False), encoding="utf-8")
    outputs = executors.build_l2_5_evidence_organizer_outputs(
        {"source_candidates.json": str(l1), "ddgs_gap_sources.json": str(l2)}
    )
    stage_dir = tmp_path / "L2_5_codex_evidence_organizer"
    stage_dir.mkdir()
    for name in ("sources.csv", "evidence.csv", "claims.md", "gaps.md"):
        (stage_dir / name).write_text(outputs[name], encoding="utf-8")
    (stage_dir / "evidence_runner_001.request.json").write_text(
        outputs["evidence_runner_*.request.json"],
        encoding="utf-8",
    )
    return executors.analyze_l2_5_evidence_organizer(tmp_path), stage_dir


def test_l2_5_audit_finance_query_is_not_bucketed_as_legal(tmp_path: Path):
    query = "未来10年 AI 持续降低审计底稿整理、财务异常检测、合规报告生成和管理层讨论分析草稿成本后，初级审计员和企业财务分析师的优势与劣势是否会发生结构性反转？"
    l1_items = [
        {
            "candidate": "audit expertise development paradox AI junior auditor",
            "evidence_type": "mechanism_or_theory",
            "coverage_axis": "mechanism_or_theory",
            "why_relevant": "AI automation reduces audit workpaper preparation and changes junior auditor learning-by-doing.",
        },
        {
            "candidate": "financial analyst AI MD&A anomaly detection workflow",
            "evidence_type": "empirical study",
            "coverage_axis": "empirical_study_or_RCT",
            "why_relevant": "Financial analysts shift from drafting MD&A and anomaly screening to validation and advisory work.",
        },
        {
            "candidate": "China audit AI working paper compliance guidance",
            "evidence_type": "local regulation / guideline",
            "coverage_axis": "local_or_contextual_source",
            "why_relevant": "Audit regulation constrains how AI-generated working papers and compliance reports can be relied on.",
        },
        {
            "candidate": "professional skepticism junior auditor AI review",
            "evidence_type": "controversy_or_counterevidence",
            "coverage_axis": "controversy_or_counterevidence",
            "why_relevant": "Professional skepticism may weaken if junior auditors skip manual evidence inspection experience.",
        },
    ]
    l2_items = [
        {"title": "AI reshapes auditing workflows", "url": "https://example.test/audit-ai", "snippet": "AI can draft audit workpapers and flag financial anomalies while requiring human review."},
        {"title": "MD&A drafting and financial analyst AI", "url": "https://example.test/mda-ai", "snippet": "Generative AI supports MD&A drafting and changes analyst review responsibilities."},
        {"title": "Audit professional skepticism under automation", "url": "https://example.test/skepticism", "snippet": "Automation creates counter-signals around overreliance and loss of junior training."},
        {"title": "Accounting compliance report automation", "url": "https://example.test/compliance", "snippet": "Compliance reports can be accelerated by AI but require accountability controls."},
    ]

    analysis, stage_dir = _write_l2_5_inputs_and_analyze(tmp_path, query, l1_items, l2_items)
    request = json.loads((stage_dir / "evidence_runner_001.request.json").read_text(encoding="utf-8"))

    assert request["sample_schema"] == "foresight_mechanism"
    assert "审计" in request["topic_anchor_terms"] or "财务" in request["topic_anchor_terms"]
    assert not {"lawyer", "contract", "case summarization", "legal ops"}.intersection(
        set(request["topic_anchor_terms"])
    )
    assert analysis["l2_5_valid"] is True
    assert analysis["l2_5_stub_detected"] is False
    assert analysis["insufficient_sources"] is False
    assert analysis["source_rows"] >= 3
    assert analysis["evidence_rows"] >= 3
    assert analysis["claim_count"] >= 4
    assert analysis["claim_source_alignment_valid"] is True


def test_l2_5_rag_architecture_query_extracts_architecture_claims(tmp_path: Path):
    query = "一个企业内部知识库产品是否应该从 Elasticsearch + 定时全量 embedding 任务，迁移到事件驱动增量索引 + 向量数据库 + hybrid reranker + 权限感知 RAG 架构？"
    l1_items = [
        {"candidate": "Elasticsearch scheduled full embedding indexing limitations", "evidence_type": "architecture note", "coverage_axis": "mechanism_or_theory", "why_relevant": "Scheduled full embedding jobs create stale retrieval and expensive reindexing for internal knowledge bases."},
        {"candidate": "event driven incremental indexing knowledge base architecture", "evidence_type": "practice report", "coverage_axis": "intervention_or_practice", "why_relevant": "Event-driven incremental indexing can reduce freshness lag and isolate document update failures."},
        {"candidate": "vector database hybrid reranker RAG permission filtering", "evidence_type": "technical guide", "coverage_axis": "authoritative_guideline_or_consensus", "why_relevant": "Hybrid reranking and permission-aware retrieval can improve relevance while preserving access control boundaries."},
        {"candidate": "RAG migration complexity observability evaluation", "evidence_type": "controversy_or_counterevidence", "coverage_axis": "controversy_or_counterevidence", "why_relevant": "Migration adds evaluation, chunking, permissions, latency, and operational complexity risks."},
    ]
    l2_items = [
        {"title": "Incremental indexing for RAG", "url": "https://example.test/incremental", "snippet": "Incremental indexing improves freshness but needs event guarantees and replay."},
        {"title": "Hybrid search and reranking", "url": "https://example.test/rerank", "snippet": "Hybrid search combines lexical and vector recall, then reranks for answer quality."},
        {"title": "Permission-aware RAG", "url": "https://example.test/permissions", "snippet": "Enterprise RAG must enforce document-level and user-level permissions before generation."},
        {"title": "Vector database migration risk", "url": "https://example.test/vector-risk", "snippet": "Vector database adoption adds cost, monitoring, and relevance evaluation requirements."},
    ]

    analysis, stage_dir = _write_l2_5_inputs_and_analyze(tmp_path, query, l1_items, l2_items)
    request = json.loads((stage_dir / "evidence_runner_001.request.json").read_text(encoding="utf-8"))
    claims = (stage_dir / "claims.md").read_text(encoding="utf-8")

    assert request["sample_schema"] == "tech_route"
    assert analysis["l2_5_valid"] is True
    assert analysis["insufficient_sources"] is False
    assert analysis["source_rows"] >= 3
    assert analysis["evidence_rows"] >= 3
    assert analysis["claim_count"] >= 4
    assert "current architecture" in claims
    assert "target architecture" in claims
    assert analysis["claim_source_alignment_valid"] is True


def test_l2_5_math_intervention_query_extracts_intervention_claims(tmp_path: Path):
    query = "8-10 岁数学计算困难、数感弱但智力正常的儿童，是否值得采用 数轴表征训练 + 明确策略教学 + 间隔检索练习， 而不是主要依赖刷题、数学游戏和泛化兴趣培养？"
    l1_items = [
        {"candidate": "number line training math learning difficulties children", "evidence_type": "systematic review", "coverage_axis": "systematic_review_or_meta_analysis", "why_relevant": "Number line representation training targets numerical magnitude and number sense in children with math difficulties."},
        {"candidate": "explicit strategy instruction arithmetic difficulties ages 8 10", "evidence_type": "intervention study", "coverage_axis": "empirical_study_or_RCT", "why_relevant": "Explicit strategy instruction supports arithmetic accuracy and reduces weak procedural guessing."},
        {"candidate": "spaced retrieval practice math fact fluency children", "evidence_type": "practice guide", "coverage_axis": "intervention_or_practice", "why_relevant": "Spaced retrieval practice strengthens math fact retention compared with massed worksheets."},
        {"candidate": "math games interest building transfer limitations", "evidence_type": "counterevidence", "coverage_axis": "controversy_or_counterevidence", "why_relevant": "Math games and interest cultivation may not transfer unless tied to explicit strategy and retrieval practice."},
    ]
    l2_items = [
        {"title": "Number line intervention evidence", "url": "https://example.test/number-line", "snippet": "Number line activities improve magnitude comparison and number sense outcomes."},
        {"title": "Explicit instruction for math difficulty", "url": "https://example.test/explicit", "snippet": "Explicit teaching benefits students with calculation difficulties when strategies are modeled."},
        {"title": "Spacing and retrieval in arithmetic", "url": "https://example.test/spacing", "snippet": "Spacing and retrieval improve durable arithmetic fact learning."},
        {"title": "Math games transfer risk", "url": "https://example.test/games", "snippet": "Games can increase engagement but may show weak transfer without targeted instruction."},
    ]

    analysis, stage_dir = _write_l2_5_inputs_and_analyze(tmp_path, query, l1_items, l2_items)
    request = json.loads((stage_dir / "evidence_runner_001.request.json").read_text(encoding="utf-8"))
    claims = (stage_dir / "claims.md").read_text(encoding="utf-8")

    assert request["sample_schema"] == "high_evidence_intervention"
    assert analysis["l2_5_valid"] is True
    assert analysis["l2_5_stub_detected"] is False
    assert analysis["insufficient_sources"] is False
    assert analysis["source_rows"] >= 3
    assert analysis["evidence_rows"] >= 3
    assert analysis["claim_count"] >= 4
    assert "population" in claims
    assert "intervention" in claims
    assert "outcome/evidence strength" in claims
    assert analysis["claim_source_alignment_valid"] is True


def test_supplementary_adhd_parent_training_is_cross_topic_for_non_adhd_samples():
    import tools.task_engine_executors as executors

    contaminated_text = """
    query: ADHD parent training children
    title: Parent Training in Behavior Management | Attention-Deficit / Hyperactivity Disorder (ADHD) | CDC
    url: https://www.cdc.gov/adhd/treatment/behavior-therapy.html
    snippet: parent training for children with ADHD
    """
    queries = [
        "B2B SaaS 分析产品是否应该从 PostgreSQL 单体 + cron ETL 迁移到事件驱动架构 + lakehouse/对象存储 + 流式特征管道？",
        "未来 10 年 AI 持续降低法律检索、合同起草、案例摘要和合规解释成本后，初级律师和企业法务分析师的优势与劣势是否会发生结构性反转？",
        "8-10 岁拼读困难儿童是否值得采用系统性音韵意识 + 明确解码训练 + 高频短时练习，而不是主要依赖泛阅读？",
    ]

    for query in queries:
        result = executors.detect_supplementary_search_topic_contamination(query, contaminated_text)
        assert result["supplementary_search_contaminated"] is True
        assert result["supplementary_contaminated"] is True
        assert result["supplementary_search_cross_topic_contamination"] is True
        assert result["issue"] == "supplementary_search_cross_topic_contamination"


def test_supplementary_queries_for_non_adhd_topics_do_not_include_parent_training():
    import tools.task_engine_executors as executors

    samples = [
        "B2B SaaS 分析产品是否应该从 PostgreSQL 单体 + cron ETL 迁移到事件驱动架构 + lakehouse/对象存储 + 流式特征管道？",
        "未来 10 年 AI 持续降低法律检索、合同起草、案例摘要和合规解释成本后，初级律师和企业法务分析师的优势与劣势是否会发生结构性反转？",
        "早期消费硬件基金是否应该提前下注 2030 年前家庭陪伴型具身 AI 机器人形成规模化消费市场？",
        "8-10 岁拼读困难儿童是否值得采用系统性音韵意识 + 明确解码训练 + 高频短时练习，而不是主要依赖泛阅读？",
    ]
    for query in samples:
        queries = executors._supplementary_search_queries(query)
        joined = " ".join(queries).lower()
        assert len(queries) >= 3
        assert "adhd parent training" not in joined
        assert "behavioral parent training" not in joined
        assert all(executors.detect_supplementary_search_topic_contamination(query, item)["supplementary_search_contaminated"] is False for item in queries)


def test_supplementary_queries_cover_ai_hardware_companion_topic():
    import tools.task_engine_executors as executors

    query = "独立硬件 AI 伴侣设备趋势是否真的在上升，并判断是否值得作为产品投资方向？"
    queries = executors._supplementary_search_queries(query)
    joined = " ".join(queries).lower()

    assert len(queries) >= 3
    assert "rabbit r1" in joined
    assert "humane ai pin" in joined
    assert "ai companion hardware" in joined
    assert "parent training" not in joined


def test_l5_packet_preserves_l4_critical_l2_5_extraction_missing(tmp_path: Path):
    import tools.task_engine_executors as executors

    records = []
    for name in ("L1_gemini_search", "L2_ddgs_supplement", "L3_r1_synthesis"):
        stage_dir = tmp_path / name
        stage_dir.mkdir()
        artifact = stage_dir / "artifact.md"
        artifact.write_text("domain evidence for PostgreSQL lakehouse migration", encoding="utf-8")
        records.append({"stage_name": name, "artifact_path": str(artifact), "outputs": {}})

    l2_5 = tmp_path / "L2_5_codex_evidence_organizer"
    l2_5.mkdir()
    payload = json.dumps(
        {
            "handoff_protocol": "Hermes-Codex evidence organizer smoke",
            "inputs": {"source_candidates.json": "x", "ddgs_gap_sources.json": "y"},
            "outputs": ["sources.csv", "evidence.csv", "claims.md", "gaps.md"],
        },
        indent=2,
    )
    outputs = {}
    for filename in ("sources.csv", "evidence.csv", "claims.md", "gaps.md"):
        path = l2_5 / filename
        path.write_text(payload, encoding="utf-8")
        outputs[filename] = str(path)
    records.insert(2, {"stage_name": "L2_5_codex_evidence_organizer", "artifact_path": str(l2_5), "outputs": outputs})

    l4 = tmp_path / "L4_gemini_audit"
    l4.mkdir()
    audit = l4 / "gemini_audit_report.md"
    audit.write_text(
        "DEFECT [Critical] - L2.5 Extraction Missing: sources.csv, evidence.csv, claims.md, gaps.md are stubbed out.",
        encoding="utf-8",
    )
    records.append({"stage_name": "L4_gemini_audit", "artifact_path": str(audit), "outputs": {}})

    packet = executors._research_acceptance_packet_from_artifacts(
        records,
        base_dir=tmp_path,
        query="B2B SaaS PostgreSQL lakehouse migration evidence grounded decision",
    )
    rendered = LocalTaskEngineExecutor().run_controller_acceptance(
        CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][5],
        packet,
    )

    assert "l2_5_valid: false" in rendered
    assert "critical_defects: [l2_5_extraction_missing]" in rendered
    assert "missing_or_invalid_artifacts: []" not in rendered
    assert "L2_5_codex_evidence_organizer/sources.csv" in rendered
    assert "evidence_packet_ready_for_decision: false" in rendered


def test_l5_acceptance_does_not_treat_domain_rejecting_as_audit_rejection():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_EVIDENCE_GROUNDED],
        "profile_acceptance_requirements": executors._research_profile_acceptance_requirements(
            [executors.PROFILE_EVIDENCE_GROUNDED]
        ),
        "missing_or_invalid_artifacts": [],
        "critical_defects": [],
        "l2_5_valid": True,
        "l2_5_analysis": {
            "l2_5_valid": True,
            "l2_5_stub_detected": False,
            "insufficient_sources": False,
            "issues": [],
            "missing_or_invalid_artifacts": [],
            "source_rows": 4,
            "evidence_rows": 4,
            "claim_count": 4,
            "gap_count": 3,
            "claim_source_alignment_valid": True,
        },
        "artifact_summaries": {
            "L1_gemini_search": "UNCLOS, Ilulissat Declaration, CLCS, Arctic maritime claims evidence.",
            "L2_ddgs_supplement": "Supplementary Arctic sovereignty and maritime dispute sources.",
            "L2_5_codex_evidence_organizer": "sources.csv evidence.csv claims.md gaps.md contain legal evidence rows.",
            "L3_r1_synthesis": "The Arctic route sovereignty dispute is governed by UNCLOS with bounded inference.",
            "L4_gemini_audit": "2008 Ilulissat Declaration reaffirms commitment to UNCLOS, rejecting new comprehensive treaties. Status: Supported.",
        },
        "claim_table": [
            {
                "claim_id": "C1",
                "claim_text": "UNCLOS provides a current legal framework for bounded Arctic maritime dispute claims.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "medium",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Ilulissat Declaration",
                        "url_or_stable_locator": "https://example.test/ilulissat",
                        "source_type": "legal_source",
                        "support_type": "full_text_verified",
                    }
                ],
                "applicability_boundary": "Applies to legal-framework claims, not factual control claims.",
                "counter_signal_or_failure_condition": "Later treaties or contradictory legal sources would downgrade it.",
                "evidence_gap": "current_state_practice_gap",
                "decision_use": "can_support_decision",
                "notes": "domain rejection wording is not an audit rejection.",
            }
        ],
        "audit_text": "2008 Ilulissat Declaration reaffirms commitment to UNCLOS, rejecting new comprehensive treaties. Status: Supported.",
        "audit_summary": "L4 audit supported the UNCLOS and Arctic maritime dispute claims.",
    }

    rendered = LocalTaskEngineExecutor().run_controller_acceptance(
        CANONICAL_STAGES[ENGINE_RESEARCH][5],
        packet,
    )

    assert "verdict: ACCEPTED" in rendered
    assert "accepted: true" in rendered
    assert "evidence_packet_ready_for_decision: true" in rendered


def test_l5_research_packet_quality_blocks_missing_fixed_sections():
    import tools.task_engine_executors as executors

    thin = (
        "research_evidence_packet\n"
        "verdict: ACCEPTED\n"
        "accepted: true\n"
        "evidence_packet_ready_for_decision: true\n"
        "audit_summary: accepted\n"
    )

    error = executors._research_evidence_packet_quality_error(thin)

    assert error.startswith("missing_research_packet_sections:")
    assert "evidence_strength" in error
    assert "foresight_hypothesis" in error


def test_l5_research_packet_quality_blocks_acceptance_summary_only():
    import tools.task_engine_executors as executors

    section_body = "Requirements satisfied and accepted for decision handoff only. No reusable content is provided."
    text = "\n".join(
        [
            "research_evidence_packet",
            "verdict: ACCEPTED",
            "accepted: true",
            "evidence_packet_ready_for_decision: true",
            "",
            *[f"## {heading}\n{section_body}\n" for heading in executors.RESEARCH_PACKET_FIXED_HEADINGS],
        ]
    )

    assert executors._research_evidence_packet_quality_error(text) in {
        "acceptance_summary_only",
        "missing_claim_table",
    }


def test_l5_research_packet_quality_blocks_raw_metadata():
    import tools.task_engine_executors as executors

    text = _complete_research_evidence_packet_text() + "\nexecutor_model: hidden\n"

    assert executors._research_evidence_packet_quality_error(text) == "raw_metadata_leak"


def test_l5_research_packet_quality_accepts_compact_evidence_packet():
    import tools.task_engine_executors as executors

    text = _complete_research_evidence_packet_text()

    assert "verdict: ACCEPTED" in text
    assert "accepted: true" in text
    assert executors._research_evidence_packet_quality_error(text) == ""
    executors._assert_artifact_quality(CANONICAL_STAGES[ENGINE_RESEARCH][-1], text)


def test_research_evidence_packet_quality_blocks_shell_packet_without_claim_table():
    import tools.task_engine_executors as executors

    shell = "\n".join(
        [
            "research_evidence_packet",
            "verdict: ACCEPTED",
            "accepted: true",
            "evidence_packet_ready_for_decision: true",
            "",
            "## evidence_strength",
            "Evidence strength discusses evidence in general terms without any claim-level rows or source anchors.",
            "",
            "## controversy",
            "Controversy remains generic and does not bind any concrete claim to a source or limitation.",
            "",
            "## evidence_gap",
            "Evidence gap is generic and lacks applicability boundaries or verification status.",
            "",
            "## evidence_supported",
            "Evidence supported is a template paragraph only.",
            "",
            "## reasonable_inference",
            "Reasonable inference is a template paragraph only.",
            "",
            "## foresight_hypothesis",
            "Foresight hypothesis is a template paragraph only.",
        ]
    )

    assert executors._research_evidence_packet_quality_error(shell).startswith("missing_research_packet_sections:")


def test_research_evidence_packet_quality_blocks_top_level_without_claim_rows():
    import tools.task_engine_executors as executors

    text = _complete_research_evidence_packet_text().replace(
        "- claim_id: C1",
        "- missing_claim: C1",
    )

    assert executors._research_evidence_packet_quality_error(text) == "missing_claim_table"


def test_claim_packet_contract_blocks_fake_claim_id_with_empty_template_claim():
    import tools.task_engine_executors as executors

    result = executors._research_claim_contract_validation(
        claim_table=[
            {
                "claim_id": "C1",
                "claim_text": "Evidence supported. More research is needed. Decision relevant.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "medium",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Verified source",
                        "url_or_stable_locator": "https://example.test/source",
                        "source_type": "peer_reviewed_source",
                        "support_type": "full_text_verified",
                    }
                ],
                "applicability_boundary": "Applies to the decision.",
                "counter_signal_or_failure_condition": "Contradictory evidence.",
                "evidence_gap": "none",
                "decision_use": "can_support_decision",
            }
        ],
        l4_defect_report={"critical_defects": [], "noncritical_defects": [], "verification_required": [], "evidence_gaps": [], "handoff_caveats": []},
        l2_5_analysis={"l2_5_stub_detected": False, "insufficient_sources": False},
    )

    assert result["valid"] is False
    assert "C1:thin_or_template_claim_text" in result["blocking_errors"]


def test_claim_packet_contract_blocks_fake_source_anchor_without_locator_or_support_type():
    import tools.task_engine_executors as executors

    result = executors._research_claim_contract_validation(
        claim_table=[
            {
                "claim_id": "C1",
                "claim_text": "A concrete current evidence claim with enough detail to require source grounding.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "medium",
                "source_anchors": [{"source_id": "S1"}],
                "applicability_boundary": "Bounded to source population.",
                "counter_signal_or_failure_condition": "Contradictory source.",
                "evidence_gap": "transfer_gap",
                "decision_use": "can_support_decision",
            }
        ],
        l4_defect_report={"critical_defects": [], "noncritical_defects": [], "verification_required": [], "evidence_gaps": [], "handoff_caveats": []},
        l2_5_analysis={"l2_5_stub_detected": False, "insufficient_sources": False},
    )

    assert result["valid"] is False
    assert "C1:source_anchor_missing_locator_or_title" in result["blocking_errors"]
    assert "C1:source_anchor_missing_support_type" in result["blocking_errors"]


def test_claim_packet_contract_blocks_snippet_source_marked_full_text_verified():
    import tools.task_engine_executors as executors

    result = executors._research_claim_contract_validation(
        claim_table=[
            {
                "claim_id": "C1",
                "claim_text": "A concrete claim derived from a search result snippet must not be marked full-text verified.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "high",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Search result",
                        "url_or_stable_locator": "https://example.test/snippet",
                        "source_type": "search_result_snippet",
                        "support_type": "full_text_verified",
                    }
                ],
                "applicability_boundary": "Bounded to source population.",
                "counter_signal_or_failure_condition": "Full text may contradict the snippet.",
                "evidence_gap": "requires_full_text_verification",
                "decision_use": "can_support_decision",
            }
        ],
        l4_defect_report={"critical_defects": [], "noncritical_defects": [], "verification_required": [], "evidence_gaps": [], "handoff_caveats": []},
        l2_5_analysis={"l2_5_stub_detected": False, "insufficient_sources": False},
    )

    assert result["valid"] is False
    assert "C1:snippet_source_marked_full_text_verified" in result["blocking_errors"]


def test_research_packet_quality_blocks_l4_defects_dropped_from_clean_handoff():
    import tools.task_engine_executors as executors

    text = _complete_research_evidence_packet_text().replace(
        "audit_summary: L4 audit accepted the compact evidence packet.",
        "audit_summary: L4 audit status PASS WITH DEFECTS; DEFECT 1 missed counter-signal evidence.",
    )

    assert executors._research_evidence_packet_quality_error(text) == "l4_defects_not_propagated"


def test_claim_packet_snippet_only_core_claim_degrades_to_conditional_ready():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_EVIDENCE_GROUNDED],
        "profile_acceptance_requirements": executors._research_profile_acceptance_requirements(
            [executors.PROFILE_EVIDENCE_GROUNDED]
        ),
        "missing_or_invalid_artifacts": [],
        "critical_defects": [],
        "l2_5_valid": True,
        "l2_5_analysis": {"l2_5_valid": True, "l2_5_stub_detected": False, "insufficient_sources": False},
        "claim_table": [
            {
                "claim_id": "C1",
                "claim_text": "Current evidence suggests a bounded intervention effect.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "low",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Search result source",
                        "url_or_stable_locator": "https://example.test/snippet",
                        "source_type": "search_result_snippet",
                        "support_type": "requires_full_text_verification",
                    }
                ],
                "applicability_boundary": "Preliminary evidence only.",
                "counter_signal_or_failure_condition": "Full text may not support the snippet.",
                "evidence_gap": "requires_full_text_verification",
                "decision_use": "use_with_caution",
                "notes": "snippet-only fixture",
            }
        ],
        "artifact_summaries": {"L2_5_codex_evidence_organizer": "snippet evidence"},
        "audit_text": "",
        "audit_summary": "L4 audit artifact present.",
    }

    rendered = LocalTaskEngineExecutor().run_controller_acceptance(CANONICAL_STAGES[ENGINE_RESEARCH][-1], packet)

    assert "verdict: ACCEPTED_WITH_DEFECTS" in rendered
    assert "accepted: true" in rendered
    assert "verification_required: [C1:evidence_supported_requires_full_text_verification, C1:requires_full_text_verification]" in rendered
    assert "evidence_packet_ready_for_decision: conditional" in rendered
    assert "evidence_packet_ready_for_decision: true" not in rendered
    assert "support_type" not in rendered or "full_text_verified" not in rendered


def test_l4_pass_with_defects_propagates_to_conditional_handoff():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "profile_acceptance_requirements": executors._research_profile_acceptance_requirements(
            [executors.PROFILE_FORESIGHT_MECHANISM]
        ),
        "missing_or_invalid_artifacts": [],
        "critical_defects": [],
        "l2_5_valid": True,
        "l2_5_analysis": {"l2_5_valid": True, "l2_5_stub_detected": False, "insufficient_sources": False},
        "claim_table": [
            {
                "claim_id": "C1",
                "claim_text": "Future interface shift may change the value of current capabilities.",
                "epistemic_tier": "foresight_hypothesis",
                "evidence_strength": "low",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Scenario source",
                        "url_or_stable_locator": "https://example.test/scenario",
                        "source_type": "source_candidate",
                        "support_type": "secondary_summary",
                    }
                ],
                "applicability_boundary": "Future scenario only.",
                "counter_signal_or_failure_condition": "Future interfaces may reverse the mechanism.",
                "evidence_gap": "future_environment_gap",
                "decision_use": "use_with_caution",
                "notes": "foresight fixture",
            }
        ],
        "artifact_summaries": {
            "L3_r1_synthesis": (
                "evidence basis plus reasonable inference and foresight hypothesis. "
                "mechanism_chain input variables to mediating mechanisms to output variables. "
                "uncertainty_boundary and counterexample_or_failure conditions are explicit."
            )
        },
        "audit_text": "Status: PASS WITH DEFECTS\nDEFECT 1 (Missed Counter-Signal Evidence): missing interface modality counter-signal.",
        "audit_summary": "PASS WITH DEFECTS",
    }

    rendered = LocalTaskEngineExecutor().run_controller_acceptance(CANONICAL_STAGES[ENGINE_RESEARCH][-1], packet)

    assert "verdict: ACCEPTED_WITH_DEFECTS" in rendered
    assert "noncritical_defects:" in rendered
    assert "l4_pass_with_defects" in rendered
    assert "missed_counter_signal" in rendered
    assert "handoff_caveats:" in rendered
    assert "evidence_packet_ready_for_decision: conditional" in rendered


def test_valid_claim_packet_good_is_clean_ready():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_EVIDENCE_GROUNDED],
        "profile_acceptance_requirements": executors._research_profile_acceptance_requirements(
            [executors.PROFILE_EVIDENCE_GROUNDED]
        ),
        "missing_or_invalid_artifacts": [],
        "critical_defects": [],
        "l2_5_valid": True,
        "l2_5_analysis": {"l2_5_valid": True, "l2_5_stub_detected": False, "insufficient_sources": False},
        "claim_table": [
            {
                "claim_id": "C1",
                "claim_text": "Full-text verified current evidence supports this bounded claim.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "medium",
                "source_anchors": [
                    {
                        "source_id": "S1",
                        "title": "Verified source",
                        "url_or_stable_locator": "https://example.test/full-text",
                        "source_type": "peer_reviewed_source",
                        "support_type": "full_text_verified",
                    }
                ],
                "applicability_boundary": "Bounded to source population.",
                "counter_signal_or_failure_condition": "Contradictory full-text source.",
                "evidence_gap": "long_horizon_transfer_gap",
                "decision_use": "can_support_decision",
                "notes": "valid fixture",
            }
        ],
        "artifact_summaries": {"L2_5_codex_evidence_organizer": "verified evidence"},
        "audit_text": "Status: Supported.",
        "audit_summary": "L4 supported the claim.",
    }

    rendered = LocalTaskEngineExecutor().run_controller_acceptance(CANONICAL_STAGES[ENGINE_RESEARCH][-1], packet)

    assert "verdict: ACCEPTED\n" in rendered
    assert "accepted: true" in rendered
    assert "evidence_packet_ready_for_decision: true" in rendered
    assert "claim_id: C1" in rendered


def test_foresight_claim_boundary_blocks_future_as_evidence_supported():
    import tools.task_engine_executors as executors

    claim_table = [
        {
            "claim_id": "C1",
            "claim_text": "Future AI interfaces will make this capability valuable over the next 10 years.",
            "epistemic_tier": "evidence_supported",
            "evidence_strength": "medium",
            "source_anchors": [{"source_id": "S1", "support_type": "full_text_verified"}],
        }
    ]

    result = executors._research_claim_contract_validation(
        claim_table=claim_table,
        l4_defect_report={"critical_defects": [], "noncritical_defects": [], "verification_required": [], "evidence_gaps": [], "handoff_caveats": []},
        l2_5_analysis={"l2_5_stub_detected": False, "insufficient_sources": False},
    )

    assert result["valid"] is False
    assert "C1:future_claim_misclassified_as_evidence_supported" in result["blocking_errors"]


def test_missing_source_anchor_bad_invalidates_claim_packet():
    import tools.task_engine_executors as executors

    result = executors._research_claim_contract_validation(
        claim_table=[
            {
                "claim_id": "C1",
                "claim_text": "Claim without retained source anchor.",
                "epistemic_tier": "evidence_supported",
                "evidence_strength": "insufficient",
                "source_anchors": [],
            }
        ],
        l4_defect_report={"critical_defects": [], "noncritical_defects": [], "verification_required": [], "evidence_gaps": [], "handoff_caveats": []},
        l2_5_analysis={"l2_5_stub_detected": False, "insufficient_sources": False},
    )

    assert result["valid"] is False
    assert "C1:missing_source_anchors" in result["blocking_errors"]


def test_decision_handoff_caveats_good_allows_accepted_with_defects():
    import tools.task_engine_executors as executors

    text = _complete_research_evidence_packet_text().replace(
        "verdict: ACCEPTED",
        "verdict: ACCEPTED_WITH_DEFECTS",
    ).replace(
        "verification_required: []",
        "verification_required: [C1:requires_full_text_verification]",
    ).replace(
        "handoff_caveats: []",
        "handoff_caveats: [C1 must be used with caution]",
    ).replace(
        "evidence_packet_ready_for_decision: true",
        "evidence_packet_ready_for_decision: conditional",
    )

    assert executors._research_evidence_packet_quality_error(text) == ""
    assert executors._l5_acceptance_text_is_accepted(text) is True


def test_foresight_l5_maps_boundary_and_counterexample_synonyms():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "artifact_summaries": {
            "L3_r1_synthesis": (
                "基础证据 evidence。合理推断 hypothesis。输入变量 → 中介机制 → 输出变量构成机制链条。"
                "边界条件包括 AI 反馈质量与学校要求；失效条件是孩子不保留验证步骤。"
                "反证信号包括兴趣驱动下降和延迟反馈耐受没有改善。"
            )
        },
    }

    assert executors._research_packet_profile_acceptance_issues(packet) == []


def test_foresight_l5_accepts_without_direct_future_evidence_when_boundaries_present():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "artifact_summaries": {
            "L1_gemini_search": "基础证据来自 ADHD 执行功能研究和学习支持 evidence basis。",
            "L3_r1_synthesis": (
                "这不是直接未来证据，而是合理推断 / hypothesis。"
                "输入变量 → 中介机制 → 输出变量构成机制链条。"
                "不确定性边界明确，置信中等；反例和失败条件包括学校环境不变、AI 使用缺少验证。"
            ),
            "L4_gemini_audit": "未把前瞻假设伪装成医学定论。",
        },
    }

    assert executors._research_packet_profile_acceptance_issues(packet) == []


def test_foresight_l5_rejects_missing_uncertainty_or_counterexample():
    import tools.task_engine_executors as executors

    packet = {
        "research_packet_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "artifact_summaries": {
            "L3_r1_synthesis": "基础证据 evidence。合理推断 hypothesis。机制链条：输入变量 → 中介机制 → 输出变量。"
        },
    }

    issues = executors._research_packet_profile_acceptance_issues(packet)

    assert "foresight_mechanism_missing:uncertainty_boundary" in issues
    assert "foresight_mechanism_missing:counterexample_or_failure" in issues


def test_output_quality_profile_blocks_foresight_final_without_mechanism_chain():
    import tools.task_engine_executors as executors

    text = "# 前瞻报告\n\n关键驱动变量：AI。确定性等级：中。情景分叉：A/B。可观察指标：核查事实。"
    errors = executors._quality_profile_errors(
        text,
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="final_controller_report",
    )

    assert "missing_mechanism_chain" in errors


def test_convergence_report_foresight_prompt_requires_quality_fields(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = []
    for spec in CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][:13]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifact = stage_dir / "artifact.md"
        artifact.write_text("fresh stage artifact", encoding="utf-8")
        stages.append({
            "stage_name": spec.stage_name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact),
            "outputs": {},
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
        })

    prompt = executors._convergence_report_prompt_from_artifacts(
        stages,
        query="未来10年 AI 结构性反转 ADHD 机制推理",
        base_dir=tmp_path,
    )

    assert "key_drivers" in prompt
    assert "mechanism_chain" in prompt
    assert "scenario_branches" in prompt
    assert "counter_signals" in prompt
    assert "falsification_signals" in prompt
    assert "uncertainty_boundary" in prompt
    assert "certainty_levels" in prompt
    assert "Do not rename these headings" in prompt
    assert "## key_drivers" in prompt
    assert "## mechanism_chain" in prompt
    assert "## scenario_branches" in prompt
    assert "## counter_signals" in prompt
    assert "## certainty_levels" in prompt
    assert "## uncertainty_boundary" in prompt


def _decision_prior_stage_records(tmp_path: Path) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for spec in CANONICAL_STAGES[ENGINE_DECISION][:7]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifact = stage_dir / spec.required_outputs[0]
        artifact.write_text(f"{spec.stage_name} fresh artifact", encoding="utf-8")
        stages.append({
            "stage_name": spec.stage_name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact),
            "outputs": {spec.required_outputs[0]: str(artifact)},
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
        })
    return stages


def _decision_final_prior_stage_records(tmp_path: Path) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for spec in CANONICAL_STAGES[ENGINE_DECISION][:9]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifact = stage_dir / spec.required_outputs[0]
        artifact.write_text(f"{spec.stage_name} fresh artifact", encoding="utf-8")
        stages.append({
            "stage_name": spec.stage_name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact),
            "outputs": {spec.required_outputs[0]: str(artifact)},
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
        })
    return stages


def test_decision_convergence_prompt_with_research_packet_requires_fixed_foresight_headings(tmp_path: Path):
    import tools.task_engine_executors as executors

    packet = tmp_path / "research_evidence_packet.md"
    packet.write_text("compact research packet digest", encoding="utf-8")
    stage = CANONICAL_STAGES[ENGINE_DECISION][7]

    prompt = executors._decision_stage_prompt(
        stage,
        _decision_prior_stage_records(tmp_path),
        query="未来10年 AI 降低知识获取成本后 ADHD 结构性反转 decision",
        base_dir=tmp_path,
        research_packet_path=packet,
    )

    for heading in (
        "## key_drivers",
        "## mechanism_chain",
        "## scenario_branches",
        "## counter_signals",
        "## certainty_levels",
        "## uncertainty_boundary",
    ):
        assert heading in prompt
    assert "Do not rename these headings" in prompt
    assert "Do not translate these headings" in prompt
    assert "Do not merge these headings" in prompt
    assert "Do not omit these headings" in prompt
    assert "research_packet_digest" in prompt


def test_decision_convergence_prompt_standalone_and_research_packet_paths_match_fixed_headings(tmp_path: Path):
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][7]
    stages = _decision_prior_stage_records(tmp_path)
    packet = tmp_path / "research_evidence_packet.md"
    packet.write_text("compact research packet digest", encoding="utf-8")
    kwargs = {
        "stage": stage,
        "stages": stages,
        "query": "future scenario structural reversal for ADHD and AI",
        "base_dir": tmp_path,
    }

    standalone = executors._decision_stage_prompt(**kwargs)
    with_packet = executors._decision_stage_prompt(**kwargs, research_packet_path=packet)

    fixed_headings = (
        "## key_drivers",
        "## mechanism_chain",
        "## scenario_branches",
        "## counter_signals",
        "## certainty_levels",
        "## uncertainty_boundary",
    )
    for heading in fixed_headings:
        assert heading in standalone
        assert heading in with_packet


def test_decision_convergence_prompt_does_not_add_foresight_headings_for_evidence_grounded(tmp_path: Path):
    import tools.task_engine_executors as executors

    prompt = executors._decision_stage_prompt(
        CANONICAL_STAGES[ENGINE_DECISION][7],
        _decision_prior_stage_records(tmp_path),
        query="ADHD 儿童医学研究进展和治疗方案证据综述",
        base_dir=tmp_path,
    )

    assert "## key_drivers" not in prompt
    assert "## mechanism_chain" not in prompt
    assert "## scenario_branches" not in prompt
    assert "## counter_signals" not in prompt
    assert "## certainty_levels" not in prompt


def test_convergence_report_foresight_quality_gate_blocks_missing_fields():
    import tools.task_engine_executors as executors

    text = "convergence_decision_framework\n确定性等级：中。"
    errors = executors._quality_profile_errors(
        text,
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="convergence_report",
    )

    assert "missing_key_drivers" in errors
    assert "missing_mechanism_chain" in errors
    assert "missing_scenario_branches" in errors
    assert "missing_counter_signals" in errors


def test_convergence_report_foresight_quality_gate_blocks_missing_certainty_levels():
    import tools.task_engine_executors as executors

    text = (
        "## key_drivers\nAI feedback cost.\n"
        "## mechanism_chain\ninput variable -> mediating mechanism -> output variable.\n"
        "## scenario_branches\nScenario A verifies; Scenario B does not.\n"
        "## counter_signals\nobservable signal shows no decline.\n"
        "## uncertainty_boundary\nevidence stops at current research."
    )
    errors = executors._quality_profile_errors(
        text,
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="convergence_report",
    )

    assert "missing_certainty_levels" in errors


def test_convergence_report_foresight_quality_gate_accepts_required_fields():
    import tools.task_engine_executors as executors

    text = (
        "## key_drivers\nAI feedback cost, task selection pressure.\n"
        "## mechanism_chain\ninput variable -> mediating mechanism -> output variable.\n"
        "## scenario_branches\nScenario A keeps verification; Scenario B outsources verification.\n"
        "## counter_signals\nfalsification_signals: observable signal shows no decline in validation behavior.\n"
        "## certainty_levels\nhigh / medium / low.\n"
        "## uncertainty_boundary\nevidence stops at current ADHD and learning-support research."
    )

    assert executors._quality_profile_errors(
        text,
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="convergence_report",
    ) == []


def test_output_quality_profile_accepts_future_scenario_alias_with_fixed_headings():
    import tools.task_engine_executors as executors

    text = (
        "## key_drivers\nAI.\n"
        "## mechanism_chain\ninput variable -> mediating mechanism -> output variable.\n"
        "## scenario_branches\nScenario A; Scenario B.\n"
        "## counter_signals\nobservable signal.\n"
        "## certainty_levels\nhigh / medium / low.\n"
        "## uncertainty_boundary\nboundary."
    )

    assert executors._quality_profile_errors(
        text,
        [executors.PROFILE_FUTURE_SCENARIO],
        stage_name="convergence_report",
    ) == []


def test_output_quality_profile_blocks_foresight_final_without_judgment_units():
    import tools.task_engine_executors as executors

    text = (
        "## 未来优势变陷阱 Top5\n关键驱动变量：AI 降低知识获取成本。确定性等级：高 / 中 / 低。\n"
        "## 未来缺陷变优势 Top5\n输入变量 → 中介机制 → 输出变量：低价值重复减少 → 任务价值敏感度提升 → 筛选优势。\n"
        "## 最危险的错误培养路径\n情景分叉：情景 A 保留验证；情景 B 替代判断。\n"
        "## danger_flag\n可观察指标 / 反证信号：是否保留推理痕迹。"
    )

    errors = executors._quality_profile_errors(
        text,
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="final_controller_report",
    )

    assert "missing_judgment_unit_fields" in errors
    assert "insufficient_evidence_tier_bindings" in errors
    assert "insufficient_decision_use_bindings" in errors


def test_user_forbids_advice_still_blocks_advice_terms():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "这是一个决策任务。不要建议，不要培养计划，不要文献综述。",
        "output_quality_profile": [executors.PROFILE_EVIDENCE_GROUNDED],
    }

    try:
        executors._assert_final_controller_packet_quality(packet, "# 决策任务最终报告\n\n## 下一步\n建议方向：专业评估。")
    except RuntimeError as exc:
        assert "forbidden_user_terms" in str(exc)
    else:
        raise AssertionError("user-forbidden advice terms must be blocked")


def test_decision_final_packet_with_research_packet_requires_evidence_boundary_keys(tmp_path: Path):
    import tools.task_engine_executors as executors

    research_packet = tmp_path / "research_evidence_packet.md"
    research_packet.write_text("compact research packet digest", encoding="utf-8")

    packet = executors._decision_final_controller_packet(
        _decision_final_prior_stage_records(tmp_path),
        query="未来10年 AI 降低知识获取成本后 ADHD 结构性反转 decision，需要证据边界",
        base_dir=tmp_path,
        research_packet_path=research_packet,
    )

    requirements = packet["final_report_requirements"]
    assert requirements["evidence_boundary_required"] is True
    assert requirements["evidence_boundary_heading"] == "## 证据边界"
    assert requirements["evidence_boundary_keys"] == ["evidence_strength", "controversy", "evidence_gap"]
    assert "not a literature review" in requirements["evidence_boundary_policy"]


def test_decision_final_packet_preserves_current_run_metadata_for_stale_gate(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = _decision_final_prior_stage_records(tmp_path)
    packet = executors._decision_final_controller_packet(
        stages,
        query="请基于当前研究成果包做最终决策。",
        base_dir=tmp_path,
    )

    first = packet["stage_trace"][0]
    assert first["stage_name"] == "intelligence_layer"
    assert first["created_in_current_run"] is True
    assert first["legacy_contaminated"] is False
    assert first["valid_for_pipeline"] is True
    assert not Path(first["artifact_path"]).is_absolute()

    text = executors._final_controller_report_from_packet(packet)
    executors._assert_final_controller_packet_quality(packet, text)


def test_decision_final_legal_evidence_boundary_uses_current_domain_not_adhd(tmp_path: Path):
    import tools.task_engine_executors as executors

    research_packet = tmp_path / "research_evidence_packet.md"
    research_packet.write_text(
        "evidence_supported: disputed maritime activity advice requires legal, maritime, and compliance boundaries.",
        encoding="utf-8",
    )
    packet = executors._decision_final_controller_packet(
        _decision_final_prior_stage_records(tmp_path),
        query="请评估南海主权争议和海洋法边界下，跨境海洋数据产品能否包装成普通商业建议。",
        base_dir=tmp_path,
        research_packet_path=research_packet,
    )

    text = executors._final_controller_report_from_packet(packet)

    assert "## 证据边界" in text
    assert "海洋法" in text
    assert "合规" in text
    assert "ADHD" not in text
    assert "执行功能" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_final_controller_quality_blocks_evidence_grounded_without_boundary_fields():
    import tools.task_engine_executors as executors

    errors = executors._quality_profile_errors(
        "# 决策任务最终报告\n\n## 结论\n只有判断，没有边界字段。",
        [executors.PROFILE_EVIDENCE_GROUNDED],
        stage_name="final_controller_report",
    )

    assert "missing_evidence_strength" in errors
    assert "missing_controversy" in errors
    assert "missing_gap" in errors


def test_final_controller_quality_accepts_short_evidence_boundary_section():
    import tools.task_engine_executors as executors

    text = "\n".join([
        "# 决策任务最终报告",
        "",
        "## 证据边界",
        "evidence_strength: strong for ADHD execution support; medium for AI-mediated feedback; weak for ten-year individual forecasts.",
        "controversy: structural reversal depends on school context, tool design, and verification habits.",
        "evidence_gap: no direct longitudinal evidence for a 100x lower knowledge-cost environment.",
    ])

    assert executors._quality_profile_errors(
        text,
        [executors.PROFILE_EVIDENCE_GROUNDED],
        stage_name="final_controller_report",
    ) == []


def test_user_forbids_literature_review_allows_short_evidence_boundary():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": _decision_future_query(),
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "research_evidence_packet_context": "research_packet_excerpt: compact digest only",
    }
    text = executors._final_controller_report_from_packet(packet)

    assert "## 证据边界" in text
    assert "证据强度：" in text
    assert "争议点：" in text
    assert "证据缺口：" in text
    assert "evidence_strength:" not in text
    assert "controversy:" not in text
    assert "evidence_gap:" not in text
    assert "文献综述" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_decision_final_report_with_evidence_boundary_does_not_emit_raw_metadata():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": _decision_future_query(),
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "research_evidence_packet_context": (
            "research_packet_path: /tmp/research_evidence_packet.md\n"
            "research_packet_excerpt:\n"
            "artifact_path: raw/path\nexecutor_model: raw model\nvalid_for_pipeline: true\n"
        ),
    }
    text = executors._final_controller_report_from_packet(packet)

    assert "artifact_path:" not in text
    assert "executor_model:" not in text
    assert "valid_for_pipeline:" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_research_decision_intelligence_uses_gemini_high_and_stops_before_stage8(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                assert model == GEMINI_HIGH
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                assert model == GEMINI_PRO_HIGH
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return _complete_research_evidence_packet_text()
            assert stage.stage_name == "intelligence_layer"
            assert stage.owner == GEMINI_HIGH
            assert stage.model == GEMINI_HIGH
            assert model == GEMINI_HIGH
            assert "research_evidence_packet.md" in prompt
            assert ADHD_PROMPT in prompt
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "intelligence mapping only"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_decision_l1_l7_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages] == [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
    ]
    intelligence = stages[-1]
    assert intelligence["owner"] == GEMINI_HIGH
    assert intelligence["model"] == GEMINI_HIGH
    assert intelligence["executor_model"] == GEMINI_HIGH
    assert intelligence["artifact_path"].endswith("intelligence_layer/intelligence_layer_report.md")
    assert Path(intelligence["artifact_path"]).read_text(encoding="utf-8") == "intelligence mapping only"
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:supplementary_search" in error for error in result["full_pipeline_validation"]["errors"])


def test_intelligence_layer_does_not_run_without_accepted_l5(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "verdict: REJECTED\naccepted: false"
            return "should not run"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    l1_l5 = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_intelligence_smoke(
        l1_l5["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "intelligence_layer"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "accepted research packet validation failed" in result["run"]["stages"][-1]["error"]


def test_intelligence_layer_artifact_missing_blocks_validation(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "intelligence mapping only"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_decision_l1_l7_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    Path(result["run"]["stages"][-1]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH_DECISION, result["run"], base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("intelligence_layer:artifact_path_not_found" in error for error in validation["errors"])


def test_intelligence_layer_output_forbidden_final_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "final_controller_report\n最终建议"

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    l1_l5 = run_research_l1_l5_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_intelligence_smoke(
        l1_l5["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "intelligence_layer"
    assert "forbidden final-output tokens" in result["run"]["stages"][-1]["error"]


def test_intelligence_layer_constraint_echo_does_not_block(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nDo not produce final report.\nThis is not a final recommendation."

        def run_ddgs(self, stage, queries):
            return [{"title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_decision_l1_l7_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["blocked_stage"] if "blocked_stage" in result else "" == ""
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"


def test_intelligence_layer_structural_final_report_blocks():
    import tools.task_engine_executors as executors

    assert "final_report_heading" in executors._intelligence_output_forbidden_tokens("# Final Controller Report\nbody")
    assert "pipeline_status=PIPELINE_COMPLETE" in executors._intelligence_output_forbidden_tokens(
        "pipeline_status=PIPELINE_COMPLETE"
    )
    assert "action_plan_recommendation_direct_advice" in executors._intelligence_output_forbidden_tokens(
        "# Action Plan\n## Recommendation\nYou should start this plan tomorrow."
    )
    assert "chinese_final_advice_heading" in executors._intelligence_output_forbidden_tokens(
        "## 行动建议\n建议你从今天开始执行。"
    )


def test_agy_alias_log_allows_early_ccpa_noise_when_override_succeeds():
    import tools.task_engine_executors as executors

    output = "\n".join(
        [
            "Model ID Gemini 3.5 Flash (High) not in local config, defaulting to CCPA",
            "Resolving model Gemini 3.5 Flash (High)",
            'Propagating selected model override to backend: label="Gemini 3.5 Flash (High)"',
            "Print mode: silent auth succeeded",
        ]
    )

    assert executors._agy_model_alias_failed(output, GEMINI_HIGH) is False


def test_agy_alias_log_blocks_ccpa_without_override():
    import tools.task_engine_executors as executors

    output = "Model ID Gemini 3.5 Flash (High) not in local config, defaulting to CCPA"

    assert executors._agy_model_alias_failed(output, GEMINI_HIGH) is True


def test_agy_preflight_success_lists_required_models(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "Gemini 3.5 Flash (High)\nGemini 3.1 Pro (High)\nOther Model\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight(timeout_s=12)

    assert result["status"] == "AGY_OK"
    assert result["blocked_stage"] == ""
    assert result["blocked_reason"] == ""
    assert result["command"] == ["/opt/homebrew/bin/agy", "models"]
    assert Path(calls[0][1]["cwd"]).is_absolute()
    assert ".hermes" not in Path(calls[0][1]["cwd"]).parts
    assert result["agy_cwd"] == calls[0][1]["cwd"]
    assert result["gemini_dir_absolute"] is None
    assert result["required_models"] == [GEMINI_HIGH, GEMINI_PRO_HIGH]
    assert result["missing_models"] == []
    assert GEMINI_HIGH in result["models"]
    assert GEMINI_PRO_HIGH in result["models"]
    assert calls[0][1]["timeout"] == 12


def test_agy_preflight_blocks_auth_required(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "You are not logged into Antigravity. Please login in your browser.",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_stage"] == "agy_preflight"
    assert result["blocked_reason"] == executors.REQUIRES_MANUAL_AGY_LOGIN
    assert result["auth_refresh"]["bare_agy"]["command"] == ["/opt/homebrew/bin/agy"]
    assert "not logged into Antigravity" in result["auth_refresh"]["print_mode_sentinel"]["stderr_tail"]
    assert result["authorization_code_note"] == "authorization code must be entered by user manually"
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]


def test_agy_preflight_blocks_authorization_code_required(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stdout": "Open the browser and enter authorization code ABC123",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_stage"] == "agy_preflight"
    assert result["blocked_reason"] == executors.REQUIRES_MANUAL_AGY_LOGIN
    assert "authorization code" in result["auth_refresh"]["bare_agy"]["stdout_tail"]
    assert result["authorization_code_note"] == "authorization code must be entered by user manually"
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]


def test_agy_preflight_blocks_silent_auth_timeout(monkeypatch):
    import tools.task_engine_executors as executors

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "Print mode: not authenticated, trying silent auth\nPrint mode: timed out after 1195 polls",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_stage"] == "agy_preflight"
    assert result["blocked_reason"] == executors.REQUIRES_MANUAL_AGY_LOGIN
    assert calls[0] == ["/opt/homebrew/bin/agy", "models"]
    assert calls[1] == ["/opt/homebrew/bin/agy"]


def test_agy_preflight_blocks_missing_required_models(monkeypatch):
    import tools.task_engine_executors as executors

    def fake_run(command, **kwargs):
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "Gemini 3.5 Flash (High)\nCCPA\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(executors.shutil, "which", lambda name: "/opt/homebrew/bin/agy")
    monkeypatch.setattr(executors.subprocess, "run", fake_run)

    result = run_agy_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_stage"] == "agy_preflight"
    assert result["blocked_reason"] == "AGY_MODEL_LIST_MISSING_REQUIRED"
    assert result["missing_models"] == [GEMINI_PRO_HIGH]


def test_stress_runner_preflight_fail_does_not_call_l1(monkeypatch, tmp_path: Path):
    calls = []

    def fake_runner_json(action, base_dir):
        calls.append(action)
        if action == "dry-run":
            return {"status": "ok", "plan": {"stage_count": 16}}
        if action == "simulated-run":
            return {
                "status": "ok",
                "pipeline_status": PIPELINE_COMPLETE,
                "validation": {"valid": True, "stage_count": 16},
            }
        if action == "agy-preflight":
            return {
                "status": "BLOCKED_STATUS",
                "blocked_stage": "agy_preflight",
                "blocked_reason": "AGY_AUTH_REQUIRES_USER",
            }
        if action == "smoke-research-l1-l2":
            raise AssertionError("L1 must not run when AGY preflight blocks")
        raise AssertionError(f"unexpected action: {action}")

    monkeypatch.setattr(stress_runner, "_run_pytest", lambda: {"passed": True, "returncode": 0, "stdout": "", "stderr": ""})
    monkeypatch.setattr(stress_runner, "_runner_json", fake_runner_json)
    monkeypatch.setattr(stress_runner, "_git_diff_summary", lambda: "")
    monkeypatch.setattr(stress_runner.sys, "argv", ["stress", "--max-rounds", "1", "--out", str(tmp_path)])

    assert stress_runner.main() == 0

    summary = json.loads((tmp_path / "round_01" / "summary.json").read_text(encoding="utf-8"))
    assert calls == ["dry-run", "simulated-run", "agy-preflight"]
    assert summary["blocked_stage"] == "agy_preflight"
    assert summary["blocked_reason"] == "AGY_AUTH_REQUIRES_USER"
    assert summary["agy_preflight"]["status"] == "BLOCKED_STATUS"
    assert summary["smoke_l1_l2"]["status"] == "skipped"
    assert not (tmp_path / "round_01" / "real_l1_l2").exists()


def test_stress_runner_preflight_success_permits_l1(monkeypatch, tmp_path: Path):
    calls = []

    def fake_runner_json(action, base_dir):
        calls.append(action)
        if action == "dry-run":
            return {"status": "ok", "plan": {"stage_count": 16}}
        if action == "simulated-run":
            return {
                "status": "ok",
                "pipeline_status": PIPELINE_COMPLETE,
                "validation": {"valid": True, "stage_count": 16},
            }
        if action == "agy-preflight":
            return {"status": "AGY_OK", "blocked_stage": "", "blocked_reason": ""}
        if action == "smoke-research-l1-l2":
            return {
                "status": "blocked",
                "pipeline_status": PIPELINE_BLOCKED,
                "blocked_stage": "L1_gemini_search",
                "error": "forced stop after verifying L1 was called",
            }
        raise AssertionError(f"unexpected action: {action}")

    monkeypatch.setattr(stress_runner, "_run_pytest", lambda: {"passed": True, "returncode": 0, "stdout": "", "stderr": ""})
    monkeypatch.setattr(stress_runner, "_runner_json", fake_runner_json)
    monkeypatch.setattr(stress_runner, "_git_diff_summary", lambda: "")
    monkeypatch.setattr(stress_runner.sys, "argv", ["stress", "--max-rounds", "1", "--out", str(tmp_path)])

    assert stress_runner.main() == 0

    summary = json.loads((tmp_path / "round_01" / "summary.json").read_text(encoding="utf-8"))
    assert calls == ["dry-run", "simulated-run", "agy-preflight", "smoke-research-l1-l2"]
    assert summary["agy_preflight"]["status"] == "AGY_OK"
    assert summary["blocked_stage"] == "L1_gemini_search"
    assert summary["blocked_reason"] == "forced stop after verifying L1 was called"


def test_omlx_preflight_blocks_when_key_missing(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(executors, "_decision_engine_api_config", lambda: {"api_key_env": "OMLX_API_KEY", "base_url": "http://127.0.0.1:8000/v1"})

    result = run_omlx_preflight()

    assert result["status"] == "BLOCKED_STATUS"
    assert result["blocked_stage"] == "omlx_preflight"
    assert result["blocked_reason"] == "OMLX_API_KEY_MISSING"
    assert result["key_fingerprint"] == {"present": False, "length": 0, "sha256_12": ""}


def test_omlx_preflight_loads_key_from_hermes_env_without_leaking(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    secret = "test-omlx-secret-value"
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    (hermes_dir / ".env").write_text(f"OMLX_API_KEY={secret}\n", encoding="utf-8")

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            self.base_url = base_url
            self.api_key = api_key

        def login(self):
            assert self.api_key == secret
            return True

        def get_models(self):
            return [{"id": R1_ACTUAL_MODEL_DEFAULT}]

    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(executors, "_decision_engine_api_config", lambda: {"api_key_env": "OMLX_API_KEY", "base_url": "http://127.0.0.1:8000/v1"})
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)

    result = run_omlx_preflight()
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["status"] == "OMLX_OK"
    assert result["blocked_stage"] == ""
    assert result["key_source"] == str(hermes_dir / ".env")
    assert result["key_fingerprint"]["present"] is True
    assert result["key_fingerprint"]["length"] == len(secret)
    assert result["model_visible"] is True
    assert secret not in serialized
    assert "OMLX_API_KEY" in result["configured_key_env"]
    assert os.environ["OMLX_API_KEY"] == secret


def test_task_engine_runner_omlx_preflight_uses_same_config_source(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    secret = "runner-env-file-secret"
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    (hermes_dir / ".env").write_text(f"OMLX_API_KEY={secret}\n", encoding="utf-8")

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            self.api_key = api_key

        def login(self):
            return self.api_key == secret

        def get_models(self):
            return [{"id": R1_ACTUAL_MODEL_DEFAULT}]

    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(executors, "_decision_engine_api_config", lambda: {"api_key_env": "OMLX_API_KEY", "base_url": "http://127.0.0.1:8000/v1"})
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)

    result = json.loads(task_engine_runner(query=ADHD_PROMPT, mode=ENGINE_RESEARCH_DECISION, action="omlx-preflight"))

    assert result["status"] == "OMLX_OK"
    assert result["key_source"] == str(hermes_dir / ".env")
    assert secret not in json.dumps(result, ensure_ascii=False)


def test_l3_omlx_auth_fail_does_not_fallback_to_other_models(monkeypatch):
    import tools.task_engine_executors as executors

    class FakeAdmin:
        def __init__(self, base_url, api_key):
            self.api_key = api_key

        def login(self):
            return False

    monkeypatch.setenv("OMLX_API_KEY", "bad-secret")
    monkeypatch.setattr(executors, "_OmlxAdmin", FakeAdmin)
    executor = LocalTaskEngineExecutor()
    stage = StageSpec("L3_r1_synthesis", R1_32B, R1_32B, ("r1_synthesis.md",))

    try:
        executor.run_omlx_model(stage, R1_32B, "prompt")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("L3 must fail closed when OMLX admin auth fails")

    assert "OMLX_AUTH_BLOCKED" in message
    assert executor.last_executor_models["L3_r1_synthesis"] == R1_ACTUAL_MODEL_DEFAULT
    assert "Controller" not in message
    assert "Flash" not in message
    assert "Qwen72B" not in message


def test_supplementary_search_uses_ddgs_and_stops_before_structure_mapper(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {
                    "source_candidates": [
                        {
                            "candidate": "CDC ADHD parent training behavior therapy guidance",
                            "evidence_type": "Guideline",
                            "coverage_axis": "authoritative_guideline_or_consensus",
                            "why_relevant": "ADHD parent training and behavior therapy guidance is relevant to child intervention decisions.",
                        },
                        {
                            "candidate": "AAP ADHD clinical practice guideline school supports",
                            "evidence_type": "Guideline",
                            "coverage_axis": "intervention_or_practice",
                            "why_relevant": "ADHD school supports and clinical guidance inform accommodations and intervention intensity.",
                        },
                        {
                            "candidate": "CLAS ADHD inattentive children parent training organization skills",
                            "evidence_type": "Intervention Study",
                            "coverage_axis": "empirical_study_or_RCT",
                            "why_relevant": "ADHD inattentive children may benefit from parent training and organization skills interventions.",
                        },
                        {
                            "candidate": "ADHD executive function children mind wandering cognitive disengagement",
                            "evidence_type": "Mechanism Review",
                            "coverage_axis": "mechanism_or_theory",
                            "why_relevant": "ADHD executive function and mind wandering mechanisms shape long-term support needs.",
                        },
                    ]
                }
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map\ndecision_dimensions_for_later_stages\nopen_items_for_stage8"

        def run_ddgs(self, stage, queries):
            if stage.stage_name == "L2_ddgs_supplement":
                return [
                    {
                        "query": queries[0],
                        "title": "ADHD parent training evidence",
                        "url": "https://example.test/ddgs",
                        "snippet": "ADHD parent training evidence supports behavior management and school-home coordination.",
                    }
                ]
            assert stage.stage_name == "supplementary_search"
            assert stage.owner == "DDGS"
            assert stage.model == "DDGS"
            assert queries[:2] == [
                "ADHD parent training children",
                "behavioral parent training ADHD inattentive children",
            ]
            return [
                {
                    "query": queries[0],
                    "title": "CDC parent training",
                    "url": "https://example.test/parent-training",
                    "snippet": "Parent training evidence.",
                }
            ]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_decision_l1_l8_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["intelligence_layer", "supplementary_search"]
    supplementary = stages[-1]
    assert supplementary["owner"] == "DDGS"
    assert supplementary["model"] == "DDGS"
    assert supplementary["executor_model"] == "DDGS"
    assert supplementary["artifact_path"].endswith("supplementary_search/parent_training_supplement.md")
    report = Path(supplementary["artifact_path"]).read_text(encoding="utf-8")
    assert "https://example.test/parent-training" in report
    assert "final_controller_report" not in report
    assert "# Final Controller Report" not in report
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:structure_mapper" in error for error in result["full_pipeline_validation"]["errors"])


def test_supplementary_search_blocks_web_search_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_DDGS_BACKENDS", "duckduckgo,generic_web_search")
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][7]

    try:
        LocalTaskEngineExecutor().run_ddgs(stage, ["ADHD parent training children"])
    except RuntimeError as exc:
        assert "cannot include web_search fallback" in str(exc)
    else:
        raise AssertionError("supplementary_search must not accept generic web_search fallback")


def test_supplementary_search_blocks_without_intelligence_layer(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_ddgs(self, stage, queries):
            return [{"title": "should not run", "url": "https://example.test"}]

    result = run_research_decision_supplementary_search_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "supplementary_search"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L5 plus intelligence_layer" in result["run"]["stages"][-1]["error"]


def test_supplementary_search_no_fresh_result_writes_controlled_caveat_artifact(tmp_path: Path):
    import tools.task_engine_executors as executors

    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            if stage.stage_name == "L2_ddgs_supplement":
                return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]
            return [
                executors._supplementary_search_no_fresh_hits_marker(
                    queries,
                    ["duckduckgo", "brave", "yahoo"],
                    ["duckduckgo:empty", "brave:DDGSException:No results found."],
                )
            ]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    l1_l7 = run_research_decision_l1_l7_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_supplementary_search_smoke(
        l1_l7["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "ok"
    assert result["pipeline_status"] == PIPELINE_INCOMPLETE
    artifact = Path(result["run"]["stages"][-1]["artifact_path"])
    text = artifact.read_text(encoding="utf-8")
    assert "supplementary_search_status: no_fresh_hits" in text
    assert "source_evidence_emitted: false" in text
    assert "No topic-consistent fresh DDGS hits remained" in text
    assert "https://" not in text


def test_supplementary_search_artifact_missing_blocks_validation(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "R1 synthesis body"

    result = run_research_decision_l1_l8_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    Path(result["run"]["stages"][-1]["artifact_path"]).unlink()

    validation = validate_pipeline(ENGINE_RESEARCH_DECISION, result["run"], base_dir=tmp_path)

    assert validation["valid"] is False
    assert any("supplementary_search:artifact_path_not_found" in error for error in validation["errors"])


def test_structure_mapper_uses_qwen72b_and_stops_before_evidence_judge(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map\ndecision_dimensions_for_later_stages"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            assert stage.stage_name == "structure_mapper"
            assert stage.owner == QWEN72B
            assert stage.model == QWEN72B
            assert model == QWEN72B
            assert "parent_training_supplement.md" in prompt
            self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
            return "problem_axes\nactor_map\ndecision_questions\nevidence_slots\nunknowns_for_later_stages"

    result = run_research_decision_l1_l9_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["supplementary_search", "structure_mapper"]
    structure = stages[-1]
    assert structure["owner"] == QWEN72B
    assert structure["model"] == QWEN72B
    assert structure["executor_model"] == QWEN72B_ACTUAL_MODEL_DEFAULT
    assert structure["artifact_path"].endswith("structure_mapper/structure_mapper.md")
    report = Path(structure["artifact_path"]).read_text(encoding="utf-8")
    assert "problem_axes" in report
    assert "final_controller_report" not in report
    assert "convergence" not in report.lower()
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:evidence_judge" in error for error in result["full_pipeline_validation"]["errors"])


def test_structure_mapper_blocks_without_supplementary_search(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("structure_mapper must not run without supplementary_search")

    result = run_research_decision_structure_mapper_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "structure_mapper"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L8 stages" in result["run"]["stages"][-1]["error"]


def test_structure_mapper_output_forbidden_final_or_later_stage_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
            return "# Final Controller Report\nconvergence_report\nfinal_controller_report"

    l1_l8 = run_research_decision_l1_l8_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_structure_mapper_smoke(
        l1_l8["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "structure_mapper"
    error = result["run"]["stages"][-1]["error"]
    assert "forbidden later-stage/final tokens" in error
    assert "debug_artifact=" in error
    assert (tmp_path / "structure_mapper" / "structure_mapper.invalid.md").exists()


def test_structure_mapper_allows_later_convergence_handoff_language():
    import tools.task_engine_executors as executors

    allowed = """problem_axes
- Map the attention problem into task-initiation, working-memory, and verification axes.
unknowns_for_later_stages
- for later convergence, compare low-intensity scaffolding against school-first support.
- convergence should consider whether internal mind-wandering is a risk amplifier.
- 后续收敛阶段应考虑家庭执行成本。
- 供收敛阶段参考：把柔术反馈回路作为保护因素。
"""

    assert executors._structure_mapper_forbidden_tokens(allowed) == []


def test_structure_mapper_blocks_structural_convergence_heading():
    import tools.task_engine_executors as executors

    assert "later_stage_heading" in executors._structure_mapper_forbidden_tokens("## convergence_report\nThis is a later stage body.")
    assert "later_stage_heading" in executors._structure_mapper_forbidden_tokens("# evidence_judge\nThis is a later stage body.")


def test_structure_mapper_blocks_pipeline_complete_marker():
    import tools.task_engine_executors as executors

    assert "pipeline_status=PIPELINE_COMPLETE" in executors._structure_mapper_forbidden_tokens(
        "problem_axes\npipeline_status=PIPELINE_COMPLETE"
    )


def test_structure_mapper_blocks_final_conclusion_heading():
    import tools.task_engine_executors as executors

    assert "final_conclusion_heading" in executors._structure_mapper_forbidden_tokens("## 最终结论\n用户应该...")


def test_evidence_judge_prefill_block_writes_diagnostic(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = []
    for spec in CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][:9]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        outputs = executors.planned_outputs(spec, tmp_path)
        for output in outputs.values():
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            body = "fresh source"
            if spec.stage_name == "L5_deepseek_acceptance":
                body = _complete_research_evidence_packet_text()
            output_path.write_text(body, encoding="utf-8")
        artifact = executors._primary_output_path(spec, outputs, stage_dir)
        stages.append({
            "stage_name": spec.stage_name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact),
            "outputs": outputs,
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
            "status": "accepted" if spec.stage_name == "L5_deepseek_acceptance" else "real",
        })

    class AlwaysPrefillExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            self.last_omlx_diagnostics[stage.stage_name] = {
                "stage_name": stage.stage_name,
                "actual_model": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "empty_content_kind": "response_error_object",
                "error_summary": "prefill_memory_exceeded",
                "prompt_chars": len(prompt),
                "prompt_estimated_tokens": max(1, (len(prompt) + 3) // 4),
            }
            raise RuntimeError("prefill_memory_exceeded")

    result = executors.run_research_decision_evidence_judge_smoke(
        {"stages": stages},
        query="未来10年 AI 结构性反转 ADHD",
        base_dir=tmp_path,
        executor=AlwaysPrefillExecutor(),
    )

    diagnostic_path = tmp_path / "evidence_judge" / "evidence_judge.diagnostic.json"
    compact_path = tmp_path / "evidence_judge" / "evidence_judge.compact_retry.diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    assert diagnostic_path.exists()
    assert compact_path.exists()
    assert diagnostic["error_summary"] == "prefill_memory_exceeded"
    assert diagnostic.get("compact_mode_used") is not True
    assert compact["compact_mode_used"] is True
    assert compact["blocked_reason_original"] == "OMLX_PREFILL_MEMORY_GUARD_BLOCKED"
    assert compact["original_prompt_chars"] > compact["compact_prompt_chars"]
    assert "diagnostic_artifact=" in result["run"]["stages"][-1]["error"]
    assert "compact_retry_diagnostic_artifact=" in result["run"]["stages"][-1]["error"]


def test_evidence_judge_prefill_compact_retry_succeeds_and_writes_diagnostic(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = []
    for spec in CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][:9]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        outputs = executors.planned_outputs(spec, tmp_path)
        for output in outputs.values():
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if spec.stage_name == "L5_deepseek_acceptance":
                body = _complete_research_evidence_packet_text()
            elif spec.stage_name == "intelligence_layer":
                body = "claim strength uncertainty applicability evidence gap " * 80
            elif spec.stage_name == "supplementary_search":
                body = "evidence support risk uncertainty applicability " * 80
            elif spec.stage_name == "structure_mapper":
                body = "problem_axes\nclaim strength and applicability axes\nunknowns_for_later_stages\nuncertainty gaps"
            else:
                body = "fresh source"
            output_path.write_text(body, encoding="utf-8")
        artifact = executors._primary_output_path(spec, outputs, stage_dir)
        stages.append({
            "stage_name": spec.stage_name,
            "owner": spec.owner,
            "model": spec.model,
            "executor_model": spec.model,
            "artifact_path": str(artifact),
            "outputs": outputs,
            "created_in_current_run": True,
            "legacy_contaminated": False,
            "valid_for_pipeline": True,
            "status": "accepted" if spec.stage_name == "L5_deepseek_acceptance" else "real",
        })

    class PrefillThenCompactExecutor(LocalTaskEngineExecutor):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def run_omlx_model(self, stage, model, prompt):
            self.prompts.append(prompt)
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            self.last_omlx_diagnostics[stage.stage_name] = {
                "stage_name": stage.stage_name,
                "actual_model": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "prompt_chars": len(prompt),
                "prompt_estimated_tokens": max(1, (len(prompt) + 3) // 4),
                "compact_mode_used": "compact_evidence_judge_packet" in prompt,
                "inference_request_sent": "compact_evidence_judge_packet" in prompt,
                "inference_response_received": "compact_evidence_judge_packet" in prompt,
            }
            if "compact_evidence_judge_packet" not in prompt:
                self.last_omlx_diagnostics[stage.stage_name]["error_summary"] = "prefill_memory_exceeded"
                raise RuntimeError("prefill_memory_exceeded")
            return "\n".join(
                [
                    "evidence_quality_map",
                    "The compact retry directly judges evidence quality.",
                    "strength_by_claim",
                    "The stronger claims are task-cost and current evidence claims.",
                    "applicability_to_user_context",
                    "Applicability is bounded and context-dependent.",
                    "uncertainty_and_limits",
                    "Uncertainty remains for long-horizon claims.",
                    "evidence_gaps_for_later_stages",
                    "Direct longitudinal evidence is missing.",
                ]
            )

    executor = PrefillThenCompactExecutor()
    result = executors.run_research_decision_evidence_judge_smoke(
        {"stages": stages},
        query="未来10年 AI 结构性反转",
        base_dir=tmp_path,
        executor=executor,
    )

    diagnostic_path = tmp_path / "evidence_judge" / "evidence_judge.diagnostic.json"
    compact_path = tmp_path / "evidence_judge" / "evidence_judge.compact_retry.diagnostic.json"
    artifact = tmp_path / "evidence_judge" / "evidence_judge.md"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8").startswith("evidence_quality_map")
    assert len(executor.prompts) == 2
    assert "compact_evidence_judge_packet" not in executor.prompts[0]
    assert "compact_evidence_judge_packet" in executor.prompts[1]
    assert "Your first non-empty line must be exactly: evidence_quality_map" in executor.prompts[1]
    assert "Do not write phrases like" in executor.prompts[1]
    assert len(executor.prompts[1]) <= 10000
    assert diagnostic["error_summary"] == "prefill_memory_exceeded"
    assert compact["compact_mode_used"] is True
    assert compact["blocked_reason_original"] == "OMLX_PREFILL_MEMORY_GUARD_BLOCKED"
    assert compact["original_prompt_chars"] > compact["compact_prompt_chars"]


def test_evidence_judge_uses_nemotron_and_stops_before_premise_auditor(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            assert stage.stage_name == "evidence_judge"
            assert stage.owner == NEMOTRON120B
            assert stage.model == NEMOTRON120B
            assert model == NEMOTRON120B
            assert "structure_mapper.md" in prompt
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"

    result = run_research_decision_l1_l10_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["structure_mapper", "evidence_judge"]
    evidence = stages[-1]
    assert evidence["owner"] == NEMOTRON120B
    assert evidence["model"] == NEMOTRON120B
    assert evidence["executor_model"] == NEMOTRON120B_ACTUAL_MODEL_DEFAULT
    assert evidence["artifact_path"].endswith("evidence_judge/evidence_judge.md")
    report = Path(evidence["artifact_path"]).read_text(encoding="utf-8")
    assert "evidence_quality_map" in report
    assert "final_controller_report" not in report
    assert "convergence_report" not in report.lower()
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:premise_auditor" in error for error in result["full_pipeline_validation"]["errors"])


def test_evidence_judge_blocks_without_structure_mapper(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("evidence_judge must not run without structure_mapper")

    result = run_research_decision_evidence_judge_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L9 stages" in result["run"]["stages"][-1]["error"]


def test_evidence_judge_output_forbidden_final_or_later_stage_terms_blocked(tmp_path: Path):
    calls = []

    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            calls.append(prompt)
            return "# Final Controller Report\nconvergence_report\nfinal_controller_report"

    l1_l9 = run_research_decision_l1_l9_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_evidence_judge_smoke(
        l1_l9["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    error = result["run"]["stages"][-1]["error"]
    assert "forbidden later-stage/final tokens" in error
    assert "debug_artifact=" in error
    assert len(calls) == 1
    assert (tmp_path / "evidence_judge" / "evidence_judge.invalid.md").exists()


def test_evidence_judge_section_start_mismatch_writes_invalid_artifact_and_diagnostic(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            self.last_omlx_diagnostics[stage.stage_name] = {
                "inference_request_sent": True,
                "inference_response_received": True,
                "compact_mode_used": False,
            }
            return "strength_by_claim\n- claim strength without required first section"

    l1_l9 = run_research_decision_l1_l9_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_evidence_judge_smoke(
        l1_l9["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    invalid_path = tmp_path / "evidence_judge" / "evidence_judge.invalid.md"
    diagnostic_path = tmp_path / "evidence_judge" / "evidence_judge.diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    assert "artifact_quality_error:schema_retry_failed:section_start_mismatch" in result["run"]["stages"][-1]["error"]
    assert invalid_path.exists()
    assert (tmp_path / "evidence_judge" / "evidence_judge.schema_retry_source.invalid.md").exists()
    assert not (tmp_path / "evidence_judge" / "evidence_judge.md").exists()
    assert diagnostic["artifact_quality_error"] == "schema_retry_failed:section_start_mismatch"
    assert diagnostic["first_nonempty_line"] == "strength_by_claim"
    assert diagnostic["normalized_first_line"] == "strength_by_claim"
    assert diagnostic["final_content_chars"] == len(invalid_path.read_text(encoding="utf-8"))
    assert diagnostic["invalid_artifact_path"] == str(invalid_path)
    assert diagnostic["inference_request_sent"] is True
    assert diagnostic["inference_response_received"] is True
    assert diagnostic["compact_mode_used"] is True
    assert diagnostic["prompt_chars"] > 0
    assert diagnostic["prompt_estimated_tokens"] > 0
    assert diagnostic["valid_for_pipeline"] is False


def test_evidence_judge_allows_handoff_to_later_stage_language():
    import tools.task_engine_executors as executors

    allowed = """evidence_quality_map
- Claim A is supported.
- convergence should consider uncertainty boundaries.
- This issue should be revisited in convergence after premise and alternative stages.
"""

    assert executors._evidence_judge_forbidden_tokens(allowed) == []


def test_evidence_judge_artifact_quality_blocks_process_narration_leak():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    polluted = """We need to output sections: evidence_quality_map, strength_by_claim.

We must judge the supplied artifacts and then craft the answer.

evidence_quality_map
The evidence is moderate.
"""

    try:
        executors._assert_artifact_quality(stage, polluted)
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("evidence_judge process narration must fail closed")

    assert "evidence_judge: artifact_quality_error:process_narration_leak" in error


def test_evidence_judge_prompt_derives_claims_when_claim_table_missing(tmp_path: Path):
    import tools.task_engine_executors as executors

    records = []
    stage_names = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
    ]
    for name in stage_names:
        stage_dir = tmp_path / name
        stage_dir.mkdir()
        artifact = stage_dir / ("parent_training_supplement.md" if name == "supplementary_search" else f"{name}.md")
        artifact.write_text(f"{name} domain material without claim table", encoding="utf-8")
        records.append({"stage_name": name, "artifact_path": str(artifact), "outputs": {}})

    prompt = executors._evidence_judge_prompt_from_artifacts(
        records,
        query="B2B SaaS PostgreSQL lakehouse migration?",
        base_dir=tmp_path,
    )

    assert "derive 4-6 decision-relevant claims" in prompt
    assert "strength_by_claim is always required" in prompt
    assert "Never write \"not applicable\" for evidence_quality_map or strength_by_claim" in prompt


def test_decision_evidence_judge_prompt_includes_schema_contract(tmp_path: Path):
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][3]
    prompt = executors._decision_stage_prompt(stage, [], query="Should we invest?", base_dir=tmp_path)

    assert "Your first non-empty line must be exactly: evidence_quality_map" in prompt
    assert "Write strength_by_claim as a standalone line exactly: strength_by_claim" in prompt
    assert "Do not write a report title" in prompt


def test_evidence_judge_not_applicable_sections_continue_to_fail_closed():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    invalid = "evidence_quality_map and strength_by_claim are not applicable as the current evidence packet lacks granular claim-level assessment."

    try:
        executors._assert_artifact_quality(stage, invalid)
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("not-applicable evidence_judge output must fail closed")

    assert "evidence_judge: artifact_quality_error:section_start_mismatch" in error


def test_evidence_judge_schema_failure_retries_with_compact_prompt(tmp_path: Path):
    calls = []

    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            calls.append(prompt)
            if len(calls) == 1:
                return "# evidence_judge Report\n\nEvidence quality is mixed but usable."
            return """evidence_quality_map
Evidence is mixed but usable.

strength_by_claim
- claim: trend evidence is uncertain
  strength: medium
  evidence_basis: research packet and structure mapper
  uncertainty_or_gap: direct retention evidence remains missing

applicability_to_user_context
Applicable with explicit uncertainty boundaries.

uncertainty_and_limits
Small-sample observations remain weak.

evidence_gaps_for_later_stages
Need direct retention and activation data.
"""

    l1_l9 = run_research_decision_l1_l9_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_evidence_judge_smoke(
        l1_l9["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert (tmp_path / "evidence_judge" / "evidence_judge.schema_retry_source.invalid.md").exists()
    assert (tmp_path / "evidence_judge" / "evidence_judge.md").exists()


def test_valid_evidence_judge_requires_independent_first_line_and_strength_section():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    valid = """evidence_quality_map
Evidence is mixed but usable.

strength_by_claim
- claim: migrate read-heavy analytics away from OLTP
  strength: medium
  evidence_basis: CDC and lakehouse materials
  uncertainty_or_gap: cost and migration risk remain context-specific

applicability_to_user_context
Applicable to B2B SaaS analytics.

uncertainty_and_limits
Migration timing remains uncertain.

evidence_gaps_for_later_stages
Need workload and cost model.
"""
    bad_first_line = "evidence_quality_map: summary\nstrength_by_claim\n- claim: x"
    missing_strength = "evidence_quality_map\nEvidence only."

    executors._assert_artifact_quality(stage, valid)
    for invalid in (bad_first_line, missing_strength):
        try:
            executors._assert_artifact_quality(stage, invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid evidence_judge structure must fail closed")


def test_evidence_judge_artifact_quality_allows_clean_ai_will_phrase():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    clean = """evidence_quality_map
Evidence quality is moderate. AI will continuously reduce routine legal task costs is a claim to be judged, not a self-instruction.

strength_by_claim
The task-cost reduction claim is stronger than the structural reversal claim.

applicability_to_user_context
Applicability is moderate for junior lawyers and corporate legal analysts.

uncertainty_and_limits
The ten-year horizon remains uncertain.

evidence_gaps_for_later_stages
Longitudinal role-level evidence is missing.
"""

    executors._assert_artifact_quality(stage, clean)


def test_evidence_judge_allows_premise_audit_referral_language():
    import tools.task_engine_executors as executors

    allowed = """applicability_assessment
- Claim B is plausible but requires premise audit.
- handoff to premise_auditor: verify school-system assumptions.
- next stage should consider whether the premise holds under different classroom conditions.
"""

    assert executors._evidence_judge_forbidden_tokens(allowed) == []


def test_evidence_judge_blocks_structural_later_stage_heading():
    import tools.task_engine_executors as executors

    assert "later_stage_heading" in executors._evidence_judge_forbidden_tokens("## convergence_report\nThis is a later stage body.")
    assert "later_stage_heading" in executors._evidence_judge_forbidden_tokens("# premise_auditor\nThis is a later stage body.")


def test_evidence_judge_blocks_final_controller_marker():
    import tools.task_engine_executors as executors

    assert "final_controller_report" in executors._evidence_judge_forbidden_tokens("evidence_quality_map\nfinal_controller_report")


def test_divergence_gates_allow_convergence_handoff_language():
    import tools.task_engine_executors as executors

    allowed = """stage body
- convergence should consider uncertainty boundaries.
- for later convergence, retain this as a low-confidence handoff.
- 后续收敛阶段应考虑执行功能与 AI 反馈质量。
- 供收敛阶段参考，不是最终结论。
"""

    assert executors._premise_auditor_forbidden_tokens(allowed) == []
    assert executors._alternative_generator_forbidden_tokens(allowed) == []
    assert executors._insight_harvester_forbidden_tokens(allowed) == []


def test_divergence_gates_block_structural_later_stage_headings():
    import tools.task_engine_executors as executors

    assert "later_stage_heading" in executors._premise_auditor_forbidden_tokens("## convergence_report\nbody")
    assert "later_stage_heading" in executors._alternative_generator_forbidden_tokens("## convergence_report\nbody")
    assert "later_stage_heading" in executors._insight_harvester_forbidden_tokens("## convergence_report\nbody")
    assert "pipeline_status=PIPELINE_COMPLETE" in executors._alternative_generator_forbidden_tokens(
        "option_tradeoffs\npipeline_status=PIPELINE_COMPLETE"
    )


def test_premise_auditor_uses_llama70b_and_stops_before_alternative_generator(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            assert stage.stage_name == "premise_auditor"
            assert stage.owner == LLAMA70B
            assert stage.model == LLAMA70B
            assert model == LLAMA70B
            assert "evidence_judge.md" in prompt
            self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
            return "implicit_premises\npremise_risks\ncounterexamples\nculture_and_school_system_differences"

    result = run_research_decision_l1_l11_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["evidence_judge", "premise_auditor"]
    premise = stages[-1]
    assert premise["owner"] == LLAMA70B
    assert premise["model"] == LLAMA70B
    assert premise["executor_model"] == LLAMA70B_ACTUAL_MODEL_DEFAULT
    assert premise["artifact_path"].endswith("premise_auditor/premise_auditor.md")
    report = Path(premise["artifact_path"]).read_text(encoding="utf-8")
    assert "implicit_premises" in report
    assert "final_controller_report" not in report
    assert "convergence_report" not in report.lower()
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:alternative_generator" in error for error in result["full_pipeline_validation"]["errors"])


def test_premise_auditor_blocks_without_evidence_judge(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("premise_auditor must not run without evidence_judge")

    result = run_research_decision_premise_auditor_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "premise_auditor"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L10 stages" in result["run"]["stages"][-1]["error"]


def test_premise_auditor_output_forbidden_final_or_later_stage_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
            return "# Final Controller Report\nconvergence_report\nfinal_controller_report"

    l1_l10 = run_research_decision_l1_l10_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_premise_auditor_smoke(
        l1_l10["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "premise_auditor"
    assert "forbidden later-stage/final tokens" in result["run"]["stages"][-1]["error"]
    assert "debug_artifact=" in result["run"]["stages"][-1]["error"]
    assert (tmp_path / "premise_auditor" / "premise_auditor.invalid.md").exists()


def test_alternative_generator_uses_gemma431b_and_stops_before_insight_harvester(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises\npremise_risks\ncounterexamples"
            assert stage.stage_name == "alternative_generator"
            assert stage.owner == GEMMA431B
            assert stage.model == GEMMA431B
            assert model == GEMMA431B
            assert "premise_auditor.md" in prompt
            self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
            return "mutually_exclusive_alternatives\nintervention_intensity_paths\nrisk_assumption_branches"

    result = run_research_decision_l1_l12_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["premise_auditor", "alternative_generator"]
    alternative = stages[-1]
    assert alternative["owner"] == GEMMA431B
    assert alternative["model"] == GEMMA431B
    assert alternative["executor_model"] == GEMMA431B_ACTUAL_MODEL_DEFAULT
    assert alternative["artifact_path"].endswith("alternative_generator/alternative_generator.md")
    report = Path(alternative["artifact_path"]).read_text(encoding="utf-8")
    assert "mutually_exclusive_alternatives" in report
    assert "final_controller_report" not in report
    assert "convergence_report" not in report.lower()
    assert "insight_harvester" not in report.lower()
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:insight_harvester" in error for error in result["full_pipeline_validation"]["errors"])


def test_alternative_generator_blocks_without_premise_auditor(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("alternative_generator must not run without premise_auditor")

    result = run_research_decision_alternative_generator_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "alternative_generator"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L11 stages" in result["run"]["stages"][-1]["error"]


def test_alternative_generator_output_forbidden_final_or_later_stage_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises\npremise_risks\ncounterexamples"
            self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
            return "# Final Controller Report\ninsight_harvester\nconvergence_report\nfinal_controller_report"

    l1_l11 = run_research_decision_l1_l11_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_alternative_generator_smoke(
        l1_l11["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "alternative_generator"
    assert "forbidden later-stage/final tokens" in result["run"]["stages"][-1]["error"]
    assert "debug_artifact=" in result["run"]["stages"][-1]["error"]
    assert (tmp_path / "alternative_generator" / "alternative_generator.invalid.md").exists()


def test_insight_harvester_uses_gemma431b_and_stops_before_convergence_report(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises\npremise_risks\ncounterexamples"
            if stage.stage_name == "alternative_generator":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "mutually_exclusive_alternatives\nintervention_intensity_paths\nrisk_assumption_branches"
            assert stage.stage_name == "insight_harvester"
            assert stage.owner == GEMMA431B
            assert stage.model == GEMMA431B
            assert model == GEMMA431B
            assert "alternative_generator.md" in prompt
            self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
            return "cross_model_insights\nconflicts_and_tensions\noutliers\ndecision_turning_points"

    result = run_research_decision_l1_l13_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["alternative_generator", "insight_harvester"]
    alternative = stages[-2]
    insight = stages[-1]
    assert insight["owner"] == GEMMA431B
    assert insight["model"] == GEMMA431B
    assert insight["executor_model"] == GEMMA431B_ACTUAL_MODEL_DEFAULT
    assert insight["artifact_path"].endswith("insight_harvester/insight_harvester.md")
    assert insight["artifact_path"] != alternative["artifact_path"]
    assert Path(insight["artifact_path"]).read_text(encoding="utf-8") != Path(alternative["artifact_path"]).read_text(encoding="utf-8")
    report = Path(insight["artifact_path"]).read_text(encoding="utf-8")
    assert "cross_model_insights" in report
    assert "final_controller_report" not in report
    assert "convergence_report" not in report.lower()
    assert "pipeline_status=PIPELINE_COMPLETE" not in report
    assert result["full_pipeline_validation"]["valid"] is False
    assert result["full_pipeline_validation"]["divergence_unique_model_count"] == 4
    assert any("missing_stage:convergence_report" in error for error in result["full_pipeline_validation"]["errors"])
    divergence_roles = {
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
    }
    by_name = {stage["stage_name"]: stage for stage in stages}
    assert all(by_name[role]["valid_for_pipeline"] is True for role in divergence_roles)
    assert len({by_name[role]["model"] for role in divergence_roles}) == 4


def test_insight_harvester_blocks_without_alternative_generator(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("insight_harvester must not run without alternative_generator")

    result = run_research_decision_insight_harvester_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "insight_harvester"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L12 stages" in result["run"]["stages"][-1]["error"]


def test_insight_harvester_output_forbidden_final_or_later_stage_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises\npremise_risks\ncounterexamples"
            if stage.stage_name == "alternative_generator":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "mutually_exclusive_alternatives\nintervention_intensity_paths"
            self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
            return "# Final Controller Report\nconvergence_report\nfinal_controller_report\npipeline_status=PIPELINE_COMPLETE"

    l1_l12 = run_research_decision_l1_l12_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_insight_harvester_smoke(
        l1_l12["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "insight_harvester"
    assert "forbidden later-stage/final tokens" in result["run"]["stages"][-1]["error"]
    assert "debug_artifact=" in result["run"]["stages"][-1]["error"]
    assert (tmp_path / "insight_harvester" / "insight_harvester.invalid.md").exists()


def test_convergence_report_uses_r1_and_stops_before_external_calibration(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map\nstrength_by_claim\nuncertainty_and_limits"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises\npremise_risks\ncounterexamples"
            if stage.stage_name == "alternative_generator":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "mutually_exclusive_alternatives\nintervention_intensity_paths"
            if stage.stage_name == "insight_harvester":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "cross_model_insights\nconflicts_and_tensions\ndecision_turning_points"
            assert stage.stage_name == "convergence_report"
            assert stage.owner == R1_32B
            assert stage.model == R1_32B
            assert model == R1_32B
            assert "insight_harvester.md" in prompt
            assert "r1_synthesis.md" in prompt
            assert "web_search" in prompt
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "divergence_role_summary\nconflicts_to_resolve\nconvergence_decision_framework\nuncertainty_boundaries"

    result = run_research_decision_l1_l14_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["insight_harvester", "convergence_report"]
    l3 = stages[3]
    convergence = stages[-1]
    assert convergence["owner"] == R1_32B
    assert convergence["model"] == R1_32B
    assert convergence["executor_model"] == R1_ACTUAL_MODEL_DEFAULT
    assert convergence["artifact_path"].endswith("convergence_report/convergence_report.md")
    assert convergence["artifact_path"] != l3["artifact_path"]
    report = Path(convergence["artifact_path"]).read_text(encoding="utf-8")
    assert "convergence_decision_framework" in report
    assert "final_controller_report" not in report
    assert "pipeline_status=PIPELINE_COMPLETE" not in report
    assert "web_search" not in report
    assert result["full_pipeline_validation"]["valid"] is False
    assert result["full_pipeline_validation"]["divergence_unique_model_count"] == 4
    assert any("missing_stage:external_calibration" in error for error in result["full_pipeline_validation"]["errors"])


def test_convergence_report_blocks_without_all_divergence_roles(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_omlx_model(self, stage, model, prompt):
            raise AssertionError("convergence_report must not run without all divergence roles")

    result = run_research_decision_convergence_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "convergence_report"
    assert result["run"]["stages"][-1]["created_in_current_run"] is False
    assert "requires fresh L1-L13 stages" in result["run"]["stages"][-1]["error"]


def test_convergence_report_blocks_when_unique_divergence_models_less_than_four(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises"
            if stage.stage_name == "alternative_generator":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "mutually_exclusive_alternatives"
            if stage.stage_name == "insight_harvester":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "cross_model_insights"
            raise AssertionError("convergence_report should be blocked before OMLX call")

    l1_l13 = run_research_decision_l1_l13_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    by_name = {stage["stage_name"]: stage for stage in l1_l13["run"]["stages"]}
    by_name["evidence_judge"]["model"] = QWEN72B

    result = run_research_decision_convergence_smoke(
        l1_l13["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "convergence_report"
    assert "unique divergence models < 4" in result["run"]["stages"][-1]["error"]


def test_convergence_report_rejects_l3_artifact_reuse(tmp_path: Path):
    class ReusingExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            self.last_executor_models[stage.stage_name] = (
                R1_ACTUAL_MODEL_DEFAULT if stage.stage_name in {"L3_r1_synthesis", "convergence_report"}
                else {
                    "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                    "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                    "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                    "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                    "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                }[stage.stage_name]
            )
            return "ok"

        def write_artifact(self, stage, content, *, base_dir):
            if stage.stage_name == "convergence_report":
                outputs = {"convergence_report.md": str(Path(base_dir) / "convergence_report" / "convergence_report.md")}
                return Path(base_dir) / "L3_r1_synthesis" / "r1_synthesis.md", outputs
            return super().write_artifact(stage, content, base_dir=base_dir)

    l1_l13 = run_research_decision_l1_l13_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=ReusingExecutor())
    result = run_research_decision_convergence_smoke(
        l1_l13["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=ReusingExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "convergence_report"
    assert "artifact_path must be distinct from L3_r1_synthesis" in result["run"]["stages"][-1]["error"]


def test_convergence_report_output_forbidden_final_or_tool_chain_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "L3_r1_synthesis":
                self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
                return "R1 synthesis body"
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes"
            if stage.stage_name == "evidence_judge":
                self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
                return "evidence_quality_map"
            if stage.stage_name == "premise_auditor":
                self.last_executor_models[stage.stage_name] = LLAMA70B_ACTUAL_MODEL_DEFAULT
                return "implicit_premises"
            if stage.stage_name == "alternative_generator":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "mutually_exclusive_alternatives"
            if stage.stage_name == "insight_harvester":
                self.last_executor_models[stage.stage_name] = GEMMA431B_ACTUAL_MODEL_DEFAULT
                return "cross_model_insights"
            self.last_executor_models[stage.stage_name] = R1_ACTUAL_MODEL_DEFAULT
            return "# Final Controller Report\nweb_search\napi_call\ncodex_exec\nfinal_controller_report\npipeline_status=PIPELINE_COMPLETE"

    l1_l13 = run_research_decision_l1_l13_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_convergence_smoke(
        l1_l13["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "convergence_report"
    error = result["run"]["stages"][-1]["error"]
    assert "forbidden final/tool-chain tokens" in error
    assert "web_search" in error
    assert "api_call" in error
    assert "codex_exec" in error


def test_external_calibration_rejects_forbidden_executor_bindings():
    executor = LocalTaskEngineExecutor()
    for forbidden in (NEMOTRON120B, R1_32B, CONTROLLER_ACCEPTANCE, QWEN72B, LLAMA70B, GEMMA431B):
        stage = StageSpec("external_calibration", forbidden, forbidden, ("external_calibration.md",))
        try:
            executor.run_external_calibration(stage, {"prompt": "calibrate"})
        except RuntimeError as exc:
            assert "external calibration binding mismatch" in str(exc)
        else:
            raise AssertionError(f"external_calibration must reject forbidden executor binding: {forbidden}")


def test_external_calibration_discovers_chatgpt_app_bridge_wrapper_without_env(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    wrapper = tmp_path / "chatgpt_app_bridge_http_cli.py"
    wrapper.write_text("print('unused')\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_GPT_BRIDGE_CMD", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_URL", raising=False)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")
    monkeypatch.setattr(executors, "CHATGPT_APP_BRIDGE_WRAPPER", wrapper)

    assert executors._discover_chatgpt_app_bridge_wrapper() == wrapper


def test_gpt_bridge_timeout_and_settle_defaults(monkeypatch):
    import tools.task_engine_executors as executors

    for key in (
        "HERMES_GPT_BRIDGE_TIMEOUT_S",
        "HERMES_GPT_BRIDGE_SETTLE_S",
        "HERMES_GPT_BRIDGE_SETTLE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")

    assert executors._gpt_bridge_timeout_s() == 360
    assert executors._gpt_bridge_settle_s() == 60


def test_external_calibration_wrapper_success_uses_gpt_bridge_without_gemini(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    secret = "super-secret-token"
    wrapper = tmp_path / "chatgpt_app_bridge_http_cli.py"
    wrapper.write_text(
        "import json, sys\n"
        "assert sys.argv[sys.argv.index('--timeout') + 1] == '360'\n"
        "assert sys.argv[sys.argv.index('--settle') + 1] == '60'\n"
        "print(json.dumps({'success': True, 'response': " + repr(COMPLETE_EXTERNAL_CALIBRATION_MINIMUM_FIELDS) + "}))\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HERMES_GPT_BRIDGE_CMD", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_URL", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_SETTLE_S", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_SETTLE_SECONDS", raising=False)
    monkeypatch.setenv("CHATGPT_BRIDGE_TOKEN", secret)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")
    monkeypatch.setattr(executors, "CHATGPT_APP_BRIDGE_WRAPPER", wrapper)

    class NoGeminiExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            raise AssertionError("Gemini fallback must not run when wrapper succeeds")

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    executor = NoGeminiExecutor()
    output = executor.run_external_calibration(stage, {"prompt": "calibrate this"})

    assert executor.last_executor_models["external_calibration"] == "ChatGPT App Bridge"
    assert "executor_model: ChatGPT App Bridge" in output
    assert "calibration_verdict" in output
    assert secret not in output


def test_external_calibration_wrapper_failure_falls_back_to_gemini(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    wrapper = tmp_path / "chatgpt_app_bridge_http_cli.py"
    wrapper.write_text(
        "import json, sys\n"
        "print(json.dumps({'success': False, 'error': 'worker busy'}))\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HERMES_GPT_BRIDGE_CMD", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_URL", raising=False)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")
    monkeypatch.setattr(executors, "CHATGPT_APP_BRIDGE_WRAPPER", wrapper)

    class FallbackExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            assert stage.model == GEMINI_PRO_HIGH
            assert model == GEMINI_PRO_HIGH
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    executor = FallbackExecutor()
    output = executor.run_external_calibration(stage, {"prompt": "calibrate this"})

    assert executor.last_executor_models["external_calibration"] == GEMINI_PRO_HIGH
    assert "GPT_BRIDGE_UNAVAILABLE:GPT_BRIDGE_BUSY_OR_UNSAFE" in output
    assert "calibration_verdict" in output


def test_external_calibration_wrapper_absent_and_env_absent_not_configured(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    monkeypatch.delenv("HERMES_GPT_BRIDGE_CMD", raising=False)
    monkeypatch.delenv("HERMES_GPT_BRIDGE_URL", raising=False)
    monkeypatch.setattr(executors, "_hermes_env_value", lambda key: "")
    monkeypatch.setattr(executors, "CHATGPT_APP_BRIDGE_WRAPPER", tmp_path / "missing.py")
    monkeypatch.setattr(executors, "_decision_engine_bridge_script", lambda: None)

    class FailingExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            raise RuntimeError("AGY_UNAVAILABLE")

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    try:
        FailingExecutor().run_external_calibration(stage, {"prompt": "calibrate this"})
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("external_calibration must block when GPT and Gemini are unavailable")

    assert "GPT_BRIDGE_UNAVAILABLE:GPT_BRIDGE_NOT_CONFIGURED" in message
    assert "AGY_UNAVAILABLE" in message


def test_external_calibration_gpt_unavailable_falls_back_to_gemini_pro(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.setattr(
        executors,
        "_run_gpt_bridge_calibration",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("GPT_BRIDGE_AUTH_BLOCKED")),
    )

    class FallbackExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            assert stage.stage_name == "external_calibration"
            assert stage.owner == GEMINI_PRO_HIGH
            assert stage.model == GEMINI_PRO_HIGH
            assert model == GEMINI_PRO_HIGH
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    executor = FallbackExecutor()
    output = executor.run_external_calibration(stage, {"prompt": "calibrate this"})

    assert executor.last_executor_models["external_calibration"] == GEMINI_PRO_HIGH
    assert "executor_model: Gemini 3.1 Pro (High)" in output
    assert "GPT_BRIDGE_UNAVAILABLE:GPT_BRIDGE_AUTH_BLOCKED" in output
    assert "calibration_verdict" in output


def test_external_calibration_gpt_header_only_retries_and_saves_diagnostic(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_bridge(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "calibration_scope\nclaim_strength_table\ncalibration_verdict\n"
        return COMPLETE_EXTERNAL_CALIBRATION_MINIMUM_FIELDS

    monkeypatch.setattr(executors, "_run_gpt_bridge_calibration", fake_bridge)
    monkeypatch.setattr(executors, "_gpt_bridge_executor_model", lambda: "ChatGPT App Bridge")
    monkeypatch.setattr(executors, "_gpt_bridge_header_retry_wait_s", lambda: 0)

    class NoGeminiExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            raise AssertionError("Gemini fallback should not run after GPT retry succeeds")

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    executor = NoGeminiExecutor()
    output = executor.run_external_calibration(stage, {"prompt": "calibrate this", "base_dir": str(tmp_path)})

    assert len(calls) == 2
    assert "GPT_BRIDGE_TARGETED_RETRY_USED" in output
    assert "calibration_verdict" in output
    invalid = tmp_path / "external_calibration" / "external_calibration.invalid.md"
    diagnostic = json.loads((tmp_path / "external_calibration" / "external_calibration.diagnostic.json").read_text(encoding="utf-8"))
    assert invalid.exists()
    assert diagnostic["attempt"] == "gpt_bridge_first"
    assert diagnostic["executor_model"] == "ChatGPT App Bridge"
    assert diagnostic["fallback_used"] is False
    assert diagnostic["raw_length"] > 0
    assert "external_calibration_header_only" in diagnostic["error_summary"]


def test_external_calibration_gpt_header_only_retry_fails_then_fallback_gemini(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    calls = []

    def fake_bridge(prompt: str) -> str:
        calls.append(prompt)
        return "calibration_scope\nclaim_strength_table\ncalibration_verdict\n"

    monkeypatch.setattr(executors, "_run_gpt_bridge_calibration", fake_bridge)
    monkeypatch.setattr(executors, "_gpt_bridge_executor_model", lambda: "ChatGPT App Bridge")
    monkeypatch.setattr(executors, "_gpt_bridge_header_retry_wait_s", lambda: 0)

    class FallbackExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return COMPLETE_EXTERNAL_CALIBRATION_MINIMUM_FIELDS

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    executor = FallbackExecutor()
    output = executor.run_external_calibration(stage, {"prompt": "calibrate this", "base_dir": str(tmp_path)})

    assert len(calls) == 2
    assert executor.last_executor_models["external_calibration"] == GEMINI_PRO_HIGH
    assert "GPT_BRIDGE_INVALID_FIRST:external_calibration_header_only" in output
    assert "GPT_BRIDGE_INVALID_RETRY:external_calibration_header_only" in output
    assert "calibration_verdict" in output


def test_external_calibration_header_only_blocks_and_saves_gemini_diagnostic(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    monkeypatch.setattr(
        executors,
        "_run_gpt_bridge_calibration",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("GPT_BRIDGE_NOT_CONFIGURED")),
    )

    class HeaderOnlyGeminiExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
            return "calibration_scope\nclaim_strength_table\ncalibration_verdict\n"

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    try:
        HeaderOnlyGeminiExecutor().run_external_calibration(stage, {"prompt": "calibrate this", "base_dir": str(tmp_path)})
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("header-only external calibration must block")

    assert "artifact_quality_error:external_calibration_header_only" in message
    diagnostic = json.loads((tmp_path / "external_calibration" / "external_calibration.diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["attempt"] == "gemini_fallback"
    assert diagnostic["fallback_used"] is True
    assert diagnostic["executor_model"] == GEMINI_PRO_HIGH


def test_external_calibration_quality_allows_minimum_required_fields():
    import tools.task_engine_executors as executors

    text = "external_calibration\nexecutor_model: ChatGPT App Bridge\nfallback_reasons: []\n\n" + COMPLETE_EXTERNAL_CALIBRATION_MINIMUM_FIELDS

    assert executors._external_calibration_quality_error(text) == ""


def test_external_calibration_quality_allows_agreement_with_convergence_when_fields_complete():
    import tools.task_engine_executors as executors

    text = "\n".join(
        [
            "calibration_verdict: supported overall; convergence is plausible and no major contradiction is found.",
            "agreement_points: supported agreement with convergence on key_drivers, mechanism_chain, scenario_branches, and counter_signals.",
            "disagreement_or_risk_points: speculative risk remains for individual forecasts and future AI environment assumptions.",
            "missing_considerations: missing direct longitudinal evidence, tool quality drift, and school context variation.",
            "final_adjustment_recommendation: keep convergence unchanged but mark long-horizon claims as plausible or speculative.",
        ]
    )

    assert executors._external_calibration_quality_error(text) == ""


def test_external_calibration_quality_ignores_metadata_verdict_false_positive():
    import tools.task_engine_executors as executors

    text = "\n".join(
        [
            "external_calibration",
            "executor_model: ChatGPT App Bridge",
            'fallback_reasons: ["external_calibration_missing_verdict"]',
            'metadata: {"calibration_verdict": "metadata must not satisfy body contract"}',
            "error_summary: missing_verdict",
            "",
            "calibration_scope",
            "Scope body with supported, plausible, speculative, and contradicted labels. " * 30,
            "claim_strength_table",
            "| Claim | Strength | Notes |",
            "| --- | --- | --- |",
            "| A | supported | bounded current evidence |",
            "| B | plausible | conditional mechanism |",
            "| C | speculative | future-facing uncertainty |",
            "| D | contradicted | rejected overreach |",
            "over_inference_checks",
            "The body checks over-inference without giving a calibration conclusion. " * 30,
            "contradiction_checks",
            "The body labels contradictions and keeps unsupported claims downgraded. " * 30,
            "handoff_notes_for_final_controller",
            "Use only supported or plausible claims; speculative claims remain conditional. " * 30,
        ]
    )

    assert executors._external_calibration_quality_error(text) == "external_calibration_missing_verdict"


def test_external_calibration_blocks_when_gpt_and_gemini_unavailable(monkeypatch):
    import tools.task_engine_executors as executors

    monkeypatch.setattr(
        executors,
        "_run_gpt_bridge_calibration",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("GPT_BRIDGE_NOT_CONFIGURED")),
    )

    class FailingExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            raise RuntimeError("AGY_AUTH_BLOCKED")

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    try:
        FailingExecutor().run_external_calibration(stage, {"prompt": "calibrate this"})
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("external_calibration must block when both GPT Bridge and Gemini are unavailable")

    assert "GPT Bridge and Gemini/agy unavailable" in message
    assert "GPT_BRIDGE_UNAVAILABLE:GPT_BRIDGE_NOT_CONFIGURED" in message
    assert "AGY_AUTH_BLOCKED" in message


def test_external_calibration_uses_bridge_or_gemini_and_stops_before_final_controller(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            actual = {
                "L3_r1_synthesis": R1_ACTUAL_MODEL_DEFAULT,
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            if stage.stage_name == "evidence_judge":
                return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"
            if stage.stage_name == "convergence_report":
                return "divergence_role_summary\nconvergence_decision_framework\nuncertainty_boundaries"
            return "stage body"

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_external_calibration(self, stage, packet):
            assert stage.stage_name == "external_calibration"
            assert stage.owner == "GPT Bridge or Gemini/agy"
            assert stage.model == "GPT Bridge or Gemini/agy"
            assert "convergence_report.md" in packet["prompt"]
            assert "research_evidence_packet.md" in packet["prompt"]
            self.last_executor_models[stage.stage_name] = "GPT Bridge"
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

    result = run_research_decision_l1_l15_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == "PIPELINE_INCOMPLETE"
    stages = result["run"]["stages"]
    assert [stage["stage_name"] for stage in stages][-2:] == ["convergence_report", "external_calibration"]
    calibration = stages[-1]
    assert calibration["owner"] == "GPT Bridge or Gemini/agy"
    assert calibration["model"] == "GPT Bridge or Gemini/agy"
    assert calibration["executor_model"] == "GPT Bridge"
    assert calibration["artifact_path"].endswith("external_calibration/external_calibration.md")
    report = Path(calibration["artifact_path"]).read_text(encoding="utf-8")
    assert "calibration_verdict" in report
    assert "final_controller_report" not in report
    assert "pipeline_status=PIPELINE_COMPLETE" not in report
    assert result["full_pipeline_validation"]["valid"] is False
    assert any("missing_stage:final_controller_report" in error for error in result["full_pipeline_validation"]["errors"])


def test_external_calibration_blocks_without_convergence_report(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_external_calibration(self, stage, packet):
            raise AssertionError("external_calibration must not run without convergence_report")

    result = run_research_decision_external_calibration_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "external_calibration"
    assert "requires fresh L1-L14 stages" in result["run"]["stages"][-1]["error"]


def test_external_calibration_output_forbidden_final_terms_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            actual = {
                "L3_r1_synthesis": R1_ACTUAL_MODEL_DEFAULT,
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            return "ok"

        def run_external_calibration(self, stage, packet):
            self.last_executor_models[stage.stage_name] = "GPT Bridge"
            return "# Final Controller Report\nfinal_controller_report\npipeline_status=PIPELINE_COMPLETE"

    l1_l14 = run_research_decision_l1_l14_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_external_calibration_smoke(
        l1_l14["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "external_calibration"
    error = result["run"]["stages"][-1]["error"]
    assert "forbidden final-output tokens" in error
    assert "final_controller_report" in error
    assert "PIPELINE_COMPLETE" in error


def test_final_controller_report_completes_16_stage_pipeline(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            actual = {
                "L3_r1_synthesis": R1_ACTUAL_MODEL_DEFAULT,
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            if stage.stage_name == "evidence_judge":
                return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"
            if stage.stage_name == "convergence_report":
                return "divergence_role_summary\nconflicts_to_resolve\nconvergence_decision_framework\nuncertainty_boundaries"
            return "stage body"

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_external_calibration(self, stage, packet):
            self.last_executor_models[stage.stage_name] = "GPT Bridge"
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

        def run_final_controller_report(self, stage, packet):
            assert stage.stage_name == "final_controller_report"
            assert stage.owner == "Controller"
            assert stage.model == FINAL_CONTROLLER
            assert "external_calibration" in packet["excerpts"]
            assert "convergence_report" in packet["excerpts"]
            assert len(packet["stage_trace"]) == 15
            self.last_executor_models[stage.stage_name] = "Hermes Controller"
            return (
                "# ADHD 儿童研究决策报告\n\n"
                "证据强度：行为支持较强；争议：个体差异和学校环境会影响效果；缺口：长期个体化路线仍需复盘。\n\n"
                "家长行为培训详细方案\n"
                "周期：4-6 周；频率：每天练一个目标、每周复盘；步骤：提示、执行、反馈；记录指标：启动时间和提醒次数；调整规则：失败时降难度。\n\n"
                "三年级准备路线"
            )

    result = run_research_decision_l1_l16_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == PIPELINE_COMPLETE
    assert result["full_pipeline_validation"]["valid"] is True
    assert result["full_pipeline_validation"]["stage_count"] == 16
    assert result["full_pipeline_validation"]["divergence_unique_model_count"] == 4
    stages = result["run"]["stages"]
    assert len([stage for stage in stages if stage["stage_name"] in {
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
    }]) == 5
    final = stages[-1]
    assert final["stage_name"] == "final_controller_report"
    assert final["owner"] == "Controller"
    assert final["model"] == FINAL_CONTROLLER
    assert final["executor_model"] == "Hermes Controller"
    assert final["artifact_path"].endswith("final_controller_report/final_decision_report.md")
    final_text = Path(final["artifact_path"]).read_text(encoding="utf-8")
    assert "家长行为培训详细方案" in final_text
    markdown = result["markdown"]
    assert "entered_engine_run_pipeline=true" in markdown
    assert "pipeline_status=PIPELINE_COMPLETE" in markdown
    assert "pipeline_validation.valid=true" in markdown
    assert "delegation_used=false" in markdown
    assert "家长行为培训详细方案" in markdown
    assert "Compact Pipeline Trace:" in markdown
    assert sum(1 for line in markdown.splitlines() if line.startswith("- ")) == 16
    assert "web_search" not in markdown
    assert "api_call" not in markdown
    assert "codex_exec" not in markdown
    assert "persona raw" not in markdown


def test_decision_final_smoke_completes_10_stage_pipeline_without_research(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            assert stage.stage_name == "intelligence_layer"
            assert "DECISION mode" in prompt
            assert "RESEARCH L1-L5" in prompt
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\ndecision_dimensions_for_later_stages"

        def run_ddgs(self, stage, queries):
            assert stage.stage_name == "supplementary_search"
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            assert "L1_gemini_search" not in prompt
            actual = {
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            if stage.stage_name == "convergence_report":
                return "divergence_role_summary\nconflicts_to_resolve\nconvergence_decision_framework\nuncertainty_boundaries"
            if stage.stage_name == "evidence_judge":
                return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"
            return f"{stage.stage_name} body"

        def run_external_calibration(self, stage, packet):
            assert "research_evidence_packet" not in packet["prompt"]
            self.last_executor_models[stage.stage_name] = "ChatGPT App Bridge"
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

        def run_final_controller_report(self, stage, packet):
            assert stage.stage_name == "final_controller_report"
            assert len(packet["stage_trace"]) == 9
            assert "external_calibration" in packet["excerpts"]
            assert "L5_deepseek_acceptance" not in packet["excerpts"]
            self.last_executor_models[stage.stage_name] = "Hermes Controller"
            return (
                "# 决策任务最终报告\n\n"
                "decision_mode=true\n\n"
                "## 决策问题\nADHD 是否要主动干预。\n\n"
                "## 建议方向\n先确认约束和升级阈值。\n\n"
                "证据强度：行为支持较强；争议：个体差异明显；缺口：本轮未执行 RESEARCH L1-L5。\n"
                "周期：4-6 周；频率：每天；步骤：观察、记录、复盘；记录指标：启动时间；调整规则：失败时降难度。"
            )

    result = run_decision_final_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "ok"
    assert result["pipeline_status"] == PIPELINE_COMPLETE
    assert result["full_pipeline_validation"]["valid"] is True
    assert result["full_pipeline_validation"]["stage_count"] == 10
    assert result["full_pipeline_validation"]["divergence_unique_model_count"] == 4
    stages = result["run"]["stages"]
    names = [stage["stage_name"] for stage in stages]
    assert names == [stage.stage_name for stage in CANONICAL_STAGES[ENGINE_DECISION]]
    assert not any(name.startswith("L") for name in names)
    assert names.count("convergence_report") == 1
    assert [stage["model"] for stage in stages if stage["model"] == R1_32B] == [R1_32B]
    calibration = stages[8]
    assert calibration["stage_name"] == "external_calibration"
    assert calibration["executor_model"] == "ChatGPT App Bridge"
    final = stages[-1]
    assert final["artifact_path"].endswith("final_controller_report/final_decision_report.md")
    markdown = result["markdown"]
    assert "pipeline_status=PIPELINE_COMPLETE" in markdown
    assert "pipeline_validation.valid=true" in markdown
    assert sum(1 for line in markdown.splitlines() if line.startswith("- ")) == 10
    assert "L1_gemini_search" not in markdown
    assert "research_evidence_packet.md" not in markdown
    assert "web_search" not in markdown
    assert "api_call" not in markdown
    assert "codex_exec" not in markdown


def test_artifact_quality_blocks_agy_error_text_at_intelligence_layer(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            return "Error: timed out waiting for response"

        def run_ddgs(self, stage, queries):
            raise AssertionError("supplementary_search must not run after invalid intelligence artifact")

    result = run_decision_final_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["pipeline_status"] == PIPELINE_BLOCKED
    assert result["blocked_stage"] == "intelligence_layer"
    stage = result["run"]["stages"][-1]
    assert stage["stage_name"] == "intelligence_layer"
    assert stage["valid_for_pipeline"] is False
    assert "artifact_quality_error:error_timeout_response" in stage["error"]


def test_artifact_quality_blocks_omlx_error_text_at_structure_mapper(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nThis notes timeout risk as a planning risk, not an executor failure."

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "[ERROR: OMLX backend failed]"
            raise AssertionError("later OMLX stages must not run after invalid structure_mapper artifact")

    result = run_decision_final_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "structure_mapper"
    stage = result["run"]["stages"][-1]
    assert stage["stage_name"] == "structure_mapper"
    assert stage["valid_for_pipeline"] is False
    assert "artifact_quality_error:bracket_error" in stage["error"]


def test_artifact_quality_allows_normal_timeout_risk_discussion(tmp_path: Path):
    stage = CANONICAL_STAGES[ENGINE_DECISION][0]
    executor = LocalTaskEngineExecutor()
    content = "user_question_map\nThis discusses timeout risk as an operational risk, not an executor error."

    artifact_path, outputs = executor.write_artifact(stage, content, base_dir=tmp_path)
    record = executor.make_stage_record(
        stage,
        base_dir=tmp_path,
        artifact_path=artifact_path,
        outputs=outputs,
        created=True,
        valid=True,
        status="real",
        executor_model=GEMINI_HIGH,
    )

    assert record.valid_for_pipeline is True
    assert Path(artifact_path).read_text(encoding="utf-8") == content


def test_artifact_quality_token_detection_is_head_scoped():
    import tools.task_engine_executors as executors

    normal = "## Risk notes\nThe team should monitor timeout risk during external calls."
    bad = "Error: timed out waiting for response\n"
    quoted_late = "Good artifact body\n" + ("\n" * 45) + "Error: timed out waiting for response"

    assert executors._artifact_error_token(normal) == ""
    assert executors._artifact_error_token(bad) == "error_timeout_response"
    assert executors._artifact_error_token(quoted_late) == ""



def test_external_calibration_quality_blocks_truncated_claim_strength_table():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][8]
    half = (
        "calibration_scope\nThis begins a calibration but is incomplete.\n\n"
        "claim_strength_table"
    )

    try:
        executors._assert_artifact_quality(stage, half)
    except RuntimeError as exc:
        assert "artifact_quality_error:external_calibration_header_only" in str(exc)
    else:
        raise AssertionError("truncated external calibration should be blocked")


def test_external_calibration_quality_allows_complete_calibration_fixture():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][8]
    complete = (
        "calibration_scope\n" + "Scope text. " * 80
        + "claim_strength_table\n| Claim | Strength | Notes |\n| --- | --- | --- |\n"
        "| A | supported | grounded in packet |\n"
        "| B | plausible | bounded inference |\n"
        "| C | speculative | needs confirmation |\n"
        "| D | contradicted | conflict noted |\n"
        + "over_inference_checks\n" + "No overreach. " * 60
        + "contradiction_checks\n" + "Contradictions are labeled. " * 60
        + "calibration_verdict\nverdict: calibrated for final controller handoff.\n"
        "handoff_notes_for_final_controller\nUse calibrated claims only.\n"
    )

    assert executors._external_calibration_quality_error(complete) == ""
    executors._assert_artifact_quality(stage, complete)


def test_final_controller_quality_blocks_raw_and_truncated_outputs():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]

    for text in (
        "# Final\n\npersona raw: convergence dump",
        "# Final\n\nThe evidence pac",
    ):
        try:
            executors._assert_artifact_quality(stage, text)
        except RuntimeError as exc:
            assert "artifact_quality_error:" in str(exc)
        else:
            raise AssertionError("invalid final controller artifact should be blocked")


def test_final_controller_quality_blocks_decision_mode_family_advice_leak():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    bad = "# ADHD 儿童研究决策报告\n\ndecision_mode=true\n\n## 家长行为培训详细方案\n..."

    try:
        executors._assert_artifact_quality(stage, bad)
    except RuntimeError as exc:
        assert "artifact_quality_error:decision_mode_family_advice_leak" in str(exc)
    else:
        raise AssertionError("DECISION final artifact should not leak family research-decision template")


def test_research_decision_final_strips_external_calibration_raw_metadata_from_body():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_RESEARCH_DECISION,
        "query": ADHD_PROMPT,
        "stage_trace": [],
        "excerpts": {
            "external_calibration": (
                "external_calibration\n"
                "executor_model: Gemini 3.1 Pro (High)\n"
                "fallback_reasons: [\"GPT bridge unavailable\"]\n"
                "### calibration_verdict\nConditionally accepted with bounded caveats."
            ),
            "convergence_report": "convergence_report\n### synthesis\nUse a stepped intervention frame.",
            "evidence_judge": "evidence_judge\nEvidence is strongest for behavioral scaffolding.",
            "premise_auditor": "premise_auditor\nAvoid overclaiming mechanism certainty.",
            "alternative_generator": "alternative_generator\nAlternative path is school-first support.",
        },
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "external_calibration executor_model" not in text
    assert "executor_model:" not in text
    assert "fallback_reasons:" not in text
    assert "Conditionally accepted" in text
    executors._assert_artifact_quality(CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][-1], text)


def test_research_decision_final_quality_allows_executor_model_only_in_compact_trace(tmp_path: Path):
    import tools.task_engine_contracts as contracts

    artifact = tmp_path / "final_controller_report" / "final_decision_report.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("# ADHD 儿童研究决策报告\n\n## 结论摘要\n正文不含 raw metadata。", encoding="utf-8")
    run = {
        "stages": [
            {
                "stage_name": "final_controller_report",
                "owner": "Controller",
                "model": FINAL_CONTROLLER,
                "executor_model": "Hermes Controller",
                "artifact_path": str(artifact),
                "valid_for_pipeline": True,
            },
            {
                "stage_name": "external_calibration",
                "owner": "GPT Bridge or Gemini/agy",
                "model": "GPT Bridge or Gemini/agy",
                "executor_model": "Gemini 3.1 Pro (High)",
                "artifact_path": "external_calibration/external_calibration.md",
                "valid_for_pipeline": True,
            },
        ]
    }
    validation = {"valid": True, "pipeline_status": PIPELINE_COMPLETE, "errors": [], "stage_count": 2}

    markdown = contracts.render_final_markdown(ENGINE_RESEARCH_DECISION, run, validation, base_dir=tmp_path)

    body = markdown.split("Compact Pipeline Trace:", 1)[0]
    trace = markdown.split("Compact Pipeline Trace:", 1)[1]
    assert "executor_model" not in body
    assert "executor_model=Gemini 3.1 Pro (High)" in trace
    assert "pipeline_status=PIPELINE_COMPLETE" in markdown


def test_research_decision_final_quality_blocks_external_calibration_metadata_in_body():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][-1]
    bad = "# ADHD 儿童研究决策报告\n\n## 证据与校准边界\n外部校准摘要：external_calibration executor_model: Gemini 3.1 Pro (High)"

    try:
        executors._assert_artifact_quality(stage, bad)
    except RuntimeError as exc:
        assert "artifact_quality_error:raw_intermediate_dump" in str(exc)
    else:
        raise AssertionError("raw external_calibration metadata in final body should be blocked")


def _decision_future_query() -> str:
    return (
        "这是一个 DECISION 任务。不要家长建议，不要培养计划，不要文献综述。"
        "请只输出：未来优势变陷阱 Top5；未来缺陷变优势 Top5；"
        "最危险的错误培养路径；最反直觉但值得追踪的假设；danger_flag。"
    )


def _valid_decision_five_section_report() -> str:
    return "\n".join(
        [
            "# 决策任务最终报告",
            "",
            "decision_mode=true",
            "",
            "## 未来优势变陷阱 Top5",
            "关键驱动变量：AI 降低知识获取成本、即时反馈变多、验证成本相对升高。确定性等级：高 / 中 / 低。",
            "1. 快速理解可能变成未经验证的快速接受。",
            "2. 兴趣驱动可能变成持续换题。",
            "3. 即时反馈偏好可能削弱慢变量耐受。",
            "4. 发散思维可能变成观点过剩。",
            "5. 低价值重复抗拒可能削弱必要核查。",
            "",
            "## 未来缺陷变优势 Top5",
            "输入变量 → 中介机制 → 输出变量：低价值重复减少 → 任务价值敏感度放大 → 筛选任务和发现异常的优势。",
            "1. 对低价值重复敏感。",
            "2. 联想跳跃利于问题发现。",
            "3. 非线性推进利于个性化学习。",
            "4. 内在想法多可形成创意池。",
            "5. 运动纪律可支撑长期项目。",
            "",
            "## 最危险的错误培养路径",
            "情景分叉：情景 A 保留验证；情景 B 用即时答案替代判断。",
            "把速度和产出放在目标选择、事实核查和收束能力之前。",
            "",
            "## 最反直觉但值得追踪的假设",
            "未来稀缺的可能是停下来判断问题是否值得，而不是更快得到答案。",
            "",
            "## danger_flag",
            "证据强度：基础执行功能证据较强；争议：未来 AI 场景差异大；缺口：缺少长期直接证据。可观察指标 / 反证信号：是否保留推理痕迹。",
            "如果长期依赖即时答案、跳过验证、频繁换题且很少闭环，风险升高。",
        ]
    )


def test_decision_final_quality_blocks_raw_external_calibration_dump():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    bad = _valid_decision_five_section_report() + "\nexternal_calibration executor_model: ChatGPT App Bridge fallback_reasons: []"

    try:
        executors._assert_artifact_quality(stage, bad)
    except RuntimeError as exc:
        assert "artifact_quality_error:raw_intermediate_dump" in str(exc)
    else:
        raise AssertionError("raw external_calibration dump should be blocked")


def test_decision_final_quality_blocks_evidence_judge_raw_table():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    bad = (
        _valid_decision_five_section_report()
        + "\n**Evidence Judge – DECISION Stage**\n"
        + "| Artifact (Stage) | Content Summary | Strength / Quality of Evidence |\n"
        + "| --- | --- | --- |\n"
        + "| evidence_judge | raw table | Low – intensity |\n"
    )

    try:
        executors._assert_artifact_quality(stage, bad)
    except RuntimeError as exc:
        assert "artifact_quality_error:raw_intermediate_dump" in str(exc) or "artifact_quality_error:raw_table_dump" in str(exc)
    else:
        raise AssertionError("Evidence Judge raw table should be blocked")


def test_decision_final_quality_blocks_truncated_tail_fragments():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    for tail in ("deficit neutr", "Low – inten", "最危险的错误培养路径 ... 过"):
        bad = _valid_decision_five_section_report() + "\n" + tail
        try:
            executors._assert_artifact_quality(stage, bad)
        except RuntimeError as exc:
            assert "artifact_quality_error:truncated_tail" in str(exc)
        else:
            raise AssertionError(f"truncated tail should be blocked: {tail}")


def test_decision_final_packet_quality_blocks_user_forbidden_advice_terms():
    import tools.task_engine_executors as executors

    packet = {"mode": ENGINE_DECISION, "query": _decision_future_query()}
    bad = _valid_decision_five_section_report() + "\n## 下一步\n建议方向：先做专业评估。"

    try:
        executors._assert_final_controller_packet_quality(packet, bad)
    except RuntimeError as exc:
        message = str(exc)
        assert "forbidden_user_terms" in message
        assert "下一步" in message
        assert "建议方向" in message
        assert "专业评估" in message
    else:
        raise AssertionError("user-forbidden advice terms should be blocked")


def test_decision_final_packet_quality_blocks_shallow_five_section_answer():
    import tools.task_engine_executors as executors

    packet = {"mode": ENGINE_DECISION, "query": _decision_future_query()}
    text = _valid_decision_five_section_report()

    try:
        executors._assert_final_controller_packet_quality(packet, text)
    except RuntimeError as exc:
        message = str(exc)
        assert "missing_judgment_unit_fields" in message or "user_facing_quality" in message
    else:
        raise AssertionError("shallow five-section final should not pass production quality")


def test_decision_final_uses_foresight_template_from_profile_without_exact_query_phrases():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "这是一个 DECISION 任务。未来10年 AI 降低知识获取成本后 ADHD 儿童结构性反转判断。",
        "output_quality_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "excerpts": {},
        "convergence_fixed_section_digest": (
            "## key_drivers\nAI feedback cost.\n"
            "## mechanism_chain\ninput variable -> mediating mechanism -> output variable.\n"
            "## scenario_branches\nScenario A keeps verification; Scenario B bypasses it.\n"
            "## counter_signals\nobservable signal.\n"
            "## certainty_levels\nhigh / medium / low.\n"
            "## uncertainty_boundary\nfuture evidence boundary."
        ),
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "## 未来优势变陷阱 Top5" in text
    assert "convergence_fixed_section_digest" not in text
    assert "## 收敛后的关键判断" in text
    assert "AI feedback cost" in text
    assert "input variable -> mediating mechanism -> output variable" in text
    assert "Scenario A keeps verification" in text
    assert "## key_drivers" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_decision_final_packet_absorbs_convergence_fixed_sections(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = _decision_final_prior_stage_records(tmp_path)
    convergence_path = tmp_path / "convergence_report" / "convergence_report.md"
    convergence_path.write_text(
        "\n".join(
            [
                "## key_drivers",
                "AI cost collapse and verification scarcity.",
                "## mechanism_chain",
                "input variable -> mediating mechanism -> output variable.",
                "## scenario_branches",
                "Scenario A keeps verification; Scenario B outsources judgment.",
                "## counter_signals",
                "observable signal: child keeps slow-loop completion.",
                "## certainty_levels",
                "high / medium / low by claim.",
                "## uncertainty_boundary",
                "No direct ten-year individual evidence.",
            ]
        ),
        encoding="utf-8",
    )
    packet = executors._decision_final_controller_packet(
        stages,
        query="未来10年 AI 降低知识获取成本后 ADHD 结构性反转 decision",
        base_dir=tmp_path,
    )
    text = executors._final_controller_report_from_packet(packet)

    assert "convergence_fixed_section_digest" not in text
    assert "AI cost collapse" in text
    assert "input variable -> mediating mechanism -> output variable" in text
    assert "Scenario A keeps verification" in text
    assert "high / medium / low" in text
    assert "## key_drivers" not in text


def test_decision_final_packet_absorbs_external_calibration_hard_constraints(tmp_path: Path):
    import tools.task_engine_executors as executors

    stages = _decision_final_prior_stage_records(tmp_path)
    calibration_path = tmp_path / "external_calibration" / "external_calibration.md"
    calibration_path.write_text(
        "\n".join(
            [
                "external_calibration",
                "calibration_verdict",
                "可进入 final controller，但必须执行降调。",
                "disagreement_or_risk_points",
                "PFC 萎缩、多巴胺抗性、严重依赖预测都不得写成事实。",
                "final_adjustment_recommendation",
                "把强机制词改为执行练习减少风险；把长期结果降级为 foresight_hypothesis；逐条绑定反证信号。",
            ]
        ),
        encoding="utf-8",
    )

    packet = executors._decision_final_controller_packet(
        stages,
        query="未来10年 AI 降低知识获取成本后结构性反转 decision",
        base_dir=tmp_path,
    )
    text = executors._final_controller_report_from_packet(packet)

    assert "external_calibration_hard_constraints" in packet
    assert "## 需要下调和落地的判断" in text
    assert "执行练习减少风险" in text
    assert "前瞻假设" in text
    assert "逐条绑定反证信号" in text
    executors._assert_final_controller_packet_quality(packet, text)



def _adhd_golden_query() -> str:
    return (
        "一个 7.5 岁、二年级男孩，IQ 124，有明确 ADHD 倾向，视觉空间优势，"
        "系列关系和抽象推理偏弱，热爱柔术，目标长期练到 300-500 小时。"
        "未来 10 年 AI 会显著降低知识获取和表达成本。请从长期培养角度判断：\n"
        "1. ADHD 倾向在 AI 时代可能变成优势的 Top 5 场景；\n"
        "2. ADHD 倾向最容易变成陷阱的 Top 5 场景；\n"
        "3. 这个孩子未来 10 年最危险的错误培养路径是什么；\n"
        "4. 最反直觉但最可能有价值的培养假设是什么；\n"
        "5. 柔术、身体经验、阅读、数学、AI 工具使用之间应该如何排序；\n"
        "6. 哪些判断是证据支持，哪些只是 plausible，哪些是 speculative；\n"
        "7. 给出面向家长的可执行长期方向，但不要写成医疗诊断或用药建议。"
    )


def _adhd_golden_packet() -> dict:
    import tools.task_engine_executors as executors

    return {
        "mode": ENGINE_DECISION,
        "query": _adhd_golden_query(),
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "convergence_fixed_section_digest": "\n".join(
            [
                "## key_drivers",
                "ADHD traits, AI, visual-spatial strength, BJJ, sequential weakness.",
                "## mechanism_chain",
                "Low-friction AI can either scaffold or bypass executive function.",
                "## scenario_branches",
                "Scenario A keeps BJJ and reading/math foundations; Scenario B overuses AI.",
                "## counter_signals",
                "Overreliance on AI and missing slow-loop completion.",
                "## certainty_levels",
                "Supported / plausible / speculative.",
                "## uncertainty_boundary",
                "Ten-year individual outcome evidence is limited.",
            ]
        ),
        "external_calibration_hard_constraints": "\n".join(
            [
                "Elevate the Risk: premature AI as executive-function crutch before working memory and sequential logic.",
                "Lock in the Sequence: BJJ and Reading/Math from ages 7.5 to 10; AI scaffold before prosthetic.",
                "Restore Counter-Intuitive Strategy: use BJJ spatial and leverage patterns to teach abstract math.",
                "Strict Epistemic Tagging: label Evidence, Plausible, Speculative.",
            ]
        ),
        "research_evidence_packet_context": "## evidence_supported\nADHD supports and structured movement.\n## reasonable_inference\nBJJ may serve as somatic anchor.\n## foresight_hypothesis\nAI-era novelty seeking may become advantage.",
        "excerpts": {},
    }


def test_adhd_golden_final_controller_user_facing_hardening_passes():
    import tools.task_engine_executors as executors

    packet = _adhd_golden_packet()
    text = executors._final_controller_report_from_packet(packet)

    assert executors._case_anchor_usage_failures(packet["query"], text) == []
    assert executors._calibration_implementation_failures(packet["external_calibration_hard_constraints"], text) == []
    for index in range(1, 8):
        assert f"## {index}." in text
    for forbidden in (
        "decision_mode=true",
        "final controller",
        "external_calibration",
        "convergence_report",
        "research packet",
        "研究包吸收",
        "校准执行",
        "StageRecord",
        "artifact",
        "pipeline",
        "Executor",
    ):
        assert forbidden not in text
    for label in ("[证据支持]", "[合理推断]", "[前瞻假设]", "[不支持/风险]"):
        assert label in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_adhd_golden_final_controller_quality_blocks_current_bad_shape():
    import tools.task_engine_executors as executors

    packet = _adhd_golden_packet()
    bad = "\n".join(
        [
            "# 决策任务最终报告",
            "decision_mode=true",
            "## 校准执行",
            "已吸收的校准要求：Elevate the Risk; Lock in the Sequence.",
            "## 未来优势变陷阱 Top5",
            "1. 快速理解可能变成未经验证的快速接受。",
            "## 最危险的错误培养路径",
            "泛泛而谈，没有 Q5 排序，没有孩子画像，没有逐条证据标签。",
        ]
    )

    try:
        executors._assert_final_controller_packet_quality(packet, bad)
    except RuntimeError as exc:
        message = str(exc)
        assert "user_facing_quality" in message or "artifact_quality_error" in message
    else:
        raise AssertionError("bad final controller output should fail user-facing quality gate")


def test_generic_enumerated_decision_requires_numbered_answers():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "请判断：\n1. 市场机会是什么？\n2. 最大风险是什么？\n3. 如何排序？",
        "output_quality_profile": [],
        "excerpts": {},
    }
    good = "# 最终答案\n\n## 1. 市场机会\n越南市场机会主要来自本地需求增长、供应链转移和早期渠道空白；触发条件是能找到可验证付费客户，中间机制是小规模试点降低进入成本，反证信号是获客成本持续高于毛利空间。\n\n## 2. 最大风险\n最大风险是渠道、合规和交付能力跟不上销售承诺；触发条件是本地伙伴质量不稳定，中间机制是履约失败会放大退款和品牌损失，反证信号是试点客户复购并能稳定交付。\n\n## 3. 如何排序\n1. 客户验证：先证明真实付费和复购，因为没有需求就不应扩张。\n2. 渠道伙伴：再验证本地交付与获客，因为渠道决定现金效率。\n3. 合规与本地化：最后扩展投入，因为它们应跟随已验证需求逐步加码。\n\n## 证据边界\n证据强度：中。争议点：取决于执行场景。证据缺口：缺少长期验证。"
    bad = "# 最终答案\n\n市场机会和风险都存在，但没有逐项回答。"

    executors._assert_final_controller_packet_quality(packet, good)
    try:
        executors._assert_final_controller_packet_quality(packet, bad)
    except RuntimeError as exc:
        assert "missing_enumerated_answer" in str(exc)
    else:
        raise AssertionError("generic enumerated question should require explicit numbered answers")


def _assert_final_quality_rejects(packet: dict, text: str, expected_token: str | None = None) -> None:
    import tools.task_engine_executors as executors

    try:
        executors._assert_final_controller_packet_quality(packet, text)
    except RuntimeError as exc:
        if expected_token:
            assert expected_token in str(exc)
    else:
        raise AssertionError("final quality gate should reject adversarial output")


def _adversarial_surface_terms() -> str:
    return " ".join(
        [
            "7.5 岁 二年级 IQ 124 ADHD 倾向 视觉空间优势 系列关系弱 抽象推理弱 柔术 300-500 小时 阅读 数学 AI 工具 未来 10 年 家长 非医疗诊断 非用药建议。",
            "执行功能拐杖 工作记忆 顺序逻辑 阅读 数学 排序 7.5-10 岁 柔术 AI 工具 scaffold 空间 杠杆 抽象数学。",
            "[证据支持] [合理推断] [前瞻假设] [不支持/风险] 触发条件：有。中间机制：有。失效条件 / 反证信号：有。确定性：中。家长怎么用：看。",
        ]
    )


def _seven_numbered_shell(body_line: str) -> str:
    return "\n".join(
        [
            "# 最终答案",
            _adversarial_surface_terms(),
            "## 1. ADHD 优势 Top 5 场景",
            "1. 重要。2. 重要。3. 重要。4. 重要。5. 重要。",
            "## 2. ADHD 陷阱 Top 5 场景",
            "1. 风险。2. 风险。3. 风险。4. 风险。5. 风险。",
            "## 3. 最危险错误路径",
            body_line,
            "## 4. 反直觉假设",
            body_line,
            "## 5. 柔术 / 身体经验 / 阅读 / 数学 / AI 工具使用排序",
            body_line,
            "## 6. 证据支持 / 合理推断 / 前瞻假设分层",
            "[证据支持] [合理推断] [前瞻假设] [不支持/风险] 这是判断。",
            "## 7. 家长可执行方向",
            body_line,
            "## 关键驱动变量与机制链",
            "关键驱动变量。机制链。",
            "## 情景分叉",
            "情景 A。情景 B。",
            "## 观察指标与反证信号",
            "反证信号。",
            "## 证据强度、争议和缺口",
            "证据强度：中。争议点：取决于场景。证据缺口：缺少长期验证。",
        ]
    )


def test_final_controller_quality_blocks_anchor_stuffing():
    packet = _adhd_golden_packet()
    bad = _seven_numbered_shell("这些内容都很重要，但这里没有把孩子画像用于判断，只是在重复空话。")

    _assert_final_quality_rejects(packet, bad, "user_facing_quality")


def test_final_controller_quality_blocks_calibration_restatement():
    packet = _adhd_golden_packet()
    bad = _seven_numbered_shell(
        "已吸收 elevate risk / lock sequence / restore counterintuitive / strict epistemic tagging，已执行校准要求。"
    )

    _assert_final_quality_rejects(packet, bad, "user_facing_quality")


def test_final_controller_quality_blocks_evidence_label_spam():
    packet = _adhd_golden_packet()
    bad = _seven_numbered_shell("[证据支持] [合理推断] [前瞻假设] [不支持/风险] 这是判断。")

    _assert_final_quality_rejects(packet, bad, "user_facing_quality")


def test_final_controller_quality_blocks_internal_language_variant():
    packet = _adhd_golden_packet()
    bad = _seven_numbered_shell("本阶段输出已经检查上游报告，校准阶段要求已经落实，artifact 检查通过。")

    _assert_final_quality_rejects(packet, bad, "internal_language")


def test_generic_enumerated_non_adhd_good_passes_without_adhd_anchors():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "请判断：\n1. 是否进入越南市场？\n2. 最大风险？\n3. 资源如何排序？\n4. 反证信号？",
        "output_quality_profile": [],
        "excerpts": {},
    }
    text = "\n".join(
        [
            "# 最终答案",
            "## 1. 是否进入越南市场",
            "建议先以低成本试点进入，而不是一次性重资产进入。触发条件是能找到明确付费客户和本地交付伙伴；中间机制是试点能验证需求、价格和交付难度；反证信号是获客成本高、客户复购弱或本地伙伴无法稳定履约。",
            "## 2. 最大风险",
            "最大风险是渠道质量、合规边界和交付能力不匹配。触发条件是销售承诺快于本地履约建设；中间机制是服务失败会放大退款、口碑和现金流压力；反证信号是连续试点都能按成本和周期交付。",
            "## 3. 资源如何排序",
            "1. 客户验证：先证明真实付费，因为没有需求就不应扩张。\n2. 渠道伙伴：再验证本地获客和交付，因为渠道决定现金效率。\n3. 合规与本地化：最后随已验证需求加码，因为过早投入会增加沉没成本。",
            "## 4. 反证信号",
            "如果三个月内没有复购客户、获客成本高于毛利空间、合规审批周期无法预测，或本地伙伴交付失败率持续偏高，就应停止扩张并回到市场验证阶段。",
            "## 证据边界",
            "证据强度：中。争议点：不同城市和渠道伙伴会改变结论。证据缺口：缺少真实试点数据和本地合规反馈。",
        ]
    )

    executors._assert_final_controller_packet_quality(packet, text)


def test_plain_non_enumerated_good_does_not_require_numbered_answers():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "请判断是否应该把一个内部数据平台迁移到新的云供应商，并给出边界。",
        "output_quality_profile": [],
        "excerpts": {},
    }
    text = "\n".join(
        [
            "# 最终答案",
            "## 核心判断",
            "可以先做可逆迁移试点，不应一次性全量迁移。理由是新供应商可能改善成本和弹性，但安全、锁定、迁移中断和团队能力仍是主要约束。",
            "## 风险边界",
            "先验证身份权限、审计日志、回滚路径、关键任务延迟和真实月度成本；如果任一指标低于现有平台，应停止扩大迁移范围。",
            "## 证据边界",
            "证据强度：中。争议点：工作负载和团队经验会改变结论。证据缺口：缺少真实压测和账单数据。",
        ]
    )

    executors._assert_final_controller_packet_quality(packet, text)


def test_topk_regression_good_passes_for_non_adhd_decision():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "请给出一个非 ADHD 的 Top 5 市场进入风险场景，并说明判断边界。",
        "output_quality_profile": [],
        "excerpts": {},
    }
    text = "\n".join(
        [
            "# 最终答案",
            "## Top 5 市场进入风险场景",
            "1. 渠道风险：如果本地渠道掌握客户关系但缺少交付能力，销售增长会先于履约能力，导致退款、投诉和现金流压力。",
            "2. 合规风险：如果许可、数据、税务或劳动规则理解不足，早期收入可能被审批延期或罚款抵消。",
            "3. 定价风险：如果价格只参考原市场而没有本地支付意愿验证，成交看似存在但毛利无法覆盖获客和交付成本。",
            "4. 供应链风险：如果关键供应商交期和质量不稳定，市场进入会变成运营救火，无法形成可复制流程。",
            "5. 管理带宽风险：如果总部团队同时管理多个新项目，进入市场会稀释核心业务注意力并延迟复盘。",
            "## 判断边界",
            "证据强度：中。争议点：行业、城市和合作伙伴差异会改变排序。证据缺口：缺少试点客户和本地成本数据。",
        ]
    )

    executors._assert_final_controller_packet_quality(packet, text)


def test_topk_title_only_bad_is_rejected():
    packet = {
        "mode": ENGINE_DECISION,
        "query": "请给出一个非 ADHD 的 Top 5 市场进入风险场景，并说明判断边界。",
        "output_quality_profile": [],
        "excerpts": {},
    }
    bad = "# 最终答案\n\n## Top 5 市场进入风险场景\n这些风险都需要平衡。\n\n## 判断边界\n证据强度：中。争议点：很多。证据缺口：很多。"

    _assert_final_quality_rejects(packet, bad, "user_facing_quality")


def test_decision_final_foresight_fallback_uses_judgment_units_and_evidence_tiers():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "未来10年技术变化下做结构性决策判断。",
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "research_evidence_packet_context": "research_packet_digest: evidence_supported / reasonable_inference / foresight_hypothesis",
        "excerpts": {},
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "## 证据支持" in text
    assert "## 合理推断" in text
    assert "## 前瞻假设" in text
    assert "触发条件：" in text
    assert "中间机制：" in text
    assert "失效条件 / 反证信号：" in text
    assert "确定性：" in text
    assert "[证据支持]" in text
    assert "决策含义：" in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_decision_final_absorbs_research_packet_evidence_tiers():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "未来10年技术变化下做结构性决策判断。",
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "research_evidence_packet_context": "\n".join(
            [
                "research_packet_path: /tmp/raw_packet.md",
                "boundary: do not dump raw packet",
                "research_packet_digest:",
                "## evidence_supported",
                "Supported alpha claim from the packet.",
                "## reasonable_inference",
                "Inference beta connects evidence to the decision mechanism.",
                "## foresight_hypothesis",
                "Hypothesis gamma is conditional and must be tracked.",
            ]
        ),
        "excerpts": {},
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "## 证据支持" in text
    assert "## 合理推断" in text
    assert "## 前瞻假设" in text
    assert "research_packet_path:" not in text
    assert "boundary: do not dump raw packet" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_research_decision_foresight_final_uses_generic_packet_synthesis_without_domain_leak():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_RESEARCH_DECISION,
        "query": "未来 10 年 AI 持续降低法律检索、合同起草、案例摘要和合规解释成本后，初级律师和企业法务分析师的优势与劣势是否会发生结构性反转？",
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "excerpts": {
            "L5_deepseek_acceptance": "research_evidence_packet says ADHD 孩子 柔术 家长训练 should never be copied into this final.",
            "convergence_report": "key_drivers and mechanism_chain support partial operational reversal.",
            "external_calibration": "Final calibration sentence: partial reversal is plausible; broad structural reversal is speculative.",
        },
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "局部/场景性结构反转可能成立，职业整体反转证据不足。" in text
    assert "### evidence_supported" in text
    assert "### reasonable_inference" in text
    assert "### foresight_hypothesis" in text
    assert "source_consumption_check：research_evidence_packet=yes; convergence_report=yes; external_calibration=yes" in text
    assert "ADHD" not in text
    assert "孩子" not in text
    assert "柔术" not in text
    assert "家长训练" not in text
    assert "fallback_reasons:" not in text
    assert "executor_model:" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_research_decision_foresight_final_keeps_audit_finance_query_out_of_legal_bucket():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_RESEARCH_DECISION,
        "query": "未来10年 AI 持续降低审计底稿整理、财务异常检测、合规报告生成和管理层讨论分析草稿成本后，初级审计员和企业财务分析师的优势与劣势是否会发生结构性反转？",
        "output_quality_profile": [
            executors.PROFILE_EVIDENCE_GROUNDED,
            executors.PROFILE_FORESIGHT_MECHANISM,
        ],
        "excerpts": {
            "L5_deepseek_acceptance": "审计底稿、财务异常检测、合规报告和管理层讨论分析草稿是当前证据边界。",
            "convergence_report": "机制链支持审计/财务分析局部角色重排。",
            "external_calibration": "职业整体反转证据不足，应降调为条件性判断。",
        },
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "审计底稿整理" in text
    assert "财务异常检测" in text
    assert "合规报告生成" in text
    assert "管理层讨论分析草稿" in text
    assert "初级审计员" in text
    assert "企业财务分析师" in text
    for leaked in ("法律检索", "合同起草", "案例摘要", "初级律师", "律师", "法律职业", "诉讼", "谈判"):
        assert leaked not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_research_decision_non_foresight_final_uses_generic_packet_synthesis_without_domain_leak():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_RESEARCH_DECISION,
        "query": "未来 3 年，一家中型工业服务公司是否应该进入越南北部电动车电池回收与梯次利用产业链？",
        "output_quality_profile": [executors.PROFILE_EVIDENCE_GROUNDED],
        "excerpts": {
            "L5_deepseek_acceptance": "research_evidence_packet: market and regulatory evidence.",
            "convergence_report": "convergence_report: entry should be staged and conditional.",
            "external_calibration": "calibration_verdict: plausible only as pilot entry; not full heavy-asset entry.",
        },
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "越南北部电动车电池回收" in text
    assert "### evidence_supported" in text
    assert "### reasonable_inference" in text
    assert "### foresight_hypothesis" in text
    assert "触发条件：" in text
    assert "中间机制：" in text
    assert "失效条件或反证信号：" in text
    assert "certainty_level：" in text
    assert "evidence_tier：" in text
    assert "decision_use：" in text
    assert "ADHD" not in text
    assert "家长训练" not in text
    assert "柔术" not in text
    assert "fallback_reasons:" not in text
    assert "executor_model:" not in text
    executors._assert_final_controller_packet_quality(packet, text)


def test_decision_final_quality_blocks_convergence_digest_heading_leak():
    import tools.task_engine_executors as executors

    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    bad = _valid_decision_five_section_report() + "\n\n## convergence_fixed_section_digest\n## key_drivers\nraw digest"

    try:
        executors._assert_artifact_quality(stage, bad)
    except RuntimeError as exc:
        assert "artifact_quality_error:raw_intermediate_dump" in str(exc)
    else:
        raise AssertionError("convergence_fixed_section_digest should be absorbed, not emitted")


def test_decision_final_foresight_template_preserves_user_top5_structure():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": _decision_future_query(),
        "output_quality_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "excerpts": {},
        "convergence_fixed_section_digest": "## key_drivers\nAI.\n## mechanism_chain\ninput variable -> mediating mechanism -> output variable.\n## scenario_branches\nScenario A; Scenario B.\n## counter_signals\nobservable signal.\n## certainty_levels\nhigh / medium / low.\n## uncertainty_boundary\nboundary.",
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "## 未来优势变陷阱 Top5" in text
    assert "## 未来缺陷变优势 Top5" in text
    assert "## danger_flag" in text


def test_decision_final_generic_template_unaffected_without_foresight_profile():
    import tools.task_engine_executors as executors

    packet = {
        "mode": ENGINE_DECISION,
        "query": "这是一个普通 DECISION 任务。是否应该主动干预？",
        "output_quality_profile": [],
        "excerpts": {},
    }

    text = executors._final_controller_report_from_packet(packet)

    assert "## 决策判断" in text
    assert "## 未来优势变陷阱 Top5" not in text
    assert "convergence_fixed_section_digest" not in text


def test_final_controller_gate_block_saves_invalid_and_diagnostic(monkeypatch, tmp_path: Path):
    import tools.task_engine_executors as executors

    monkeypatch.setattr(executors, "_final_controller_report_from_packet", lambda packet: "# 决策任务最终报告\n\n## 结论\n缺少前瞻字段。")
    stage = CANONICAL_STAGES[ENGINE_DECISION][-1]
    packet = {
        "mode": ENGINE_DECISION,
        "query": "未来10年 AI 降低知识获取成本后 ADHD 结构性反转 decision",
        "base_dir": str(tmp_path),
        "output_quality_profile": [executors.PROFILE_FORESIGHT_MECHANISM],
        "excerpts": {},
    }

    try:
        LocalTaskEngineExecutor().run_final_controller_report(stage, packet)
    except RuntimeError as exc:
        message = str(exc)
        assert "output_quality_profile_error" in message or "user_facing_quality" in message
    else:
        raise AssertionError("foresight final gate should block missing required fields")

    invalid = tmp_path / "final_controller_report" / "final_decision_report.invalid.md"
    diagnostic = tmp_path / "final_controller_report" / "final_controller_report.diagnostic.json"
    assert invalid.exists()
    data = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert data["stage_name"] == "final_controller_report"
    assert data["executor_model"] == "Hermes Controller"
    assert "missing_key_drivers" in data["error_summary"] or "user_facing_quality" in data["error_summary"]


def test_final_controller_gate_still_blocks_missing_foresight_fields():
    import tools.task_engine_executors as executors

    errors = executors._quality_profile_errors(
        "# 决策任务最终报告\n\n## 结论\n只有泛泛判断。",
        [executors.PROFILE_FORESIGHT_MECHANISM],
        stage_name="final_controller_report",
    )

    assert "missing_key_drivers" in errors
    assert "missing_mechanism_chain" in errors
    assert "missing_scenario_branches" in errors
    assert "missing_certainty_levels" in errors


def test_task_engine_runner_decision_final_action_does_not_use_research(tmp_path: Path, monkeypatch):
    import tools.task_engine_runner as runner

    calls = []

    def fake_decision_smoke(query, *, base_dir):
        calls.append((query, base_dir))
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
            "full_pipeline_validation": {"valid": True, "stage_count": 10},
            "run": {
                "mode": ENGINE_DECISION,
                "execution_mode": "real-smoke-decision-final",
                "stages": [],
            },
        }

    monkeypatch.setattr(runner, "run_decision_final_smoke", fake_decision_smoke)

    result = json.loads(
        task_engine_runner(
            query="这是一个决策任务。请用 task_engine_runner full 执行决策管线。",
            mode=ENGINE_DECISION,
            action="smoke-decision-final",
            base_dir=str(tmp_path),
        )
    )

    assert result["status"] == "ok"
    assert result["pipeline_status"] == PIPELINE_COMPLETE
    assert result["artifact_dir"] == str(tmp_path)
    assert len(calls) == 1


def test_decision_final_evidence_judge_quality_failure_writes_invalid_artifact_and_diagnostic(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model, timeout_s=None):
            self.last_executor_models[stage.stage_name] = stage.model
            return "decision intelligence report"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            if stage.stage_name == "structure_mapper":
                self.last_executor_models[stage.stage_name] = QWEN72B_ACTUAL_MODEL_DEFAULT
                return "problem_axes\nactor_map\ndecision_questions\nevidence_slots"
            self.last_executor_models[stage.stage_name] = NEMOTRON120B_ACTUAL_MODEL_DEFAULT
            self.last_omlx_diagnostics[stage.stage_name] = {
                "inference_request_sent": True,
                "inference_response_received": True,
                "compact_mode_used": False,
            }
            return "strength_by_claim\n- claim strength without required first section"

    result = run_decision_final_smoke(
        "这是一个决策任务。是否投资早期硬件项目？",
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    invalid_path = tmp_path / "evidence_judge" / "evidence_judge.invalid.md"
    diagnostic_path = tmp_path / "evidence_judge" / "evidence_judge.diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "evidence_judge"
    assert "artifact_quality_error:schema_retry_failed:section_start_mismatch" in result["run"]["stages"][-1]["error"]
    assert invalid_path.exists()
    assert (tmp_path / "evidence_judge" / "evidence_judge.schema_retry_source.invalid.md").exists()
    assert not (tmp_path / "evidence_judge" / "evidence_judge.md").exists()
    assert diagnostic["artifact_quality_error"] == "schema_retry_failed:section_start_mismatch"
    assert diagnostic["first_nonempty_line"] == "strength_by_claim"
    assert diagnostic["invalid_artifact_path"] == str(invalid_path)
    assert diagnostic["inference_request_sent"] is True
    assert diagnostic["inference_response_received"] is True


def _write_valid_new_research_packet_run(root: Path) -> Path:
    for spec in CANONICAL_STAGES[ENGINE_RESEARCH]:
        outputs = planned_outputs(spec, root)
        body = _complete_research_evidence_packet_text() if spec.stage_name == "L5_deepseek_acceptance" else f"{spec.stage_name} artifact"
        for output in outputs.values():
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
    return root / "L5_deepseek_acceptance" / "research_evidence_packet.md"


def _assert_decision_latest_alias_resolves(query: str, tmp_path: Path, monkeypatch) -> None:
    import tools.task_engine_runner as runner

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("HERMES_TASK_ENGINE_ARTIFACT_DIR", raising=False)
    packet = _write_valid_new_research_packet_run(tmp_path / "research_artifacts")
    captured = {}

    def fake_decision_smoke(query, *, base_dir, research_packet_path=None):
        captured["research_packet_path"] = research_packet_path
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
            "full_pipeline_validation": {"valid": True, "stage_count": 10},
            "run": {"mode": ENGINE_DECISION, "execution_mode": "real-smoke-decision-final", "stages": []},
        }

    monkeypatch.setattr(runner, "run_decision_final_smoke", fake_decision_smoke)

    result = json.loads(
        task_engine_runner(
            query=query,
            mode=ENGINE_DECISION,
            action="smoke-decision-final",
            base_dir=str(tmp_path / "decision_artifacts"),
        )
    )

    assert result["status"] == "ok"
    assert captured["research_packet_path"] == str(packet.resolve())


def test_decision_chinese_latest_research_packet_alias_resolves(tmp_path: Path, monkeypatch):
    _assert_decision_latest_alias_resolves("这是一个决策任务。最新研究成果包。", tmp_path, monkeypatch)


def test_decision_chinese_based_on_latest_research_packet_alias_resolves(tmp_path: Path, monkeypatch):
    _assert_decision_latest_alias_resolves("这是一个决策任务。请基于最新研究成果包继续判断。", tmp_path, monkeypatch)


def test_decision_chinese_research_packet_equals_latest_alias_resolves(tmp_path: Path, monkeypatch):
    _assert_decision_latest_alias_resolves("这是一个决策任务。研究成果包=最新。", tmp_path, monkeypatch)


def test_decision_chinese_research_packet_equals_latest_english_alias_resolves(tmp_path: Path, monkeypatch):
    _assert_decision_latest_alias_resolves("这是一个决策任务。研究成果包=latest。", tmp_path, monkeypatch)


def test_decision_chinese_research_packet_path_alias_resolves(tmp_path: Path, monkeypatch):
    import tools.task_engine_runner as runner

    captured = {}
    packet = tmp_path / "research_evidence_packet.md"

    def fake_decision_smoke(query, *, base_dir, research_packet_path=None):
        captured["research_packet_path"] = research_packet_path
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
            "full_pipeline_validation": {"valid": True, "stage_count": 10},
            "run": {"mode": ENGINE_DECISION, "execution_mode": "real-smoke-decision-final", "stages": []},
        }

    monkeypatch.setattr(runner, "run_decision_final_smoke", fake_decision_smoke)

    result = json.loads(
        task_engine_runner(
            query=f"这是一个决策任务。研究成果包：{packet}",
            mode=ENGINE_DECISION,
            action="smoke-decision-final",
            base_dir=str(tmp_path / "decision_artifacts"),
        )
    )

    assert result["status"] == "ok"
    assert captured["research_packet_path"] == str(packet)


def test_decision_latest_research_packet_alias_fails_closed_when_missing(tmp_path: Path, monkeypatch):
    import tools.task_engine_runner as runner

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("HERMES_TASK_ENGINE_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(
        runner,
        "run_decision_final_smoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DECISION must not run without a valid research packet")),
    )

    result = json.loads(
        task_engine_runner(
            query="这是一个决策任务。使用最新研究成果包。",
            mode=ENGINE_DECISION,
            action="smoke-decision-final",
            base_dir=str(tmp_path / "decision_artifacts"),
        )
    )

    assert result["status"] == "blocked"
    assert result["pipeline_status"] == PIPELINE_BLOCKED
    assert result["blocked_stage"] == "research_packet_discovery"
    assert result["blocked_reason"] == "no_valid_new_research_packet_found"


def test_plain_decision_does_not_trigger_research_packet_alias(tmp_path: Path, monkeypatch):
    import tools.task_engine_runner as runner

    _write_valid_new_research_packet_run(tmp_path / "research_artifacts")
    calls = []

    def fake_decision_smoke(query, *, base_dir):
        calls.append((query, base_dir))
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
            "full_pipeline_validation": {"valid": True, "stage_count": 10},
            "run": {"mode": ENGINE_DECISION, "execution_mode": "real-smoke-decision-final", "stages": []},
        }

    monkeypatch.setattr(runner, "run_decision_final_smoke", fake_decision_smoke)

    result = json.loads(
        task_engine_runner(
            query="这是一个决策任务。请直接执行 DECISION full。",
            mode=ENGINE_DECISION,
            action="smoke-decision-final",
            base_dir=str(tmp_path / "decision_artifacts"),
        )
    )

    assert result["status"] == "ok"
    assert len(calls) == 1


def test_final_controller_blocks_without_external_calibration(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_final_controller_report(self, stage, packet):
            raise AssertionError("final_controller_report must not run without L1-L15")

    result = run_research_decision_final_controller_smoke(
        {"stages": []},
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "final_controller_report"
    assert "requires fresh L1-L15 stages" in result["run"]["stages"][-1]["error"]


def test_final_controller_legacy_artifact_cannot_complete(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            actual = {
                "L3_r1_synthesis": R1_ACTUAL_MODEL_DEFAULT,
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            if stage.stage_name == "evidence_judge":
                return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"
            if stage.stage_name == "convergence_report":
                return "divergence_role_summary\nconflicts_to_resolve\nconvergence_decision_framework\nuncertainty_boundaries"
            return "ok"

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_external_calibration(self, stage, packet):
            self.last_executor_models[stage.stage_name] = "GPT Bridge"
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

        def run_final_controller_report(self, stage, packet):
            raise AssertionError("final_controller_report must not run with legacy external_calibration")

    l1_l15 = run_research_decision_l1_l15_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    l1_l15["run"]["stages"][-1]["legacy_contaminated"] = True

    result = run_research_decision_final_controller_smoke(
        l1_l15["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "final_controller_report"
    assert "external_calibration is legacy contaminated" in result["run"]["stages"][-1]["error"]


def test_final_controller_output_forbidden_tool_chain_blocked(tmp_path: Path):
    class FakeExecutor(LocalTaskEngineExecutor):
        def run_agy_gemini(self, stage, prompt, model):
            if stage.stage_name == "L1_gemini_search":
                return {"source_candidates": [{"title": "fake"}]}
            if stage.stage_name == "L4_gemini_audit":
                self.last_executor_models[stage.stage_name] = GEMINI_PRO_HIGH
                return "Gemini audit body"
            self.last_executor_models[stage.stage_name] = GEMINI_HIGH
            return "user_question_map\nresearch_packet_map"

        def run_ddgs(self, stage, queries):
            return [{"query": queries[0], "title": "fake ddgs", "url": "https://example.test/ddgs"}]

        def run_omlx_model(self, stage, model, prompt):
            actual = {
                "L3_r1_synthesis": R1_ACTUAL_MODEL_DEFAULT,
                "convergence_report": R1_ACTUAL_MODEL_DEFAULT,
                "structure_mapper": QWEN72B_ACTUAL_MODEL_DEFAULT,
                "evidence_judge": NEMOTRON120B_ACTUAL_MODEL_DEFAULT,
                "premise_auditor": LLAMA70B_ACTUAL_MODEL_DEFAULT,
                "alternative_generator": GEMMA431B_ACTUAL_MODEL_DEFAULT,
                "insight_harvester": GEMMA431B_ACTUAL_MODEL_DEFAULT,
            }[stage.stage_name]
            self.last_executor_models[stage.stage_name] = actual
            if stage.stage_name == "evidence_judge":
                return "evidence_quality_map\nstrength_by_claim\napplicability_to_user_context\nuncertainty_and_limits"
            if stage.stage_name == "convergence_report":
                return "divergence_role_summary\nconflicts_to_resolve\nconvergence_decision_framework\nuncertainty_boundaries"
            return "ok"

        def run_controller_acceptance(self, stage, packet):
            self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
            return _complete_research_evidence_packet_text()

        def run_external_calibration(self, stage, packet):
            self.last_executor_models[stage.stage_name] = "GPT Bridge"
            return COMPLETE_EXTERNAL_CALIBRATION_FIXTURE

        def run_final_controller_report(self, stage, packet):
            self.last_executor_models[stage.stage_name] = "Hermes Controller"
            return "# ADHD 儿童研究决策报告\nweb_search\napi_call\ncodex_exec"

    l1_l15 = run_research_decision_l1_l15_smoke(ADHD_PROMPT, base_dir=tmp_path, executor=FakeExecutor())
    result = run_research_decision_final_controller_smoke(
        l1_l15["run"],
        query=ADHD_PROMPT,
        base_dir=tmp_path,
        executor=FakeExecutor(),
    )

    assert result["status"] == "blocked"
    assert result["blocked_stage"] == "final_controller_report"
    error = result["run"]["stages"][-1]["error"]
    assert "forbidden raw/tool-chain tokens" in error
    assert "web_search" in error
    assert "api_call" in error
    assert "codex_exec" in error


def test_nightly_runner_parses_completed_stress_stdout(monkeypatch):
    complete = _complete_stress_summary()

    def fake_run(command, *, timeout):
        if "work/stress_task_engine_adhd.py" in command:
            return _child_result(stdout=json.dumps(complete), returncode=0)
        return _child_result(returncode=0)

    monkeypatch.setattr(nightly_runner, "_run", fake_run)
    monkeypatch.setattr(nightly_runner, "_changed_files", lambda: [])
    monkeypatch.setattr(nightly_runner, "_legacy_parity_diff", lambda: {"ok": True, "diff": []})

    result = nightly_runner.run_round(1, 0)

    assert result["stress_status"] == "passed"
    assert result["stress_summary_source"] == "stdout"
    assert result["runner_timeout"] is False
    assert result["child_process_timeout"] is False
    assert result["pipeline_blocked"] is False
    assert result["blocked_stage"] == ""
    assert result["next_action"] == "final_controller_report_passed_pipeline_complete"


def test_nightly_runner_timeout_after_completed_stress_is_runner_timeout_not_pipeline_blocker(monkeypatch):
    complete = _complete_stress_summary()

    def fake_run(command, *, timeout):
        if "work/stress_task_engine_adhd.py" in command:
            return _child_result(
                returncode=124,
                stderr="timeout after 1800s",
                timed_out=True,
            )
        return _child_result(returncode=0)

    monkeypatch.setattr(nightly_runner, "_run", fake_run)
    monkeypatch.setattr(nightly_runner, "_changed_files", lambda: [])
    monkeypatch.setattr(nightly_runner, "_legacy_parity_diff", lambda: {"ok": True, "diff": []})
    monkeypatch.setattr(nightly_runner, "_latest_stress_summary_from_file", lambda: complete)

    result = nightly_runner.run_round(2, 0)

    assert result["stress_status"] == "passed"
    assert result["stress_summary_source"] == "summary_file"
    assert result["runner_timeout"] is True
    assert result["child_process_timeout"] is True
    assert result["child_stderr_tail"].endswith("timeout after 1800s")
    assert result["pipeline_blocked"] is False
    assert result["external_blocker"] is False
    assert result["blocked_stage"] == ""
    assert result["blocked_reason"] == ""
    assert result["next_action"] == "inspect_runner_timeout_after_completed_stress"


def test_nightly_runner_records_real_pipeline_block_from_stress(monkeypatch):
    blocked = _complete_stress_summary()
    blocked["blocked_stage"] = "L2_ddgs_supplement"
    blocked["blocked_reason"] = "DDGS returned no fresh hits"
    blocked["next_action"] = "fix_ddgs_real_search_adapter_then_rerun"

    def fake_run(command, *, timeout):
        if "work/stress_task_engine_adhd.py" in command:
            return _child_result(stdout=json.dumps(blocked), returncode=0)
        return _child_result(returncode=0)

    monkeypatch.setattr(nightly_runner, "_run", fake_run)
    monkeypatch.setattr(nightly_runner, "_changed_files", lambda: [])
    monkeypatch.setattr(nightly_runner, "_legacy_parity_diff", lambda: {"ok": True, "diff": []})

    result = nightly_runner.run_round(1, 0)

    assert result["stress_status"] == "passed"
    assert result["runner_timeout"] is False
    assert result["pipeline_blocked"] is True
    assert result["external_blocker"] is True
    assert result["blocked_stage"] == "L2_ddgs_supplement"
    assert result["blocked_reason"] == "DDGS returned no fresh hits"
    assert result["next_action"] == "stop_external_blocker"


def _complete_stress_summary() -> dict:
    return {
        "round_id": "round_01",
        "pytest": {"passed": True, "returncode": 0},
        "l5_deepseek_acceptance_smoke": {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
        },
        "evidence_judge_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "premise_auditor_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "alternative_generator_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "insight_harvester_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "convergence_report_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "external_calibration_smoke": {
            "status": "ok",
            "pipeline_status": "PIPELINE_INCOMPLETE",
        },
        "final_controller_report_smoke": {
            "status": "ok",
            "pipeline_status": PIPELINE_COMPLETE,
        },
        "blocked_stage": "",
        "blocked_reason": "",
        "any_contract_violation": False,
        "next_action": "final_controller_report_passed_pipeline_complete",
    }


def _child_result(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    timed_out: bool = False,
) -> dict:
    return {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_seconds": 0.1,
        "timed_out": timed_out,
    }


def _make_run(tmp_path: Path, mode: str) -> dict:
    stages = []
    for spec in CANONICAL_STAGES[mode]:
        stage_dir = tmp_path / spec.stage_name
        stage_dir.mkdir()

        if spec.stage_name == "L2_5_codex_evidence_organizer":
            for name in (
                "source_candidates.json",
                "ddgs_gap_sources.json",
                "evidence_runner_001.request.md",
                "evidence_runner_001.request.json",
                "sources.csv",
                "evidence.csv",
                "claims.md",
                "gaps.md",
            ):
                (stage_dir / name).write_text("ok", encoding="utf-8")
            artifact = stage_dir
        elif spec.required_outputs != ("artifact_path",):
            artifact = stage_dir / spec.required_outputs[0]
            body = "ok"
            if spec.stage_name == "L5_deepseek_acceptance":
                body = _complete_research_evidence_packet_text()
            if spec.stage_name == "final_controller_report":
                body = "FINAL CONTROLLER BODY"
            artifact.write_text(body, encoding="utf-8")
        else:
            artifact = stage_dir / "report.md"
            body = "ok"
            if spec.stage_name == "convergence_report":
                body = "R1 CONVERGENCE BODY"
            if spec.stage_name == "final_controller_report":
                body = "FINAL CONTROLLER BODY"
            artifact.write_text(body, encoding="utf-8")

        stages.append(
            {
                "stage_name": spec.stage_name,
                "owner": spec.owner,
                "model": spec.model,
                "artifact_path": str(artifact),
                "created_in_current_run": True,
                "legacy_contaminated": False,
                "valid_for_pipeline": True,
            }
        )
    return {"stages": stages}


def test_final_controller_packet_from_artifacts_preserves_current_run_metadata(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH_DECISION)
    stages = run["stages"][:-1]
    for index, stage in enumerate(stages, start=1):
        stage["run_id"] = "S01_20260622T071900Z_bf888737"
        stage["output_root"] = str(tmp_path)
        stage["prompt_sha256"] = "abc123"
        stage["created_at"] = "2026-06-22T07:19:00Z"
        stage["stage_index"] = index

    packet = _final_controller_packet_from_artifacts(stages, query=ADHD_PROMPT, base_dir=tmp_path)

    assert packet["base_dir"] == str(tmp_path.resolve())
    first = packet["stage_trace"][0]
    assert first["stage_name"] == "L1_gemini_search"
    assert first["created_in_current_run"] is True
    assert first["legacy_contaminated"] is False
    assert first["valid_for_pipeline"] is True
    assert first["run_id"] == "S01_20260622T071900Z_bf888737"
    assert first["output_root"] == str(tmp_path)
    assert first["prompt_sha256"] == "abc123"
    assert first["created_at"] == "2026-06-22T07:19:00Z"
    assert first["stage_index"] == 1
    assert not Path(first["artifact_path"]).is_absolute()


def test_final_controller_packet_from_artifacts_does_not_fabricate_metadata(tmp_path: Path):
    run = _make_run(tmp_path, ENGINE_RESEARCH_DECISION)
    stages = run["stages"][:-1]
    del stages[0]["created_in_current_run"]
    del stages[0]["legacy_contaminated"]

    packet = _final_controller_packet_from_artifacts(stages, query=ADHD_PROMPT, base_dir=tmp_path)

    first = packet["stage_trace"][0]
    assert "created_in_current_run" not in first
    assert "legacy_contaminated" not in first
    assert first["valid_for_pipeline"] is True
