"""KENSEI CUSTOM — delegate verify/synthesis primitives (ported from fork delegate_tool).

run_kensei_primitives() executes the skeptic-verification (single-task) and batch-synthesis
primitives against finalized child results, using upstream's _run_single_child + child builders.
All failures are contained: any exception degrades to no enrichment (logged, never raised).
"""
import json
import logging
import time

logger = logging.getLogger("tools.delegate_tool")


def run_kensei_primitives(*, results, batch, parent_agent):
    """Run verify (n==1) and synthesize (n>1) primitives; returns dict of extras or None."""
    if not results:
        return None
    from tools.delegate_tool_config import _get_synthesis_enabled, _get_verify_enabled

    task_list = batch.task_list
    n_tasks = len(task_list)
    if n_tasks == 0:
        return None
    task_labels = [t.get("goal", "")[:40] for t in task_list]

    # Dispatch args ride on the batch (fork carried them as locals).
    verify = bool(getattr(batch, "verify", False))
    verify_rubric = getattr(batch, "verify_rubric", None)
    synthesize = bool(getattr(batch, "synthesize", False))
    synthesis_prompt = getattr(batch, "synthesis_prompt", None)
    profile = getattr(batch, "profile", None)
    profile_content = getattr(batch, "profile_content", None)
    effective_max_iter = getattr(batch, "max_iterations", None) or 250

    extras: dict = {}

    # ── verification primitive (single-task mode only) ──
    _effective_verify = bool(verify or verify_rubric)
    if _effective_verify and n_tasks == 1 and _get_verify_enabled():
        try:
            entry = results[0]
            producer_summary = entry.get("summary") or ""
            from tools.delegate_tool_results import _extract_finding
            finding = _extract_finding(producer_summary)
            if not finding:
                logger.warning(
                    "delegate_task: verify=True but could not extract finding "
                    "from child output (%d chars); skipping verification.",
                    len(producer_summary),
                )
            else:
                rubric = verify_rubric or (
                    "Evaluate whether this finding is factually correct, logically sound, "
                    "and complete. Identify any unsupported claims."
                )
                skeptic_goal = (
                    "You are a skeptical reviewer. Your job is to find flaws "
                    "in the following finding. Be rigorous and evidence-based.\n\n"
                    f"FINDING TO EVALUATE:\n{finding}\n\n"
                    f"RUBRIC:\n{rubric}\n\n"
                    "Respond with a structured verdict:\n"
                    '{"claim": "<restated claim>", '
                    '"survived": true|false, '
                    '"verdict": "<your assessment>", '
                    '"reasoning": "<evidence-based reasoning>", '
                    '"confidence": 0.0-1.0}'
                )
                creds = getattr(batch, "creds", None) or {}
                from tools.delegate_tool import _build_child_agent, _run_single_child
                skeptic_result = _run_single_child(
                    task_index=0,
                    goal=skeptic_goal,
                    child=_build_child_agent(
                        task_index=len(task_list),
                        goal=skeptic_goal,
                        context=None,
                        toolsets=None,
                        model=creds.get("model"),
                        max_iterations=effective_max_iter,
                        task_count=1,
                        parent_agent=parent_agent,
                        override_provider=creds.get("provider"),
                        override_base_url=creds.get("base_url"),
                        override_api_key=creds.get("api_key"),
                        override_api_mode=creds.get("api_mode"),
                    ),
                    parent_agent=parent_agent,
                )
                sk_summary = skeptic_result.get("summary") or ""
                cleaned = sk_summary.strip()
                for fence in ("```json", "```"):
                    if cleaned.startswith(fence):
                        cleaned = cleaned[len(fence):]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                try:
                    sk_parsed = json.loads(cleaned) if cleaned.startswith("{") else {"raw": sk_summary, "survived": None}
                except json.JSONDecodeError:
                    sk_parsed = {"raw": sk_summary, "survived": None}
                entry["verification"] = {
                    "verified": sk_parsed.get("survived"),
                    "skeptic_verdict": sk_parsed.get("verdict", ""),
                    "skeptic_reasoning": sk_parsed.get("reasoning", ""),
                    "skeptic_confidence": sk_parsed.get("confidence", 0.0),
                }
        except Exception as exc:
            logger.warning("delegate_task: verification failed: %s", exc, exc_info=True)
            if results and "verification" not in results[0]:
                results[0]["verification"] = {"verified": None, "error": f"Verification agent failed: {exc}"}

    # ── synthesis primitive (batch mode only, n > 1) ──
    synth_output = None
    missing_tasks = []
    if synthesize and n_tasks > 1 and _get_synthesis_enabled():
        try:
            summaries = []
            for entry in results:
                if entry.get("status") == "completed" and entry.get("summary"):
                    summaries.append(entry["summary"])
                elif entry.get("status") != "completed":
                    idx = entry.get("task_index", -1)
                    gl = task_labels[idx] if 0 <= idx < len(task_labels) else f"Task {idx}"
                    missing_tasks.append(gl)

            if not summaries:
                logger.warning("delegate_task: synthesize=True but no children produced usable summaries")
            else:
                synth_prompt = synthesis_prompt or (
                    "Synthesize the following findings from {n} parallel "
                    "sub-agents into a single coherent output. Resolve "
                    "contradictions. Deduplicate overlapping content. "
                    "Flag confidence levels. Treat claims without "
                    "verifiable receipts (file paths, diff stats) as "
                    "unverified — say so instead of repeating them as "
                    "fact.\n\n"
                    "{summary_block}\n\n"
                    "Provide your synthesis as a structured report."
                )
                summary_block = "\n---\n".join(
                    f"[TASK {i+1}]\n{s}" for i, s in enumerate(summaries)
                )
                had_block_token = "{summary_block}" in synth_prompt
                synth_prompt = synth_prompt.replace(
                    "{summary_block}", summary_block
                ).replace("{n}", str(len(summaries)))
                if not had_block_token:
                    synth_prompt += f"\n\n{summary_block}"

                if missing_tasks:
                    synth_prompt += (
                        f"\n\nNOTE: {len(missing_tasks)} task(s) did "
                        f"not complete or produced no output: "
                        f"{', '.join(missing_tasks)}. Account for these "
                        f"gaps — do not fabricate findings."
                    )

                creds = getattr(batch, "creds", None) or {}
                from tools.delegate_tool import _build_child_agent, _run_single_child
                synth_result = _run_single_child(
                    task_index=len(task_list),
                    goal=synth_prompt,
                    child=_build_child_agent(
                        task_index=len(task_list),
                        goal=synth_prompt,
                        context=None,
                        toolsets=None,
                        model=creds.get("model"),
                        max_iterations=effective_max_iter,
                        task_count=1,
                        parent_agent=parent_agent,
                        override_provider=creds.get("provider"),
                        override_base_url=creds.get("base_url"),
                        override_api_key=creds.get("api_key"),
                        override_api_mode=creds.get("api_mode"),
                        override_request_overrides=creds.get("request_overrides"),
                        override_max_tokens=creds.get("max_output_tokens"),
                        profile=profile,
                        profile_content=profile_content,
                    ),
                    parent_agent=parent_agent,
                )
                synth_output = synth_result.get("summary") or None
        except Exception as exc:
            logger.warning("delegate_task: synthesis failed: %s", exc, exc_info=True)

    if synthesize and n_tasks > 1 and _get_synthesis_enabled():
        if synth_output is not None:
            extras["synthesis"] = synth_output
        if missing_tasks:
            extras["missing_tasks"] = missing_tasks
    return extras or None
