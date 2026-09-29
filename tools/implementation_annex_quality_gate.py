"""Independent quality gate for implementation annexes.

Runs deterministically on a generated annex file and returns a structured
verdict.  Can be imported and called from the generator, run as a CLI tool,
or used in a test witness to validate fixture outputs.

Usage (CLI):
    python -m tools.implementation_annex_quality_gate \
        --annex path/to/execution_annex.md \
        --manifest path/to/manifest.json \
        --domain child_adhd_education
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ── helpers ─────────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_profile(domain_key: str,
                  profiles_dir: Path | None = None) -> dict:
    import yaml
    if profiles_dir is None:
        profiles_dir = Path(__file__).resolve().parent.parent / "config"
    profile_path = profiles_dir / "domain_safety_profiles.yaml"
    if not profile_path.exists():
        return {"forbidden_terms": [], "overcommit_guards": [],
                "required_sections": []}
    with open(profile_path, "r", encoding="utf-8") as f:
        all_profiles = yaml.safe_load(f) or {}
    return all_profiles.get(domain_key, {})


def _extract_headings(text: str) -> list[str]:
    """Return stripped heading text from markdown."""
    headings = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("##") or line.startswith("###"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            headings.append(heading.lower())
    return headings


# ── check functions (return list of failures; empty = pass) ───────────────


def check_no_fffd(annex: str, label: str = "annex") -> list[str]:
    """Fail if the text contains U+FFFD replacement character."""
    failures: list[str] = []
    if "\ufffd" in annex:
        lines = [f"{i+1}:{ln}" for i, ln in enumerate(annex.splitlines())
                 if "\ufffd" in ln]
        failures.append(f"{label}: U+FFFD replacement character found: {lines}")
    return failures


def check_internal_terms(annex: str, domain_key: str,
                         profiles_dir: Path | None = None) -> list[str]:
    """Fail if internal evidence-tier labels leak into user-facing text."""
    failures: list[str] = []
    profile = _load_profile(domain_key, profiles_dir)
    forbidden = profile.get("forbidden_terms", [])
    for term in forbidden:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = [(i + 1, ln) for i, ln in enumerate(annex.splitlines())
                   if pattern.search(ln)]
        if matches:
            lines_str = "; ".join(f"L{ln_no}:{ln[:60]}" for ln_no, ln in matches)
            failures.append(f"internal term '{term}' leaked: {lines_str}")
    return failures


def check_no_user_facing_internal_terms(
    annex: str, domain_key: str,
    profiles_dir: Path | None = None,
) -> list[str]:
    """Fail if user-facing internal terms (收敛/外部校准/pipeline/artifact/scenario_branches等) appear."""
    failures: list[str] = []
    profile = _load_profile(domain_key, profiles_dir)
    forbidden = profile.get("user_facing_forbidden", [])
    for term in forbidden:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = [(i + 1, ln) for i, ln in enumerate(annex.splitlines())
                   if pattern.search(ln)]
        if matches:
            lines_str = "; ".join(f"L{ln_no}:{ln[:60]}" for ln_no, ln in matches)
            failures.append(
                f"user-facing internal term '{term}' leaked: {lines_str}"
            )
    return failures


def check_no_empty_placeholders(
    annex: str, domain_key: str,
    profiles_dir: Path | None = None,
) -> list[str]:
    """Fail if empty placeholder text (无明确/未提取/TODO等) appears."""
    failures: list[str] = []
    profile = _load_profile(domain_key, profiles_dir)
    placeholders = profile.get("empty_placeholders", [])
    for term in placeholders:
        pattern = re.compile(re.escape(term))
        matches = [(i + 1, ln) for i, ln in enumerate(annex.splitlines())
                   if pattern.search(ln)]
        if matches:
            lines_str = "; ".join(f"L{ln_no}:{ln[:60]}" for ln_no, ln in matches)
            failures.append(f"empty placeholder '{term}' found: {lines_str}")
    return failures


def check_no_drug_names(annex: str) -> list[str]:
    """Fail if drug names or dosages appear."""
    failures: list[str] = []
    drug_patterns = [
        r"\b(methylphenidate|ritalin|concerta|哌甲酯|利他林|专注达)\b",
        r"\b(dexmethylphenidate|focalin|右哌甲酯)\b",
        r"\b(amphetamine|adderall|vyvanse|苯丙胺|阿德拉)\b",
        r"\b(atomoxetine|strattera|托莫西汀)\b",
        r"\b(guanfacine|intuniv|可乐定|clonidine)\b",
        r"\b\d+\s*mg\b",
        r"\b(用药|服药|剂量|处方药)\b",
    ]
    for pat in drug_patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for i, ln in enumerate(annex.splitlines(), 1):
            if regex.search(ln):
                failures.append(
                    f"drug/dosage term matched '{pat}' at L{i}:{ln[:80]}"
                )
                break
    return failures


def check_no_treatment_instructions(annex: str) -> list[str]:
    """Fail if treatment/medical instructions appear (beyond behavioral)."""
    failures: list[str] = []
    treatment = [
        r"\b(治疗方案|治疗计划|医嘱|临床建议|处方)\b",
        r"\b(必须.*治疗|应当.*服药|建议.*药物治疗)\b",
    ]
    for pat in treatment:
        regex = re.compile(pat)
        for i, ln in enumerate(annex.splitlines(), 1):
            if regex.search(ln):
                failures.append(f"treatment instruction at L{i}:{ln[:80]}")
                break
    return failures


def check_grade3_forward_only(annex: str) -> list[str]:
    """Fail if Grade-3 / third-grade content reads as operational plan."""
    failures: list[str] = []
    operational = [
        r"三年级.*(必须|应当|需要|计划).*(训练|课程|练习|作业)",
        r"三年级.*(具体|详细|每周|每日).*(安排|方案|计划)",
    ]
    for pat in operational:
        regex = re.compile(pat)
        for i, ln in enumerate(annex.splitlines(), 1):
            # Skip lines that are clearly disclaimers / caveats
            if re.search(r"(属于前瞻|不宜|需持续验证|注意|⚠️)", ln):
                continue
            if regex.search(ln):
                failures.append(
                    f"grade 3 over-operationalization at L{i}:{ln[:80]}"
                )
                break
    return failures


def check_overcommit_guards(annex: str, domain_key: str,
                            profiles_dir: Path | None = None) -> list[str]:
    """Fail if overcommit guards are violated."""
    failures: list[str] = []
    profile = _load_profile(domain_key, profiles_dir)
    for guard_entry in profile.get("overcommit_guards", []):
        topic = guard_entry["topic"]
        guard_text = guard_entry["guard"]
        if "BPT" in topic or "家长行为培训" in topic:
            overcommit = re.compile(
                r"(BPT|家长行为培训).*(保证|确保|一定|必然|承诺)(有效|改善|解决)",
                re.IGNORECASE,
            )
            for i, ln in enumerate(annex.splitlines(), 1):
                if overcommit.search(ln):
                    failures.append(
                        f"BPT overcommit at L{i}: {ln[:80]} — {guard_text}"
                    )
                    break
        if "CLAS" in topic:
            clas_oc = re.compile(r"CLAS.*(有效|推荐|首选)", re.IGNORECASE)
            for i, ln in enumerate(annex.splitlines(), 1):
                if clas_oc.search(ln):
                    # Skip if the claim is negated (proper caveat)
                    if re.search(r"(不承诺|不是|不适合|未证实|证据等级较低|谨慎)", ln):
                        continue
                    failures.append(
                        f"CLAS overcommit at L{i}: {ln[:80]} — {guard_text}"
                    )
                    break
    return failures


def check_required_sections(annex: str, domain_key: str,
                            profiles_dir: Path | None = None) -> list[str]:
    """Fail if any required section heading is missing."""
    failures: list[str] = []
    profile = _load_profile(domain_key, profiles_dir)
    required = profile.get("required_sections", [])
    headings = _extract_headings(annex)
    for sec in required:
        if not any(sec in h for h in headings):
            failures.append(f"required section '{sec}' missing from annex")
    return failures


def check_concrete_actions_present(annex: str) -> list[str]:
    """Real content checks beyond heading-only.

    Hard-fail conditions:
    - '未来 2 周启动方案' section has fewer than 4 concrete action items
    - '作业流程' section is missing 作业前 / 作业中 / 作业后
    - '不要做的事' has fewer than 6 items
    - '每周观察表' has fewer than 5 observation indicators
    - '维持、升级、复评信号' missing any of: 维持 / 升级 / 复评
    """
    failures: list[str] = []
    lines = annex.splitlines()

    # Helper: extract content between section headings
    def _section_content(heading_substr: str) -> list[str]:
        in_sec = False
        sec_lines: list[str] = []
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("## ") or stripped.startswith("##\t"):
                h_text = re.sub(r"^#+\s*", "", stripped).strip().lower()
                if heading_substr.lower() in h_text:
                    in_sec = True
                    continue
                if in_sec:
                    break
            if in_sec:
                sec_lines.append(stripped)
        return sec_lines

    # 1. 未来 2 周启动方案 — at least 4 bullet items
    tww = _section_content("未来 2 周启动方案")
    action_bullets = [l for l in tww if l.startswith("- ") or l.startswith("* ")]
    if len(action_bullets) < 4:
        failures.append(
            f"concrete_actions: '未来 2 周启动方案' has {len(action_bullets)} "
            f"action items, need >= 4"
        )

    # 2. 作业流程 — must have 作业前 / 作业中 / 作业后
    hw = _section_content("作业流程")
    hw_text = " ".join(hw)
    for phase in ["作业前", "作业中", "作业后"]:
        if phase not in hw_text and phase not in [l for l in hw]:
            # Check if a heading covers it
            if not any(phase in l for l in hw):
                failures.append(
                    f"concrete_actions: '作业流程' missing '{phase}' phase"
                )

    # 3. 不要做的事 — at least 6 items
    dnd = _section_content("不要做的事")
    dnd_bullets = [
        l for l in dnd
        if l.startswith("- ") or l.startswith("* ") or re.match(r"^\d+\.\s+\*\*", l)
    ]
    if len(dnd_bullets) < 6:
        failures.append(
            f"concrete_actions: '不要做的事' has {len(dnd_bullets)} "
            f"items, need >= 6"
        )

    # 4. 每周观察表 — at least 5 items
    obs = _section_content("每周观察表")
    obs_bullets = [l for l in obs if l.startswith("- ") or l.startswith("* ")]
    if len(obs_bullets) < 5:
        failures.append(
            f"concrete_actions: '每周观察表' has {len(obs_bullets)} "
            f"observation items, need >= 5"
        )

    # 5. 维持、升级、复评信号 — must contain all three
    ess = _section_content("维持、升级、复评信号")
    ess_text = " ".join(ess)
    for signal_type in ["维持", "升级", "复评"]:
        if signal_type not in ess_text:
            failures.append(
                f"concrete_actions: '维持、升级、复评信号' missing '{signal_type}' category"
            )

    return failures


def check_pseudo_threshold(annex: str) -> list[str]:
    """Fail or warn on hard threshold expressions that lack softening.

    Hard thresholds to intercept:
      - 连续 N 周 / N 分钟 / 每周不超过 N 次 / N 个月 / 完成率低于 / 达到多少就升级

    Allowed only if same or adjacent paragraph contains one of:
      - 参考信号 / 不是固定标准 / 不是升级阈值 / 需结合孩子压力、睡眠、学校反馈和专业评估
    """
    failures: list[str] = []
    threshold_patterns = [
        r"连续\s*\d+\s*周",
        r"(坚持|至少|必须)\s*\d+\s*分钟",
        r"每周不超过\s*\d+\s*次",
        r"\d+\s*个月\s*(内|后|以上|以下|左右)",
        r"完成率低于",
        r"达到.*就升级",
    ]
    softeners = [
        "参考信号",
        "不是固定标准",
        "不是升级阈值",
        "需结合孩子压力",
        "需结合.*学校反馈",
    ]

    lines = annex.splitlines()
    full_text_lower = annex.lower()

    for pat in threshold_patterns:
        regex = re.compile(pat)
        for i, ln in enumerate(lines, 1):
            if not regex.search(ln):
                continue
            # Check nearby context (same line + 1 before + 1 after)
            surrounding = ""
            if i > 0:
                surrounding += lines[i - 1] + " "
            surrounding += ln + " "
            if i < len(lines) - 1:
                surrounding += lines[i + 1] + " "
            surrounding_lower = surrounding.lower()

            softened = any(
                re.search(s, surrounding_lower) for s in softeners
            )
            if not softened:
                failures.append(
                    f"pseudo_threshold: hard threshold without softening "
                    f"at L{i}:{ln[:80]}"
                )

    return failures


def check_manifest_contract_distinct(
    manifest_path: Path | None,
    profiles_dir: Path | None = None,
) -> list[str]:
    """Fail if contract sha256 equals final_report sha256 in manifest."""
    failures: list[str] = []
    if not manifest_path or not manifest_path.exists():
        return failures
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return failures
    prov = manifest.get("provenance", {}).get("inputs", {})
    final_sha = prov.get("final_report", {}).get("sha256")
    contract_sha = prov.get("contract", {}).get("sha256")
    if final_sha and contract_sha and final_sha == contract_sha:
        failures.append(
            "manifest: contract sha256 equals final_report sha256 — "
            "contract fixture must be distinct from final report"
        )
    return failures


# ── main gate ─────────────────────────────────────────────────────────────


def run_quality_gate(
    annex_path: Path,
    manifest_path: Path | None = None,
    domain_key: str = "child_adhd_education",
    profiles_dir: Path | None = None,
) -> dict:
    """Run all quality checks and return a structured verdict.

    Returns:
        dict with keys:
            passed (bool)
            checks (list of {name, passed, failures})
            summary (str)
    """
    annex_text = _read(annex_path)
    check_results: list[dict] = []

    def _check(name: str, fn, *args) -> None:
        failures = fn(*args)
        check_results.append({
            "name": name,
            "passed": len(failures) == 0,
            "failures": failures,
        })

    _check("no_ufffd", check_no_fffd, annex_text)
    _check("no_internal_terms", check_internal_terms,
           annex_text, domain_key, profiles_dir)
    _check("no_user_facing_internal_terms", check_no_user_facing_internal_terms,
           annex_text, domain_key, profiles_dir)
    _check("no_empty_placeholders", check_no_empty_placeholders,
           annex_text, domain_key, profiles_dir)
    _check("no_drug_names", check_no_drug_names, annex_text)
    _check("no_treatment_instructions", check_no_treatment_instructions,
           annex_text)
    _check("grade3_forward_only", check_grade3_forward_only, annex_text)
    _check("overcommit_guards", check_overcommit_guards,
           annex_text, domain_key, profiles_dir)
    _check("required_sections", check_required_sections,
           annex_text, domain_key, profiles_dir)
    _check("concrete_actions_present", check_concrete_actions_present,
           annex_text)
    _check("pseudo_threshold", check_pseudo_threshold, annex_text)

    all_passed = all(c["passed"] for c in check_results)
    summary = (
        "ALL CHECKS PASSED" if all_passed
        else f"{sum(1 for c in check_results if not c['passed'])} check(s) failed"
    )

    verdict = {
        "gate_version": "1.0.0",
        "domain": domain_key,
        "annex_path": str(annex_path.resolve()),
        "annex_sha256": _sha256(annex_path),
        "passed": all_passed,
        "checks": check_results,
        "summary": summary,
    }

    if manifest_path:
        verdict["manifest_path"] = str(manifest_path.resolve())
        # Also check manifest contract distinctness
        manifest_check = check_manifest_contract_distinct(
            manifest_path, profiles_dir
        )
        if manifest_check:
            verdict["checks"].append({
                "name": "manifest_contract_distinct",
                "passed": False,
                "failures": manifest_check,
            })
            verdict["passed"] = False
            verdict["summary"] = (
                f"{sum(1 for c in verdict['checks'] if not c['passed'])} "
                f"check(s) failed"
            )

    return verdict


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Implementation annex quality gate"
    )
    parser.add_argument("--annex", required=True, type=Path,
                        help="Path to execution_annex.md")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to manifest.json (optional)")
    parser.add_argument("--domain", default="child_adhd_education",
                        help="Domain profile key")
    args = parser.parse_args()

    verdict = run_quality_gate(
        annex_path=args.annex,
        manifest_path=args.manifest,
        domain_key=args.domain,
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
