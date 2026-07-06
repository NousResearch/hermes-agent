"""Implementation Annex Generator — Phase 1 (ADHD-I user-facing quality).

Takes a final decision report, external calibration, and decision context
contract and produces a user-facing execution annex with provenance manifest.
Output uses the ADHD-I one_pager_v2 section structure with concrete,
non-empty user-facing content.

Phase 1 scope:
  - Domain: child_adhd_education only
  - Deterministic, template-assisted generation (no LLM)
  - Block-level provenance (not line-level)
  - Caveats preserved in user-facing language
  - Quality gate hard-fails on internal term leaks, empty placeholders,
    missing concrete actions, pseudo-thresholds, and contract duplication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── path helpers ──────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── profile loader ────────────────────────────────────────────────────────


def load_domain_profile(domain_key: str,
                        profiles_dir: Path | None = None) -> dict:
    import yaml
    if profiles_dir is None:
        profiles_dir = Path(__file__).resolve().parent.parent / "config"
    profile_path = profiles_dir / "domain_safety_profiles.yaml"
    with open(profile_path, "r", encoding="utf-8") as f:
        all_profiles = yaml.safe_load(f) or {}
    profile = all_profiles.get(domain_key)
    if not profile:
        raise ValueError(
            f"Unknown domain profile '{domain_key}'. "
            f"Available: {list(all_profiles.keys())}"
        )
    return profile


# ── evidence tier mapping ─────────────────────────────────────────────────


def _map_evidence_tier(internal_label: str, mapping: dict) -> str:
    """Map an internal evidence-tier label to its user-facing equivalent."""
    return mapping.get(internal_label, internal_label)


def _replace_evidence_tiers(text: str, mapping: dict) -> str:
    """Replace all evidence-tier labels in text with user-facing equivalents."""
    for internal, user_facing in mapping.items():
        text = re.sub(
            rf"(?<!\w){re.escape(internal)}(?!\w)",
            user_facing,
            text,
        )
    return text


# ── section extraction helpers ────────────────────────────────────────────


def _extract_section(text: str, heading: str) -> str:
    """Extract a markdown section by heading (## or ###).

    Returns content starting from the heading, WITHOUT the heading line itself
    (to avoid leaking internal heading names into user-facing output).
    """
    lines = text.splitlines()
    result: list[str] = []
    in_section = False
    heading_pattern = re.compile(r"^##+\s+")
    target = heading.strip().lower()
    for line in lines:
        stripped = line.strip()
        if heading_pattern.match(stripped):
            h_text = heading_pattern.sub("", stripped).strip().lower()
            if h_text == target:
                in_section = True
                continue
            if in_section:
                break
        if in_section:
            result.append(stripped)
    return "\n".join(result).strip()


# ── provenance record builder ─────────────────────────────────────────────


def _build_provenance(
    inputs: dict[str, Path],
    annex_path: Path,
) -> dict:
    """Build block-level provenance record."""
    input_provenance = {}
    for label, path in inputs.items():
        if path and path.exists():
            input_provenance[label] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        else:
            input_provenance[label] = {
                "path": str(path) if path else None,
                "sha256": None,
                "bytes": None,
            }
    annex_provenance = {
        "path": str(annex_path.resolve()) if annex_path.exists() else None,
        "sha256": _sha256(annex_path) if annex_path.exists() else None,
        "bytes": annex_path.stat().st_size if annex_path.exists() else None,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": "2.0.0-adhd-i",
        "generator_module": "tools.implementation_annex_generator",
        "inputs": input_provenance,
        "outputs": {"execution_annex": annex_provenance},
    }


# ── template-based section generators (ADHD-I one_pager_v2) ──────────────


def _extract_contract_preamble_field(contract_text: str, field: str) -> str:
    """Extract a value from the contract preamble by field prefix."""
    for line in contract_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _gen_current_judgment(
    contract_text: str,
    calibration_text: str,
    report_text: str,
    mapping: dict,
) -> str:
    """Generate '当前判断' section (replaces old 目的与依据 + 核心建议)."""
    topic = _extract_contract_preamble_field(contract_text, "task_topic")
    topic_user = _replace_evidence_tiers(topic, mapping) if topic else "ADHD-I 儿童教育支持决策"

    certainty = _extract_section(contract_text, "certainty_levels")
    if certainty:
        certainty = _replace_evidence_tiers(certainty, mapping)
    if not certainty:
        certainty = (
            "- 是否要主动干预：有证据支持（ADHD-I 执行功能缺陷的认知框架有科学依据）\n"
            "- 干预强度：基于合理推断（需根据个体情况动态调整）\n"
            "- 家长行为培训：基于合理推断（BPT 作为组成部分有支持）\n"
            "- 三年级准备：前瞻性推断（缺乏直接证据，需持续验证）"
        )

    agreement = _extract_section(calibration_text, "agreement_points")
    if agreement:
        agreement = _replace_evidence_tiers(agreement, mapping)

    return f"""## 1. 当前判断

**决策主题**：{topic_user}

### 干预必要性

根据评估和证据，孩子 ADHD 注意力缺陷型（ADHD-I）的执行功能缺陷需要主动关注和适当支持。主动干预（行为训练、环境调整）可以帮助补偿这些缺陷，改善学业表现和生活质量。

### 证据确定性

{certainty}

### 已确认的共识

{agreement if agreement else "根据国际指南和中国临床实践共识，ADHD-I 需要多模态综合管理方案，行为干预与环境调整是基础组成部分。"}

> ⚠️ 本指南是对上述决策判断的执行补充，不替代专业医疗评估。具体干预方案请咨询主治医师或专业评估机构。"""


def _gen_first_two_weeks(
    report_text: str,
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '未来 2 周启动方案' section."""
    mechanism = _extract_section(contract_text, "mechanism_chain")
    if mechanism:
        mechanism = _replace_evidence_tiers(mechanism, mapping)

    return f"""## 2. 未来 2 周启动方案

以下动作建议在 **2 周内**启动并建立基础。这些是参考信号，不是固定标准，需结合孩子压力、睡眠、学校反馈和专业评估灵活调整。

### 第 1 周

- **完成或预约专业评估**：如尚未在具备 ADHD 诊断资质的专业机构完成评估，本周内预约。
- **建立基础观察日志**：每天记录注意力表现（专注时长）、作业完成时间、情绪状态和睡眠时长。
- **与班主任进行首次沟通**：说明孩子情况，了解课堂表现，协商初步支持方案。
- **调整家庭作业环境**：减少干扰源（电视、玩具），固定作业时间和位置。

### 第 2 周

- **实施简单结构化管理**：使用可视化清单（每日任务表）帮助孩子建立作业前准备习惯。
- **记录初步效果**：对比第 1 周和第 2 周的作业完成效率和情绪变化。
- **与学校确认支持措施**：获取教师对课堂调整（座位安排、分步指令）的反馈。
- **整理观察记录**：为后续评估和调整提供基础数据。"""


def _gen_homework_flow(
    report_text: str,
    mapping: dict,
) -> str:
    """Generate '作业流程' section with before/during/after phases."""
    return f"""## 3. 作业流程

作业是 ADHD-I 孩子日常最大的执行功能挑战。以下流程帮助降低启动难度、维持专注、减少对抗。这些是参考信号，不是固定标准，需结合孩子压力、睡眠、学校反馈和专业评估灵活调整。

### 作业前（准备阶段）

1. **固定时间**：每天同一时间开始作业（如放学休息 30 分钟后），建立条件反射。
2. **环境准备**：清理桌面无关物品，准备好所有需要用品（书本、文具）。
3. **任务分解**：和孩子一起将作业拆分为 3–5 个小单元，每单元预计 10–15 分钟。
4. **预估时间**：和孩子确认每个小单元预计完成时间，建立时间意识。
5. **约定休息**：每完成一个单元允许 3–5 分钟短暂休息，避免一次性过长作业。

### 作业中（执行阶段）

1. **定时器辅助**：使用可视化计时器（非手机）帮助孩子感知时间流逝。
2. **分段推进**：每次只专注一个小单元，完成后标记并奖励短暂休息。
3. **正向反馈**：对专注表现给予即时肯定（"你刚才专心做了 10 分钟，做得很好"）。
4. **最小干预**：孩子走神时先用非语言信号（轻敲桌面、指计时器）提示，避免频繁打断。
5. **记录中断原因**：如果无法继续，记录原因（疲劳？难度？情绪？）而非强迫完成。

### 作业后（收尾阶段）

1. **检查整理**：和孩子一起检查作业完成情况，整理书包和桌面。
2. **简短回顾**：用 1-2 句话评价今天作业过程（"今天数学单元做了 15 分钟，有进步"）。
3. **正向结束**：以积极活动结束（阅读、游戏），不以作业表现影响亲子关系。
4. **记录数据**：家长记录当天的作业完成时间、情绪变化和专注表现，用于观察趋势。"""


def _gen_parent_behavior_support(
    report_text: str,
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '家长行为支持动作' section (from stakeholder parents)."""
    counter_signals = _extract_section(contract_text, "counter_signals")
    if counter_signals:
        counter_signals = _replace_evidence_tiers(counter_signals, mapping)

    return f"""## 4. 家长行为支持动作

### 日常支持

- **观察与记录**：关注孩子的注意力表现、作业完成情况和情绪变化，记录具体行为模式，而非标签化评价。
- **环境调整**：减少干扰源，建立固定的作息和作业时间表，使用可视化日程帮助孩子建立预期。
- **正向引导优先**：对良好行为给予即时、具体的正向反馈（"你今天专注了 15 分钟，很好"），避免过度关注负面行为。
- **参与家长培训**：如有条件，参加循证家长行为培训（BPT）项目。BPT 是组成部分，不是单独充分方案，不承诺固定周期或效果。

### 情绪管理

- **识别疲劳信号**：孩子易怒、抗拒、发呆时，先评估是否为疲劳或感官过载，而非直接要求"专心"。
- **避免对抗升级**：当孩子已经情绪激动时，暂停作业和要求，先处理情绪再处理任务。
- **一致性**：保持规则和期望的一致性（父母之间、家庭与学校之间），减少孩子的不确定感。

### 自我照顾

- **家长压力和执行力直接影响干预效果**：照顾好自己的作息和情绪管理能力，必要时寻求支持。
- **记录自己的观察和困惑**：定期与专业评估机构沟通，获取指导。

### 需要谨慎的事项

{counter_signals if counter_signals else "- 行为干预需长期坚持，短期内可能看不到显著变化，不因此过早放弃。"}"""


def _gen_school_communication(
    report_text: str,
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '学校沟通策略' section with concrete scripts."""
    return f"""## 5. 学校沟通策略

有效的家校协作是 ADHD-I 支持的关键组成部分。以下策略和参考话术帮助家长与学校建立合作关系。这些是参考信号，不是固定标准，需结合孩子压力、睡眠、学校反馈和专业评估灵活调整。

### 首次沟通（建立合作）

**目的**：说明孩子情况，争取学校理解和支持。

**参考话术**：
> "老师您好，我家孩子最近在专注力和作业完成方面遇到一些挑战。医生/评估机构建议我们做一些环境调整和行为支持来帮助他/她。我想了解一下孩子在课堂上的表现，也想听听您有什么建议。"

**要点**：
- 简明说明情况，不过度描述诊断细节
- 表达合作态度而非"要求"
- 询问教师观察到的具体情况

### 课堂调整协商

**目的**：争取合理的课堂支持措施。

**参考话术**：
> "我们了解到 ADHD 孩子在前排座位和分步指令方面会更有帮助。如果您觉得合适，能否让孩子坐在靠前的位置？在他/她走神时，轻敲桌面或提前给个提示可能会有帮助。"

**可协商的调整**：
- **座位安排**：靠前座位，减少干扰
- **指令方式**：分步骤、书面 + 口头
- **任务拆分**：大作业拆分为小单元
- **反馈方式**：正向强化优先，避免公开批评

**注意**：每日报告卡（DRC）等西方模型在中国高密度班级中的可行性需协商验证。

### 定期沟通机制

- **频率**：建议每周或每两周简短沟通（微信/短信），每 1–2 月深入沟通
- **内容**：课堂表现 + 作业完成情况 + 社交互动
- **记录**：保持简单书面记录（学校反馈笔记），用于趋势判断

### 应对困难情况

- 如果学校不配合或资源有限：优先保证家庭支持到位，减少学校期望
- 如果孩子在学校出现显著行为问题：先了解原因（疲劳？社交冲突？），而非直接要求学校改变
- 学校干预需具体协商，不承诺统一方案"""


def _gen_weekly_observation(
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '每周观察表' section (from monitoring indicators)."""
    key_drivers = _extract_section(contract_text, "key_drivers")
    if key_drivers:
        key_drivers = _replace_evidence_tiers(key_drivers, mapping)

    if not key_drivers:
        key_drivers = (
            "- 注意力表现：能否在无干扰环境下维持专注（时长和深度）\n"
            "- 作业完成效率：完成相同作业所需时间和独立程度\n"
            "- 情绪与压力：对学业要求的情绪反应强度和恢复时间\n"
            "- 学校反馈：教师对课堂参与和任务完成的评价\n"
            "- 社交适应：同伴互动质量与参与度\n"
            "- 睡眠质量：入睡时间、夜间醒来频率、早晨起床状态"
        )

    return f"""## 6. 每周观察表

每周回顾以下指标，用于判断支持方案是否需要调整。这些是参考信号，不是固定标准，不是升级阈值，需结合孩子压力、睡眠、学校反馈和专业评估综合判断。

### 注意力与作业

{key_drivers}

### 情绪与行为

- 对抗行为频率和强度：作业启动困难？过度拒绝？
- 挫败后的恢复时间：情绪平复需要多久？
- 学校表现：教师反馈的课堂参与度和行为表现

### 作息与健康

- 睡眠：入睡时间、睡眠时长、早晨起床状态
- 饮食：是否有明显食欲变化

### 观察记录方法

- 每天花 5 分钟记录关键点（可固定在作业后）
- 使用简单符号（✓/△/✗）而非长篇文字
- 每周日回顾本周趋势，判断是否需要调整

### 预警信号（建议及时与专业机构沟通）

- 学业表现持续下降或出现显著情绪崩溃
- 睡眠和作息稳定性明显恶化
- 对学校和家庭环境的负面情绪增加
- 与同龄人的社交互动明显减少"""


def _gen_escalation_signals(
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '维持、升级、复评信号' section (from scenario branches)."""
    scenarios = _extract_section(contract_text, "scenario_branches")
    if scenarios:
        scenarios = _replace_evidence_tiers(scenarios, mapping)

    if scenarios:
        # Convert scenario_branches raw text into structured headings
        scenario_lines = scenarios.splitlines()
        processed = []
        for ln in scenario_lines:
            stripped = ln.strip()
            # Remove leading numbers and bold formatting
            if stripped.startswith("1.") or stripped.startswith("2.") or stripped.startswith("3."):
                # Handle "1. **低强度干预（观察与支持）**："
                stripped = re.sub(r"^\d+\.\s*", "", stripped)
                processed.append(stripped)
            elif stripped.startswith("- **触发条件**") or stripped.startswith("- **核心策略**"):
                processed.append(f"  {stripped}")
            else:
                processed.append(stripped)
        scenarios_text = "\n".join(processed)
    else:
        scenarios_text = ""

    heading_mark = "### 方案参考"
    scenarios_block = ""
    if scenarios_text:
        scenarios_block = heading_mark + "\n\n" + scenarios_text

    return f"""## 7. 维持、升级、复评信号

以下信号帮助家长判断当前支持方案的有效性。这些是参考信号，不是固定标准，不是升级阈值，需结合孩子压力、睡眠、学校反馈和专业评估综合判断。

### 维持信号（当前方案有效，继续执行）

- 作业完成时间稳定或略有改善
- 情绪对抗明显减少，作业启动阻力降低
- 学校反馈积极或保持稳定
- 孩子情绪和睡眠状态稳定
- 家长自身执行力和压力在可控范围

### 升级信号（需要增加支持强度）

- 作业完成时间持续增加（2 周以上趋势）
- 情绪对抗显著增加，每天作业启动都有严重困难
- 学校反馈明显变差或出现行为报告
- 睡眠质量持续下降或出现明显的早晨抗拒上学
- 孩子出现自我否定表达（"我什么都做不好"）
- 家长执行力下降，无法维持当前支持方案

### 复评信号（需要重新评估干预方案）

- 学业表现显著下降（成绩从原来水平大幅下滑）
- 出现明显的情绪崩溃或行为问题（如激烈对抗、逃学）
- 睡眠严重不足或饮食出现显著变化
- 社交互动显著减少或完全回避同伴
- 家长感觉当前所有方法都无效，持续 4 周以上

{scenarios_block}
"""

def _gen_do_not_do(
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '不要做的事' section (replaces old 注意事项与边界)."""
    return f"""## 8. 不要做的事

### 不要做的行为

1. **不要拿孩子和其他孩子比较**：每个 ADHD-I 孩子的表现和进步节奏不同，比较只会增加孩子和家长的挫败感。
2. **不要在情绪对抗中坚持完成作业**：当孩子已经情绪激动时，暂停作业要求，先处理情绪再处理任务。强迫完成会加深对抗模式。
3. **不要一次性布置过多任务**：将任务分解为小单元（参考作业流程），避免一次性要求"把作业全部做完再休息"。
4. **不要过度聚焦"专心"的指令**：频繁要求"专心"对孩子没有帮助，应该提供具体的行为指引（"先看第一题""拿出数学书"）。
5. **不要用惩罚替代支持**：惩罚不能改善执行功能缺陷，反而会增加孩子的挫败和逃避行为。
6. **不要忽视自己的情绪和压力**：家长的焦虑、疲劳和挫败会直接影响孩子。照顾好自己的状态是有效支持的前提。
7. **不要期望短期见效**：行为干预需要长期坚持，短期内可能看不到显著变化，不因此过早放弃。

### 不要做的干预

1. **本指南不提供医疗干预方案**，不含药物名称、剂量或治疗指令。药物评估需由具备 ADHD 诊断资质的专业机构完成。
2. **运动是辅助支持**，不是替代性干预，不承诺改善核心 ADHD 症状。
3. **CLAS 方法证据等级较低**，不承诺其有效性，请谨慎对待相关培训项目。
4. **干预强度是连续谱**，不宜将低/中/高三分法视为固定层级。方案应根据孩子的实际响应动态调整。
5. **学校支持因校而异**，需具体协商，无法承诺统一标准。"""


def _gen_evidence_boundary(
    contract_text: str,
    mapping: dict,
) -> str:
    """Generate '证据边界' section (replaces certainty + uncertainty)."""
    uncertainty = _extract_section(contract_text, "uncertainty_boundary")
    if uncertainty:
        uncertainty = _replace_evidence_tiers(uncertainty, mapping)

    if not uncertainty:
        uncertainty = (
            "- ADHD-I 对行为干预的个体化响应差异\n"
            "- 干预模式在中国环境中的适用性\n"
            "- 三年级转折点的执行功能赤字恶化风险"
        )

    return f"""## 9. 证据边界

### 我们知道的（有证据支持）

- ADHD-I 的核心是执行功能缺陷，行为训练和环境调整是有效的支持手段（国际指南共识）
- 多模态主动管理方案优于单一干预（有证据支持）
- 家长行为培训（BPT）是基础干预模块，重要但非单独充分
- 结构化的环境支持（固定作息、任务分解、正向反馈）可以改善日常功能

### 我们不确定的（需持续验证）

- 干预强度的具体阈值（低/中/高的划分是基于合理推断，不是精确标准）
- 三年级转折点的具体影响程度（属于前瞻性推断，缺乏直接证据）
- 西方干预模式（如每日报告卡）在中国学校环境中的实际可行性
- 个体化响应差异（每个孩子的反应不同，需动态调整）

### 必须注意的边界

{uncertainty}

### 执行效果的关键变量

- 家庭执行力：家长能否持续一致地执行支持方案
- 学校配合度：学校是否愿意和能够提供合理的课堂调整
- 孩子压力/挫败反应：孩子的个体差异决定了对干预的接受度
- 睡眠与作息稳定性：睡眠质量直接影响执行功能和情绪调节
- 专业评估可及性：专业的评估和指导是有效干预的基础
- 家校沟通成本：教师时间和精力有限，沟通频率和方式需现实可行"""


# ── main generator ────────────────────────────────────────────────────────


def generate_annex(
    final_report: Path,
    external_calibration: Path,
    contract: Path,
    output_dir: Path,
    domain: str = "child_adhd_education",
    profiles_dir: Path | None = None,
    verify_quality: bool = True,
) -> dict[str, Any]:
    """Generate an execution annex and provenance manifest.

    Args:
        final_report: Path to the final decision report (final_decision_report.md)
        external_calibration: Path to external calibration (.md)
        contract: Path to the decision context contract (.md or .json)
        output_dir: Directory to write outputs to
        domain: Domain profile key (default: child_adhd_education)
        profiles_dir: Override config directory (default: ../config/)
        verify_quality: Whether to run quality gate after generation (default: True)

    Returns:
        dict with keys: annex_path, manifest_path, annex_sha256, manifest,
        quality_verdict (if verify_quality=True)
    """
    # Resolve paths
    final_report = final_report.resolve()
    external_calibration = external_calibration.resolve()
    contract = contract.resolve()
    output_dir = output_dir.resolve()

    # Load domain profile
    profile = load_domain_profile(domain, profiles_dir)
    mapping = profile.get("evidence_tier_mapping", {})

    # Read inputs
    report_text = _read(final_report)
    calibration_text = _read(external_calibration)
    contract_text = _read(contract)

    # Build annex sections
    sections = []

    sections.append(f"# {profile.get('annex_title', 'ADHD-I 家庭执行指南')}\n")
    sections.append(
        f"> 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"> 领域：{profile.get('label', domain)}\n"
        f"> 生成器版本：2.0.0-adhd-i\n"
    )

    # 9 ADHD-I specific sections
    sections.append(
        _gen_current_judgment(contract_text, calibration_text, report_text, mapping)
    )
    sections.append(
        _gen_first_two_weeks(report_text, contract_text, mapping)
    )
    sections.append(
        _gen_homework_flow(report_text, mapping)
    )
    sections.append(
        _gen_parent_behavior_support(report_text, contract_text, mapping)
    )
    sections.append(
        _gen_school_communication(report_text, contract_text, mapping)
    )
    sections.append(
        _gen_weekly_observation(contract_text, mapping)
    )
    sections.append(
        _gen_escalation_signals(contract_text, mapping)
    )
    sections.append(
        _gen_do_not_do(contract_text, mapping)
    )
    sections.append(
        _gen_evidence_boundary(contract_text, mapping)
    )

    annex_text = "\n\n".join(sections)

    # Write annex
    annex_path = output_dir / "execution_annex.md"
    _write(annex_path, annex_text)

    # Build and write manifest
    inputs = {
        "final_report": final_report,
        "external_calibration": external_calibration,
        "contract": contract,
    }
    provenance = _build_provenance(inputs, annex_path)
    manifest = {
        "domain": domain,
        "domain_label": profile.get("label", domain),
        "provenance": provenance,
    }

    manifest_path = output_dir / "manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))

    result: dict[str, Any] = {
        "annex_path": str(annex_path),
        "manifest_path": str(manifest_path),
        "annex_sha256": _sha256(annex_path),
        "manifest": manifest,
    }

    # Run quality gate if requested
    if verify_quality:
        from tools.implementation_annex_quality_gate import run_quality_gate
        quality_verdict = run_quality_gate(
            annex_path=annex_path,
            manifest_path=manifest_path,
            domain_key=domain,
            profiles_dir=profiles_dir,
        )
        result["quality_verdict"] = quality_verdict

    return result


# ── CLI entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an implementation annex from decision artifacts"
    )
    parser.add_argument("--final-report", required=True, type=Path,
                        help="Path to final_decision_report.md")
    parser.add_argument("--external-calibration", required=True, type=Path,
                        help="Path to external_calibration.md")
    parser.add_argument("--contract", required=True, type=Path,
                        help="Path to decision_context_contract.md")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write outputs")
    parser.add_argument("--domain", default="child_adhd_education",
                        help="Domain profile key")
    parser.add_argument("--skip-quality-gate", action="store_true",
                        help="Skip automatic quality gate run")
    args = parser.parse_args()

    result = generate_annex(
        final_report=args.final_report,
        external_calibration=args.external_calibration,
        contract=args.contract,
        output_dir=args.output_dir,
        domain=args.domain,
        verify_quality=not args.skip_quality_gate,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    annex_path = Path(result["annex_path"])
    print(f"\n--- Annex written to: {annex_path.resolve()}")
    print(f"    SHA256: {result['annex_sha256']}")
    print(f"    Size:   {annex_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
