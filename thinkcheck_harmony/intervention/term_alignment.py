'''术语共识工作坊捕获器'''
from typing import List
from .base import BaseHarvester
from ..report import HarmonyReport

class TermAlignmentHarvester(BaseHarvester):
    def analyze(self, report: HarmonyReport) -> bool:
        return len(report.drift_warnings) > 0

    def generate_intervention(self, report: HarmonyReport) -> List[str]:
        suggestions = []
        for warn in report.drift_warnings[:3]:
            term = warn['term']
            micro_experiment = (
                f"📖 术语「{term}」概念漂移 (一致性={warn['consistency']:.3f})。"
                f"建议启动「术语共识工作坊」微实验：\n"
                f"   1. 提取包含「{term}」的上下文片段（见下方）\n"
                f"   2. 邀请 2-3 位相关方对片段进行快速标注（投票或分级）\n"
                f"   3. 形成临时定义共识，并在后续 3 段论述中统一使用\n"
                f"   4. 观察 H 值变化，反馈给系统"
            )
            suggestions.append(micro_experiment)
        return suggestions

    def get_name(self) -> str:
        return "TermAlignmentHarvester"
