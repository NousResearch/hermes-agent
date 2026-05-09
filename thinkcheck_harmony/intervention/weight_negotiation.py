'''权重协商会议捕获器'''
from .base import BaseHarvester
from ..report import HarmonyReport
from ..config import INTERVENTION_TRIGGERS

class WeightNegotiationHarvester(BaseHarvester):
    def analyze(self, report: HarmonyReport) -> bool:
        return report.A > INTERVENTION_TRIGGERS["high_A"]

    def generate_intervention(self, report: HarmonyReport) -> List[str]:
        micro_experiment = (
            f"🔧 检测到较高对抗性 (A={report.A:.3f})。"
            f"建议启动「权重协商会议」微实验：\n"
            f"   1. 召集相关方（开发者、领域专家、终端用户）\n"
            f"   2. 基于当前评估样本，各方独立拖动 U/D/A 权重滑块\n"
            f"   3. 系统计算妥协解（如取均值或 Nash 解），生成新配置\n"
            f"   4. 用新配置重新评估，观察 H 值变化，记录协商过程"
        )
        return [micro_experiment]

    def get_name(self) -> str:
        return "WeightNegotiationHarvester"
