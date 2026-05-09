'''评估报告数据结构：体现三重嵌套拓扑'''
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class HarmonyReport:
    H: float
    U: float
    D: float
    A: float
    lambda_weights: Dict[str, float]
    drift_warnings: List[Dict]
    micro_details: Dict[str, Any]
    meso_details: Dict[str, Any]
    macro_details: Dict[str, Any]
    audit: Dict[str, Any]
    suggestions: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            "H": round(self.H, 4),
            "U": round(self.U, 4),
            "D": round(self.D, 4),
            "A": round(self.A, 4),
            "weights": self.lambda_weights,
            "warnings": self.drift_warnings,
            "layers": {
                "micro": self.micro_details,
                "meso": self.meso_details,
                "macro": self.macro_details
            },
            "audit": self.audit,
            "suggestions": self.suggestions or []
        }
