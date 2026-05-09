"""
全局配置与默认权重
权重默认值来自论文《关系的艺术》协商起点，体现可协商原则
"""

DEFAULT_LAMBDA_U = 0.4
DEFAULT_LAMBDA_D = 0.4
DEFAULT_LAMBDA_A = 0.2

DRIFT_THRESHOLD = 0.65
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ADVERSARIAL_MARKERS = [
    "但是", "然而", "不过", "尽管", "需要注意的是", "必须指出",
    "与此相反", "另一方面", "存在风险", "有待商榷"
]

INTERVENTION_TRIGGERS = {
    "high_A": 0.3,
    "drift_warning": True,
    "low_D": 0.4,
}

FEATURE_FLAGS = {
    "enable_field_probe": False,
    "enable_pattern_resonance": False,
}
