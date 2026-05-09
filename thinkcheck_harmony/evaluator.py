'''主评估器：实践介入论的接口层，集成建议引擎'''
from typing import List, Optional
from .core import compute_U, compute_D, compute_A, compute_harmony
from .concept_graph import ConceptGraph
from .contradiction_detector import ContradictionDetector
from .report import HarmonyReport
from .intervention.suggestion_engine import SuggestionEngine
from .config import DEFAULT_LAMBDA_U, DEFAULT_LAMBDA_D, DEFAULT_LAMBDA_A

class HarmonyEvaluator:
    def __init__(self,
                 domain: str = "general",
                 lambda_u: float = DEFAULT_LAMBDA_U,
                 lambda_d: float = DEFAULT_LAMBDA_D,
                 lambda_a: float = DEFAULT_LAMBDA_A,
                 custom_terms: Optional[List[str]] = None,
                 enable_suggestions: bool = True):
        self.domain = domain
        self.lambda_u = lambda_u
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.enable_suggestions = enable_suggestions

        from .presets import get_preset
        preset = get_preset(domain)
        self.key_terms = custom_terms if custom_terms else preset.get("terms", [])
        self.contradiction_rules = preset.get("contradiction_rules", [])
        self.detector = ContradictionDetector(self.contradiction_rules)

        self.suggestion_engine = SuggestionEngine() if enable_suggestions else None
        self.last_report: Optional[HarmonyReport] = None

    def evaluate(self, text: str) -> HarmonyReport:
        concept_graph = ConceptGraph(text, self.key_terms)
        U = compute_U(concept_graph)
        D = compute_D(text, concept_graph)
        A = compute_A(text, self.detector)
        H = compute_harmony(U, D, A, self.lambda_u, self.lambda_d, self.lambda_a)
        drift_warnings = concept_graph.get_drift_warnings()

        report = HarmonyReport(
            H=H, U=U, D=D, A=A,
            lambda_weights={"U": self.lambda_u, "D": self.lambda_d, "A": self.lambda_a},
            drift_warnings=drift_warnings,
            micro_details={"term_consistencies": concept_graph.consistency_scores},
            meso_details={"sentence_count": len(concept_graph.sentences)},
            macro_details={"domain": self.domain},
            audit={"evaluator_version": "3.0.0", "domain": self.domain}
        )

        if self.enable_suggestions and self.suggestion_engine:
            report.suggestions = self.suggestion_engine.generate_suggestions(report)

        self.last_report = report
        return report
