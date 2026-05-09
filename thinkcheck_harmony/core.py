'''谐振调谐论：和谐度计算引擎 H = λU·U + λD·D - λA·A'''
import numpy as np
from .concept_graph import ConceptGraph
from .contradiction_detector import ContradictionDetector

def compute_U(concept_graph: ConceptGraph) -> float:
    return concept_graph.get_avg_consistency()

def compute_D(text: str, concept_graph: ConceptGraph) -> float:
    positions = concept_graph.get_first_occurrence_positions()
    if len(positions) <= 1:
        return 0.0
    positions_normalized = [p / len(text) for p in positions]
    std = np.std(positions_normalized)
    return min(1.0, std * 2.5)

def compute_A(text: str, detector: ContradictionDetector) -> float:
    return detector.compute_A(text)

def compute_harmony(U: float, D: float, A: float,
                     lambda_u: float, lambda_d: float, lambda_a: float) -> float:
    return lambda_u * U + lambda_d * D - lambda_a * A
