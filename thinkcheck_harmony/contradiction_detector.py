'''矛盾动力论：A指标检测'''
from .utils.text_processing import split_sentences
from .config import ADVERSARIAL_MARKERS

class ContradictionDetector:
    def __init__(self, contradiction_rules: list = None):
        self.contradiction_rules = contradiction_rules or []

    def compute_A(self, text: str) -> float:
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        marker_count = 0
        for sent in sentences:
            for marker in ADVERSARIAL_MARKERS:
                if marker in sent:
                    marker_count += 1
                    break
        marker_density = marker_count / len(sentences)
        rule_match_count = 0
        for term1, term2 in self.contradiction_rules:
            if term1 in text and term2 in text:
                rule_match_count += 1
        rule_score = min(1.0, rule_match_count * 0.3)
        return min(1.0, 0.6 * marker_density + 0.4 * rule_score)
